#!/usr/bin/env python3
"""
PRAMAAN - train.py: reproducible model training for all five detections.

    python src/train.py paysim        [--stage features|fit|all]
    python src/train.py ieee_cis      [--stage features|fit|all]
    python src/train.py rba
    python src/train.py cmu_keystroke
    python src/train.py cert_insider  [--stage features|fit|all]

Design rules:
  * temporal train/test splits (no look-ahead leakage)
  * fixed seeds everywhere (SEED=42)
  * every model writes a scores table to data/processed/<ds>_scores.parquet
    (y_true + one column per model variant) consumed by evaluate.py
  * ablations are first-class: PaySim with/without balance-derived features,
    IEEE-CIS with/without device/identity columns, RBA with/without IP reputation.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest

SEED = 42
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
PROC.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(SEED)


def hgb(**kw) -> HistGradientBoostingClassifier:
    p = dict(max_iter=200, learning_rate=0.1, max_leaf_nodes=31,
             early_stopping=False, random_state=SEED,
             categorical_features="from_dtype")
    p.update(kw)
    return HistGradientBoostingClassifier(**p)


# ---------------------------------------------------------------- PaySim (#3)
PAYSIM_FEATS = PROC / "paysim_features.parquet"
BEHAV_COLS = ["hour", "log_amount", "amount", "type", "orig_txn_idx",
              "dest_in_deg", "dest_is_merchant"]
BALANCE_COLS = ["oldbalanceOrg", "newbalanceOrig", "oldbalanceDest",
                "newbalanceDest", "err_orig", "err_dest", "orig_emptied",
                "dest_zero_before", "amt_ratio", "frac_orig_moved",
                "amount_eq_oldorig"]


def paysim_features() -> None:
    t0 = time.time()
    dt = {"step": "int32", "type": "category", "amount": "float64",
          "nameOrig": "string", "oldbalanceOrg": "float64",
          "newbalanceOrig": "float64", "nameDest": "string",
          "oldbalanceDest": "float64", "newbalanceDest": "float64",
          "isFraud": "int8", "isFlaggedFraud": "int8"}
    df = pd.read_csv(RAW / "paysim" / "paysim.csv", dtype=dt)
    df["hour"] = (df.step % 24).astype("int16")
    df["log_amount"] = np.log1p(df.amount).astype("float32")
    # causal per-entity history (cumcount = strictly past events)
    df["orig_txn_idx"] = df.groupby("nameOrig", sort=False).cumcount().astype("int32")
    df["dest_in_deg"] = df.groupby("nameDest", sort=False).cumcount().astype("int32")
    df["dest_is_merchant"] = df.nameDest.str.startswith("M").astype("int8")
    # balance-derived (available at decision time, but simulator-flavoured -
    # reported as an ablation pair, see the DATA_SOURCES leakage note)
    df["err_orig"] = (df.oldbalanceOrg - df.amount - df.newbalanceOrig).astype("float32")
    df["err_dest"] = (df.oldbalanceDest + df.amount - df.newbalanceDest).astype("float32")
    df["orig_emptied"] = ((df.newbalanceOrig == 0) & (df.oldbalanceOrg > 0)).astype("int8")
    df["dest_zero_before"] = (df.oldbalanceDest == 0).astype("int8")
    df["amt_ratio"] = (df.amount / (df.oldbalanceOrg + 1.0)).astype("float32")
    df["frac_orig_moved"] = (df.amount / (df.oldbalanceOrg + 1.0)).clip(0, 5).astype("float32")
    df["amount_eq_oldorig"] = (np.isclose(df.amount, df.oldbalanceOrg)).astype("int8")
    for c in ["oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]:
        df[c] = df[c].astype("float32")
    keep = ["step", "isFraud", "isFlaggedFraud"] + BEHAV_COLS + BALANCE_COLS
    df[keep].to_parquet(PAYSIM_FEATS, index=False)
    print(f"paysim features: {df.shape[0]:,} rows -> {PAYSIM_FEATS.name} "
          f"({time.time()-t0:.1f}s)")


def paysim_fit() -> None:
    t0 = time.time()
    df = pd.read_parquet(PAYSIM_FEATS)
    df["type"] = df["type"].astype("category")
    test = df[df.step > 600].copy()
    train = df[df.step <= 600]
    # all fraud + 2.0M sampled legit (memory-safe; class_weight handles imbalance;
    # ranking is unaffected and eval is on the FULL untouched test set)
    tr = pd.concat([train[train.isFraud == 1],
                    train[train.isFraud == 0].sample(2_000_000, random_state=SEED)])
    del df, train
    print(f"train(sampled) {len(tr):,} (fraud {int(tr.isFraud.sum()):,}) | "
          f"test {len(test):,} (fraud {int(test.isFraud.sum()):,})")
    out = pd.DataFrame({"y": test.isFraud.values,
                        "flagged_baseline": test.isFlaggedFraud.values})
    for label, cols in [("full", BEHAV_COLS + BALANCE_COLS), ("behavioral_only", BEHAV_COLS)]:
        m = hgb(max_leaf_nodes=63, max_iter=300,
                class_weight="balanced").fit(tr[cols], tr.isFraud)
        out[f"score_{label}"] = m.predict_proba(test[cols])[:, 1]
        print(f"  model {label}: {len(cols)} feats  ({time.time()-t0:.1f}s)")
    out.to_parquet(PROC / "paysim_scores.parquet", index=False)
    print(f"scores -> paysim_scores.parquet ({time.time()-t0:.1f}s)")


# -------------------------------------------------------------- IEEE-CIS (#1/#2)
IEEE_FEATS = PROC / "ieee_features.parquet"
TXN_NUM = ["TransactionAmt", "card1", "card2", "card3", "card5", "addr1", "addr2",
           "dist1", "dist2"] + [f"C{i}" for i in range(1, 15)] + \
          ["D1", "D2", "D3", "D4", "D5", "D10", "D15"]
TXN_CAT = ["ProductCD", "card4", "card6", "P_emaildomain", "R_emaildomain",
           "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9"]
ID_NUM = ["id_01", "id_02", "id_05", "id_06", "id_11"]
ID_CAT = ["DeviceType", "DeviceInfo_top", "id30_fam", "id31_fam", "id_15", "id_28", "id_29"]


def ieee_features() -> None:
    t0 = time.time()
    tx = pd.read_csv(RAW / "ieee_cis" / "train_transaction.csv",
                     usecols=["TransactionID", "TransactionDT", "isFraud"] + TXN_NUM + TXN_CAT)
    idn = pd.read_csv(RAW / "ieee_cis" / "train_identity.csv",
                      usecols=["TransactionID", "DeviceType", "DeviceInfo",
                               "id_30", "id_31", "id_15", "id_28", "id_29"] + ID_NUM)
    top_dev = idn.DeviceInfo.value_counts().head(100).index
    idn["DeviceInfo_top"] = np.where(idn.DeviceInfo.isin(top_dev), idn.DeviceInfo,
                                     np.where(idn.DeviceInfo.notna(), "OTHER", None))
    idn["id30_fam"] = idn.id_30.str.split().str[0].str.lower()
    idn["id31_fam"] = idn.id_31.str.split().str[0].str.lower()
    df = tx.merge(idn.drop(columns=["DeviceInfo", "id_30", "id_31"]),
                  on="TransactionID", how="left")
    df["has_identity"] = df.TransactionID.isin(set(idn.TransactionID)).astype("int8")
    for c in TXN_CAT + ID_CAT:
        df[c] = df[c].astype("category")
    for c in TXN_NUM + ID_NUM:
        df[c] = df[c].astype("float32")
    df.drop(columns=["TransactionID"]).to_parquet(IEEE_FEATS, index=False)
    print(f"ieee features: {df.shape} -> {IEEE_FEATS.name} ({time.time()-t0:.1f}s)")


def ieee_fit() -> None:
    t0 = time.time()
    df = pd.read_parquet(IEEE_FEATS)
    for c in TXN_CAT + ID_CAT:
        df[c] = df[c].astype("category")
    cut = df.TransactionDT.quantile(0.8)
    train, test = df[df.TransactionDT <= cut], df[df.TransactionDT > cut]
    print(f"time split: train {len(train):,} (fraud {int(train.isFraud.sum()):,}) | "
          f"test {len(test):,} (fraud {int(test.isFraud.sum()):,})")
    A = TXN_NUM + TXN_CAT                      # transaction-only
    B = A + ID_NUM + ID_CAT + ["has_identity"]  # + identity/device
    out = pd.DataFrame({"y": test.isFraud.values,
                        "has_identity": test.has_identity.values})
    for label, cols in [("txn_only", A), ("with_device", B)]:
        m = hgb().fit(train[cols], train.isFraud)
        out[f"score_{label}"] = m.predict_proba(test[cols])[:, 1]
        print(f"  model {label}: {len(cols)} feats ({time.time()-t0:.1f}s)")
    out.to_parquet(PROC / "ieee_scores.parquet", index=False)
    print(f"scores -> ieee_scores.parquet ({time.time()-t0:.1f}s)")


# ------------------------------------------------------------------- RBA (#4)
def rba_fit() -> None:
    """Successful-login RBA on the stratified sample (all 141 ATO + 2% negatives).

    Honest-evaluation notes (documented in metrics):
      * negatives were uniformly sampled at 0.6%*~3.3 ≈ 2.0% within the first-10M
        window -> FPR / recall / ROC-AUC unbiased; precision corrected via rate.
      * evaluation restricted to the time window that HAS negatives.
      * `Is Attack IP` (IP reputation feed) is an ablation: score_noip vs score_ip.
    """
    t0 = time.time()
    df = pd.read_csv(ROOT / "data" / "samples" / "rba_sample.csv")
    df.columns = [c.strip() for c in df.columns]
    ts = pd.to_datetime(df["Login Timestamp"], errors="coerce")
    df = df.assign(ts=ts).dropna(subset=["ts"])
    df["y"] = df["Is Account Takeover"].astype(str).str.lower().eq("true").astype(int)
    df["attack_ip"] = df["Is Attack IP"].astype(str).str.lower().eq("true").astype(int)
    df["success"] = df["Login Successful"].astype(str).str.lower().eq("true")
    # RBA scores successful logins (that's where step-up happens)
    df = df[df.success].copy()
    # negatives exist only inside the sampled window: clamp eval there
    wmax = df.loc[df.y == 0, "ts"].max()
    df = df[df.ts <= wmax].copy()
    df["hour"] = df.ts.dt.hour.astype("int16")
    df["rtt"] = np.log1p(pd.to_numeric(df["Round-Trip Time [ms]"], errors="coerce"))
    df["rtt"] = df.rtt.fillna(df.rtt.median())
    df["device_type"] = df["Device Type"].fillna("unknown").astype("category")
    df["browser"] = df["Browser Name and Version"].astype(str).str.split().str[0]
    df["os"] = df["OS Name and Version"].astype(str).str.split().str[0]
    # temporal cut anchored on ATO timestamps so the test window is not ATO-sparse
    # (ATO injection ends ~Nov 2020 while genuine traffic runs to Feb 2021).
    cut = df.loc[df.y == 1, "ts"].quantile(0.6)
    train, test = df[df.ts <= cut].copy(), df[df.ts > cut].copy()
    # frequency encodings computed on TRAIN genuine traffic only (causal)
    for col, newc in [("Country", "f_country"), ("ASN", "f_asn"),
                      ("browser", "f_browser"), ("os", "f_os")]:
        freq = train.loc[train.y == 0, col].value_counts(normalize=True)
        for part in (train, test):
            part[newc] = np.log1p(part[col].map(freq).fillna(0) * 1e6).astype("float32")
    feats = ["hour", "rtt", "f_country", "f_asn", "f_browser", "f_os", "device_type"]
    print(f"window<= {wmax} | train {len(train):,} (ATO {int(train.y.sum())}) | "
          f"test {len(test):,} (ATO {int(test.y.sum())})")
    out = pd.DataFrame({"y": test.y.values, "attack_ip_baseline": test.attack_ip.values})
    for label, cols in [("noip", feats), ("ip", feats + ["attack_ip"])]:
        m = hgb(max_iter=300).fit(train[cols], train.y)
        out[f"score_{label}"] = m.predict_proba(test[cols])[:, 1]
        print(f"  model {label} ({time.time()-t0:.1f}s)")
    out.to_parquet(PROC / "rba_scores.parquet", index=False)
    print(f"scores -> rba_scores.parquet ({time.time()-t0:.1f}s)")


# --------------------------------------------------------- CMU keystroke (#1)
def cmu_fit() -> None:
    """Scaled-Manhattan per-user anomaly detector (Killourhy & Maxion protocol).

    Train: first 200 reps per user. Genuine test: last 200 reps.
    Impostor test: first 5 reps of every other user (250 attempts).
    """
    t0 = time.time()
    df = pd.read_csv(RAW / "cmu_keystroke" / "DSL-StrongPasswordData.csv")
    feats = [c for c in df.columns if c not in ("subject", "sessionIndex", "rep")]
    rows = []
    for subj, g in df.groupby("subject"):
        g = g.sort_values(["sessionIndex", "rep"])
        tr, gen = g[feats].iloc[:200], g[feats].iloc[200:]
        mu = tr.mean()
        mad = (tr - mu).abs().mean().replace(0, 1e-6)  # mean abs deviation
        imp = df[df.subject != subj].groupby("subject", sort=False).head(5)[feats]
        s_gen = -((gen - mu).abs() / mad).sum(axis=1)
        s_imp = -((imp - mu).abs() / mad).sum(axis=1)
        rows.append(pd.DataFrame({
            "user": subj,
            "y": np.r_[np.zeros(len(s_gen), int), np.ones(len(s_imp), int)],
            # score = anomaly = higher means more impostor-like
            "score": -np.r_[s_gen.values, s_imp.values]}))
    out = pd.concat(rows, ignore_index=True)
    out.to_parquet(PROC / "cmu_scores.parquet", index=False)
    print(f"cmu scores: {out.shape[0]:,} genuine+impostor attempts, "
          f"{out.user.nunique()} users ({time.time()-t0:.1f}s)")


# -------------------------------------------------------- CERT insider (#5)
CERT_FEATS = PROC / "cert_userday.parquet"


def cert_features() -> None:
    t0 = time.time()
    base = RAW / "cert_insider" / "r4.2"
    lo = pd.read_csv(base / "logon.csv")
    lo["dt"] = pd.to_datetime(lo.date, format="%m/%d/%Y %H:%M:%S")
    lo["day"] = lo.dt.dt.date
    lo["off"] = (lo.dt.dt.hour < 6) | (lo.dt.dt.hour >= 20)
    modal_pc = lo[lo.activity == "Logon"].groupby("user").pc.agg(lambda s: s.mode().iat[0])
    lo["other_pc"] = lo.pc.ne(lo.user.map(modal_pc))
    logon = lo[lo.activity == "Logon"]
    f1 = logon.groupby(["user", "day"]).agg(
        n_logon=("id", "count"), n_off_logon=("off", "sum"),
        n_pcs=("pc", "nunique"), n_other_pc=("other_pc", "sum")).reset_index()
    dv = pd.read_csv(base / "device.csv")
    dv["dt"] = pd.to_datetime(dv.date, format="%m/%d/%Y %H:%M:%S")
    dv["day"] = dv.dt.dt.date
    dv["off"] = (dv.dt.dt.hour < 6) | (dv.dt.dt.hour >= 20)
    conn = dv[dv.activity == "Connect"]
    f2 = conn.groupby(["user", "day"]).agg(
        n_usb=("id", "count"), n_off_usb=("off", "sum")).reset_index()
    fl = pd.read_csv(base / "file.csv", usecols=["id", "date", "user", "pc"])
    fl["dt"] = pd.to_datetime(fl.date, format="%m/%d/%Y %H:%M:%S")
    fl["day"] = fl.dt.dt.date
    fl["off"] = (fl.dt.dt.hour < 6) | (fl.dt.dt.hour >= 20)
    f3 = fl.groupby(["user", "day"]).agg(
        n_file=("id", "count"), n_off_file=("off", "sum")).reset_index()
    ud = f1.merge(f2, on=["user", "day"], how="outer").merge(f3, on=["user", "day"], how="outer")
    ud = ud.fillna(0)
    # labels: user-day inside that user's insider-threat window (answers/insiders.csv, r4.2)
    ins = pd.read_csv(RAW / "cert_insider" / "answers" / "insiders.csv")
    ins = ins[ins.dataset.astype(str) == "4.2"]
    win = {r.user: (pd.to_datetime(r.start).date(), pd.to_datetime(r.end).date())
           for r in ins.itertuples()}
    def lab(row):
        w = win.get(row.user)
        return int(w is not None and w[0] <= row.day <= w[1])
    ud["y"] = ud.apply(lab, axis=1)
    # --- per-user robust deviation features (UEBA: deviation from a user's OWN norm) ---
    base = ["n_logon", "n_off_logon", "n_pcs", "n_other_pc",
            "n_usb", "n_off_usb", "n_file", "n_off_file"]
    g = ud.groupby("user")
    med = g[base].transform("median")
    mad = g[base].transform(lambda s: (s - s.median()).abs().median())
    mad = mad.replace(0, np.nan)
    dev = (ud[base] - med) / (1.4826 * mad)          # robust z (one-sided interest: spikes)
    dev = dev.clip(lower=0).fillna(0.0)              # only positive deviations are suspicious
    dev.columns = [f"dev_{c}" for c in base]
    ud = pd.concat([ud, dev.astype("float32")], axis=1)
    ud.to_parquet(CERT_FEATS, index=False)
    print(f"cert user-days: {len(ud):,} rows, {ud.user.nunique()} users, "
          f"malicious user-days {int(ud.y.sum()):,} over {len(win)} insiders "
          f"(+per-user deviation feats) ({time.time()-t0:.1f}s)")


def cert_fit() -> None:
    t0 = time.time()
    ud = pd.read_parquet(CERT_FEATS)
    dev = [c for c in ud.columns if c.startswith("dev_")]
    iso = IsolationForest(n_estimators=400, random_state=SEED, n_jobs=-1).fit(ud[dev])
    ud["score_iforest"] = -iso.score_samples(ud[dev])     # unsupervised, per-user-normalized
    # transparent rule: summed positive deviation on exfil/off-hours signals
    ud["score_zsum"] = ud[["dev_n_off_logon", "dev_n_other_pc", "dev_n_off_usb",
                           "dev_n_off_file", "dev_n_usb", "dev_n_file"]].sum(axis=1)
    ud[["user", "day", "y", "score_iforest", "score_zsum"]].to_parquet(
        PROC / "cert_scores.parquet", index=False)
    print(f"cert scores -> cert_scores.parquet ({time.time()-t0:.1f}s)")


# ----------------------------------------------------------------------- main
STAGES = {
    "paysim": {"features": paysim_features, "fit": paysim_fit},
    "ieee_cis": {"features": ieee_features, "fit": ieee_fit},
    "rba": {"fit": rba_fit},
    "cmu_keystroke": {"fit": cmu_fit},
    "cert_insider": {"features": cert_features, "fit": cert_fit},
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=STAGES)
    ap.add_argument("--stage", default="all", choices=["features", "fit", "all"])
    a = ap.parse_args()
    todo = STAGES[a.dataset]
    for st, fn in todo.items():
        if a.stage in ("all", st):
            fn()
