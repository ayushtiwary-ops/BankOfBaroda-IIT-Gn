#!/usr/bin/env python3
"""
PRAMAAN -- train_rba_full.py: RBA / Wiefling on the FULL 31.3M-login dataset.

    python src/train_rba_full.py [--train-neg-cap 2000000]

Why this exists: train.py's rba_fit reads the small committed sample
(data/samples/rba_sample.csv, ~15k rows). The committed headline ("31.3M logins")
needs the full dataset, and the locked test split must use the FULL negative
population, not a 2% subsample. This script streams rba-dataset.csv straight out
of the committed zip (no 8.4 GB extraction), builds the same temporal split and
the same causal frequency-encoded features as train.py, trains on all ATO plus a
memory-capped negative subsample, and scores the ENTIRE held-out test split.

Result: results/evaluation has population FPR / recall / PR-AUC with no sampling
correction, and the numbers are reproducible from the verified raw zip.

Leakage discipline (identical to train.py):
  * temporal split: cut = 0.6 quantile of ATO timestamps among successful logins
    (genuine traffic continues after ATO injection ends, so this keeps ATO in test).
  * frequency encodings (Country/ASN/browser/OS) are fit on TRAIN genuine logins
    ONLY, then applied to train and test (no peeking at the future or at labels).
  * scores successful logins only (that is where a step-up happens).
  * `Is Attack IP` (IP-reputation feed) is kept as an ablation arm, never a label.
"""
from __future__ import annotations

import argparse
import io
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

SEED = 42
ROOT = Path(__file__).resolve().parents[1]
ZIP = ROOT / "data" / "raw" / "rba" / "rba-dataset.zip"
CSV_IN_ZIP = "rba-dataset.csv"
PROC = ROOT / "data" / "processed"
PROC.mkdir(parents=True, exist_ok=True)

USECOLS = ["Login Timestamp", "Round-Trip Time [ms]", "Country", "ASN",
           "Browser Name and Version", "OS Name and Version", "Device Type",
           "Login Successful", "Is Attack IP", "Is Account Takeover"]
CHUNK = 1_500_000


def _open_csv():
    z = zipfile.ZipFile(ZIP)
    return z, io.TextIOWrapper(z.open(CSV_IN_ZIP), encoding="utf-8")


def pass_a_cut() -> pd.Timestamp:
    """Light pass: ATO-successful timestamps -> the 0.6 temporal quantile cut."""
    t0 = time.time()
    z, fh = _open_csv()
    ato_ts = []
    n_succ = 0
    for ch in pd.read_csv(fh, usecols=["Login Timestamp", "Login Successful",
                                       "Is Account Takeover"], chunksize=CHUNK):
        succ = ch["Login Successful"].astype(str).str.lower().eq("true")
        ato = ch["Is Account Takeover"].astype(str).str.lower().eq("true")
        n_succ += int(succ.sum())
        m = succ & ato
        if m.any():
            ato_ts.append(pd.to_datetime(ch.loc[m, "Login Timestamp"], errors="coerce"))
    z.close()
    ts = pd.concat(ato_ts) if ato_ts else pd.Series([], dtype="datetime64[ns]")
    cut = ts.quantile(0.6)
    print(f"pass A: {n_succ:,} successful logins, {len(ts)} ATO; cut={cut} "
          f"({time.time()-t0:.1f}s)")
    return cut


def _prep_chunk(ch: pd.DataFrame) -> pd.DataFrame:
    succ = ch["Login Successful"].astype(str).str.lower().eq("true")
    ch = ch[succ].copy()
    ch["ts"] = pd.to_datetime(ch["Login Timestamp"], errors="coerce")
    ch = ch.dropna(subset=["ts"])
    ch["y"] = ch["Is Account Takeover"].astype(str).str.lower().eq("true").astype("int8")
    ch["attack_ip"] = ch["Is Attack IP"].astype(str).str.lower().eq("true").astype("int8")
    ch["hour"] = ch.ts.dt.hour.astype("int16")
    ch["rtt"] = np.log1p(pd.to_numeric(ch["Round-Trip Time [ms]"], errors="coerce")).astype("float32")
    ch["device_type"] = ch["Device Type"].fillna("unknown").astype("category")
    ch["browser"] = ch["Browser Name and Version"].astype(str).str.split().str[0]
    ch["os"] = ch["OS Name and Version"].astype(str).str.split().str[0]
    ch["country"] = ch["Country"].astype(str)
    ch["asn"] = ch["ASN"].astype(str)
    return ch[["ts", "y", "attack_ip", "hour", "rtt", "device_type",
               "browser", "os", "country", "asn"]]


def pass_b_collect(cut, train_neg_cap):
    """Full pass: collect ALL test rows + (all ATO train rows + capped neg sample)."""
    t0 = time.time()
    z, fh = _open_csv()
    train_parts, test_parts = [], []
    n_train_neg_seen = 0
    for ch in pd.read_csv(fh, usecols=USECOLS, chunksize=CHUNK, low_memory=False):
        d = _prep_chunk(ch)
        is_test = d.ts > cut
        test_parts.append(d[is_test])
        tr = d[~is_test]
        n_train_neg_seen += int((tr.y == 0).sum())
        train_parts.append(tr) # thin after we know totals
    z.close()
    test = pd.concat(test_parts, ignore_index=True)
    train = pd.concat(train_parts, ignore_index=True)
    del train_parts, test_parts
    # cap train negatives (keep all ATO)
    tr_pos = train[train.y == 1]
    tr_neg = train[train.y == 0]
    if len(tr_neg) > train_neg_cap:
        tr_neg = tr_neg.sample(train_neg_cap, random_state=SEED)
    train = pd.concat([tr_pos, tr_neg], ignore_index=True)
    print(f"pass B: train {len(train):,} (ATO {int(train.y.sum())}, neg seen "
          f"{n_train_neg_seen:,} capped to {len(tr_neg):,}) | "
          f"test {len(test):,} (ATO {int(test.y.sum())}, neg {int((test.y==0).sum()):,}) "
          f"({time.time()-t0:.1f}s)")
    return train, test


def add_freq_features(train, test):
    """Causal frequency encodings fit on TRAIN genuine traffic only."""
    for col, newc in [("country", "f_country"), ("asn", "f_asn"),
                      ("browser", "f_browser"), ("os", "f_os")]:
        freq = train.loc[train.y == 0, col].value_counts(normalize=True)
        for part in (train, test):
            part[newc] = np.log1p(part[col].map(freq).fillna(0) * 1e6).astype("float32")
    return ["hour", "rtt", "f_country", "f_asn", "f_browser", "f_os", "device_type"]


def main(train_neg_cap):
    t0 = time.time()
    cut = pass_a_cut()
    train, test = pass_b_collect(cut, train_neg_cap)
    feats = add_freq_features(train, test)
    train["device_type"] = train.device_type.astype("category")
    test["device_type"] = test.device_type.astype("category")
    out = pd.DataFrame({"y": test.y.values.astype(int),
                        "attack_ip_baseline": test.attack_ip.values.astype(int)})
    for label, cols in [("noip", feats), ("ip", feats + ["attack_ip"])]:
        m = HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.1, max_leaf_nodes=31, random_state=SEED,
            class_weight="balanced", categorical_features="from_dtype").fit(
            train[cols], train.y)
        out[f"score_{label}"] = m.predict_proba(test[cols])[:, 1]
        print(f" model {label}: {len(cols)} feats ({time.time()-t0:.1f}s)")
    dest = PROC / "rba_full_scores.parquet"
    out.to_parquet(dest, index=False)
    print(f"FULL-DATA scores -> {dest.name}: {len(out):,} test rows, "
          f"{int(out.y.sum())} ATO, neg_sample_rate=1.0 (no correction needed) "
          f"({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-neg-cap", type=int, default=2_000_000)
    a = ap.parse_args()
    main(a.train_neg_cap)
