#!/usr/bin/env python3
"""
PRAMAAN -- eval_full.py: the complete offline metric suite for every detection.

    python src/eval_full.py paysim|ieee_cis|rba|cmu_keystroke|cert_insider|all

Reads the score tables produced by train.py (data/processed/<ds>_scores.parquet)
and writes the full metric set per detection to results/evaluation/<detection>/:

    metrics_full.json pr_curve.png roc_curve.png pauc.png
    calibration.png (probability scores only) confusion_at_op.png
    detection_vs_stepup.png

It also writes results/evaluation/leaderboard.csv summarising all detections.

What "complete" means here (one protocol, applied to every detector):
  * threshold-free: PR-AUC (average precision, primary), ROC-AUC, partial AUC at
    FPR<=1%, KS statistic, Gini.
  * operating points at FPR/step-up in {0.1,0.5,1,2,5}%: precision, recall, F1,
    F2, specificity, FPR, full-population TP/FP/FN/TN, lift over base rate.
  * calibration (probability scores): reliability curve, Brier, ECE.
  * cost-weighted: expected-cost-minimising threshold and the cost at each
    operating point, using a real per-event loss where the data carries an amount
    (PaySim, IEEE) and a stated cost ratio otherwise (RBA, CERT).
  * the detection-vs-step-up curve with the chosen operating point marked.
  * a baseline-lift comparison against the shipped rule / blocklist at one budget.
  * bootstrap 95% confidence intervals (>=1000 resamples) on the headline numbers.

Negative subsampling (RBA): negatives are reweighted by 1/neg_sample_rate so the
sampling-sensitive metrics (PR-AUC, precision, Brier, ECE) report population
values; rank/class-conditional metrics (ROC-AUC, recall, FPR, KS) are unbiased.

No model is retrained here; this regenerates numbers from frozen score tables.
The locked test split is read once per detection by train.py upstream.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
    roc_curve,
)

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
RAW = ROOT / "data" / "raw"
EVAL = ROOT / "results" / "evaluation"
EVAL.mkdir(parents=True, exist_ok=True)

SEED = 42
FPR_TARGETS = (0.001, 0.005, 0.01, 0.02, 0.05)
OP_FPR = 0.01 # default operating point reported as "operating_point"
N_BOOT = 1000
# friction cost of one false step-up: infra (A4 = INR 2) + CX (A5 = INR 15); see
# docs/BUSINESS_CASE.md. cost_fp is per false-positive challenge of a genuine user.
COST_FP = 17.0


def _trapz(y, x):
    fn = getattr(np, "trapezoid", None) or np.trapz
    return float(fn(y, x))


def sample_weights(y: np.ndarray, neg_rate: float) -> np.ndarray:
    """Weight negatives by 1/neg_rate so subsampled negatives represent the
    full population; positives keep weight 1."""
    w = np.ones(len(y), dtype=float)
    if neg_rate != 1.0:
        w[y == 0] = 1.0 / neg_rate
    return w


def _roc_pack(y, s, neg_rate):
    """ROC arrays plus sampling-corrected precision aligned to the thresholds."""
    fpr, tpr, thr = roc_curve(y, s)
    P = float((y == 1).sum())
    N_full = float((y == 0).sum()) / neg_rate
    tp = tpr * P
    fp_full = fpr * N_full
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(tp + fp_full > 0, tp / (tp + fp_full), 1.0)
    return fpr, tpr, thr, precision, P, N_full


def _op_at_fpr(fpr, tpr, thr, precision, target):
    i = int(np.searchsorted(fpr, target, side="right") - 1)
    i = max(i, 0)
    return i, dict(fpr=float(fpr[i]), recall=float(tpr[i]),
                   precision=float(precision[i]), threshold=float(thr[i]))


def threshold_free(y, s, neg_rate):
    w = sample_weights(y, neg_rate)
    fpr, tpr, thr, precision, P, N_full = _roc_pack(y, s, neg_rate)
    roc = float(roc_auc_score(y, s)) # rank metric: unbiased
    ap = float(average_precision_score(y, s, sample_weight=w)) # corrected PR-AUC
    pr_trapz = _trapz(np.nan_to_num(precision, nan=1.0), tpr)
    try:
        pauc1 = float(roc_auc_score(y, s, max_fpr=0.01)) # McClish-standardised
    except Exception:
        pauc1 = float("nan")
    ks = float(np.max(tpr - fpr))
    return {
        "pr_auc": round(ap, 4),
        "pr_auc_trapz_over_recall": round(pr_trapz, 4),
        "roc_auc": round(roc, 4),
        "pauc_fpr_le_1pct_standardized": round(pauc1, 4),
        "ks_statistic": round(ks, 4),
        "gini": round(2 * roc - 1, 4),
    }


def operating_points(y, s, neg_rate, targets=FPR_TARGETS):
    fpr, tpr, thr, precision, P, N_full = _roc_pack(y, s, neg_rate)
    base_rate = P / (P + N_full)
    out = {}
    for t in targets:
        _, o = _op_at_fpr(fpr, tpr, thr, precision, t)
        r, pr = o["recall"], o["precision"]
        f1 = (2 * pr * r / (pr + r)) if (pr + r) > 0 else 0.0
        f2 = (5 * pr * r / (4 * pr + r)) if (4 * pr + r) > 0 else 0.0
        tp = r * P
        fp = o["fpr"] * N_full
        out[f"stepup_{t*100:g}pct".replace(".", "_")] = {
            "step_up_rate_genuine": round(o["fpr"], 5),
            "recall": round(r, 4),
            "precision": round(pr, 6),
            "f1": round(f1, 4),
            "f2": round(f2, 4),
            "specificity": round(1 - o["fpr"], 5),
            "lift_over_base_rate": round(pr / base_rate, 1) if base_rate > 0 else None,
            "threshold": round(o["threshold"], 6),
            "full_pop_confusion": {
                "tp": int(round(tp)), "fn": int(round(P - tp)),
                "fp": int(round(fp)), "tn": int(round(N_full - fp)),
            },
        }
    return out, base_rate


def calibration(y, s, neg_rate, n_bins=10, is_proba=None):
    """Weighted reliability curve + Brier + ECE.

    Only meaningful for calibrated probabilities. `is_proba` is set explicitly by
    the caller (True for predict_proba detectors, False for anomaly-rank detectors);
    if left None we fall back to a range check, but range alone is not sufficient --
    an IsolationForest score can land in [0,1] without being a probability.
    """
    if is_proba is False:
        return None
    if is_proba is None and not (np.nanmin(s) >= -1e-9 and np.nanmax(s) <= 1 + 1e-9):
        return None
    w = sample_weights(y, neg_rate)
    s = np.clip(s, 0, 1)
    brier = float(brier_score_loss(y, s, sample_weight=w))
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(s, edges[1:-1]), 0, n_bins - 1)
    W = w.sum()
    ece = 0.0
    pts = []
    for b in range(n_bins):
        m = idx == b
        wb = w[m].sum()
        if wb == 0:
            continue
        conf = float(np.average(s[m], weights=w[m]))
        acc = float(np.average(y[m], weights=w[m]))
        ece += (wb / W) * abs(acc - conf)
        pts.append({"bin": b, "confidence": round(conf, 4),
                    "empirical": round(acc, 4), "weight_frac": round(wb / W, 4)})
    return {"brier": round(brier, 5), "ece": round(ece, 5),
            "reliability": pts, "n_bins": n_bins}


def cost_analysis(y, s, neg_rate, loss=None, cost_fn_scalar=None,
                  cost_fp=COST_FP, ratios=(10, 100, 1000)):
    """Expected-cost-minimising threshold.

    cost(tau) = sum(loss of missed positives) + cost_fp * (#false-positives, full pop).
    If `loss` (per-positive monetary amount) is given it is used directly; otherwise
    a stated cost_fn_scalar (cost of one missed positive) is used and we sweep a few
    cost ratios so the operating point is shown to be robust to the assumption.
    """
    w = sample_weights(y, neg_rate)
    order = np.argsort(-s)
    ys, ss, ws = y[order], s[order], w[order]
    pos = (ys == 1).astype(float)
    neg_w = ws * (ys == 0)
    cum_fp = np.cumsum(neg_w) # false positives (full pop) if we flag top-k
    if loss is not None:
        lo = loss[order].astype(float)
        tot_loss = float((lo * pos).sum())
        cum_caught_loss = np.cumsum(lo * pos)
        missed_loss = tot_loss - cum_caught_loss
        total = missed_loss + cost_fp * cum_fp
        k = int(np.argmin(total))
        thr = float(ss[k])
        caught = float(pos[:k + 1].sum())
        return {
            "mode": "amount_based",
            "cost_fp_per_challenge": cost_fp,
            "total_fraud_amount_at_risk": round(tot_loss, 2),
            "min_cost_threshold": round(thr, 6),
            "min_cost_value": round(float(total[k]), 2),
            "recall_at_min_cost": round(caught / max(pos.sum(), 1), 4),
            "step_up_rate_at_min_cost": round(float(cum_fp[k]) /
                                              (len(y) - pos.sum()) * neg_rate, 5),
            "amount_recovered_at_min_cost": round(float(cum_caught_loss[k]), 2),
            "note": "missed-fraud cost = sum of missed transaction amounts (real loss); "
                    "false-positive cost = COST_FP per challenged genuine user",
        }
    # ratio-based (no per-event amount): report cost-min threshold per ratio
    P = pos.sum()
    cum_caught = np.cumsum(pos)
    res = {"mode": "ratio_based", "cost_fp_per_challenge": cost_fp,
           "note": "no per-event amount; cost_fn = ratio x cost_fp (stated assumption [A])",
           "by_ratio": {}}
    for rt in ratios:
        cfn = rt * cost_fp
        missed = P - cum_caught
        total = cfn * missed + cost_fp * cum_fp
        k = int(np.argmin(total))
        res["by_ratio"][f"cfn_over_cfp_{rt}"] = {
            "min_cost_threshold": round(float(ss[k]), 6),
            "recall_at_min_cost": round(float(cum_caught[k]) / max(P, 1), 4),
            "step_up_rate_at_min_cost": round(float(cum_fp[k]) /
                                              max(len(y) - P, 1) * neg_rate, 5),
        }
    return res


def _recall_at(fpr, tpr, t):
    i = int(np.searchsorted(fpr, t, side="right") - 1)
    return float(tpr[max(i, 0)])


def bootstrap(y, s, neg_rate, n=N_BOOT, seed=SEED, boot_neg_cap=None):
    """Stratified bootstrap 95% CIs for ROC-AUC, PR-AUC, recall@1/2/5% step-up.

    boot_neg_cap: when the negative pool is huge (RBA full test ~6.8M), resample a
    capped number of negatives per iteration and reweight PR-AUC negatives by
    (n_neg / cap) so the population PR-AUC is preserved while keeping each iteration
    tractable. ROC/recall are rank/class-conditional and are unbiased by the cap.
    Point estimates elsewhere always use the full test split.
    """
    rng = np.random.default_rng(seed)
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    n_neg_draw = len(neg)
    neg_w = 1.0 / neg_rate
    if boot_neg_cap is not None and len(neg) > boot_neg_cap:
        n_neg_draw = boot_neg_cap
        neg_w = (len(neg) / boot_neg_cap) / neg_rate # cap + sampling correction
    acc = {k: [] for k in ("roc_auc", "pr_auc", "recall_1pct", "recall_2pct", "recall_5pct")}
    for _ in range(n):
        bi = np.concatenate([rng.choice(pos, len(pos), replace=True),
                             rng.choice(neg, n_neg_draw, replace=True)])
        yb, sb = y[bi], s[bi]
        wb = np.where(yb == 1, 1.0, neg_w)
        if yb.sum() == 0 or (yb == 0).sum() == 0:
            continue
        try:
            acc["roc_auc"].append(roc_auc_score(yb, sb))
            acc["pr_auc"].append(average_precision_score(yb, sb, sample_weight=wb))
            fpr, tpr, _ = roc_curve(yb, sb)
            acc["recall_1pct"].append(_recall_at(fpr, tpr, 0.01))
            acc["recall_2pct"].append(_recall_at(fpr, tpr, 0.02))
            acc["recall_5pct"].append(_recall_at(fpr, tpr, 0.05))
        except Exception:
            continue
    out = {}
    for k, v in acc.items():
        if v:
            a = np.array(v)
            out[k] = {"mean": round(float(a.mean()), 4), "std": round(float(a.std()), 4),
                      "ci95": [round(float(np.percentile(a, 2.5)), 4),
                               round(float(np.percentile(a, 97.5)), 4)]}
    return out


# ----------------------------------------------------------------- plotting
def _plots(out_dir, name, y, s, neg_rate, op_target=OP_FPR, cal=None):
    out_dir.mkdir(parents=True, exist_ok=True)
    fpr, tpr, thr, precision, P, N_full = _roc_pack(y, s, neg_rate)

    # detection vs step-up (signature plot) with chosen op marked
    plt.figure(figsize=(6.4, 4.6))
    plt.plot(fpr * 100, tpr * 100, lw=2, label=name)
    for t in (0.01, 0.02, 0.05):
        _, o = _op_at_fpr(fpr, tpr, thr, precision, t)
        plt.scatter([o["fpr"] * 100], [o["recall"] * 100], zorder=3)
        plt.annotate(f'{o["recall"]*100:.0f}% @ {t*100:.0f}%',
                     (o["fpr"] * 100, o["recall"] * 100),
                     textcoords="offset points", xytext=(8, -4), fontsize=9)
    plt.xscale("log")
    plt.xlabel("Step-up rate -- % of genuine events challenged (log)")
    plt.ylabel("Detection -- % of positives caught")
    plt.title(f"Detection vs friction -- {name}")
    plt.grid(alpha=0.3); plt.legend(loc="lower right"); plt.tight_layout()
    plt.savefig(out_dir / "detection_vs_stepup.png", dpi=140); plt.close()

    # PR curve (sampling-corrected)
    plt.figure(figsize=(5.6, 4.4))
    plt.plot(tpr, precision, lw=2)
    plt.xlabel("Recall"); plt.ylabel("Precision (sampling-corrected)")
    plt.title(f"PR -- {name}")
    plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(out_dir / "pr_curve.png", dpi=140); plt.close()

    # ROC
    plt.figure(figsize=(5.6, 4.4))
    plt.plot(fpr, tpr, lw=2); plt.plot([0, 1], [0, 1], "--", lw=1, color="grey")
    plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.title(f"ROC -- {name}")
    plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(out_dir / "roc_curve.png", dpi=140); plt.close()

    # partial-AUC region (ROC zoomed to FPR<=1%, the bank's operating region)
    plt.figure(figsize=(5.6, 4.4))
    m = fpr <= 0.01
    plt.plot(fpr[m] * 100, tpr[m] * 100, lw=2)
    plt.fill_between(fpr[m] * 100, 0, tpr[m] * 100, alpha=0.15)
    plt.xlabel("FPR (%)"); plt.ylabel("Recall (%)")
    plt.title(f"Operating region FPR<=1% -- {name}")
    plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(out_dir / "pauc.png", dpi=140); plt.close()

    # confusion at the chosen operating point (full population)
    _, o = _op_at_fpr(fpr, tpr, thr, precision, op_target)
    tp = o["recall"] * P; fp = o["fpr"] * N_full
    cells = np.array([[tp, P - tp], [fp, N_full - fp]])
    plt.figure(figsize=(4.6, 4.0))
    plt.imshow(np.log10(cells + 1), cmap="Blues")
    for (i, j), v in np.ndenumerate(cells):
        plt.text(j, i, f"{int(round(v)):,}", ha="center", va="center", fontsize=10)
    plt.xticks([0, 1], ["pred +", "pred -"]); plt.yticks([0, 1], ["actual +", "actual -"])
    plt.title(f"Confusion @ {op_target*100:g}% step-up -- {name}")
    plt.tight_layout(); plt.savefig(out_dir / "confusion_at_op.png", dpi=140); plt.close()

    # calibration / reliability
    if cal is not None and cal.get("reliability"):
        rel = cal["reliability"]
        plt.figure(figsize=(5.0, 4.4))
        plt.plot([0, 1], [0, 1], "--", color="grey", label="perfect")
        plt.plot([p["confidence"] for p in rel], [p["empirical"] for p in rel],
                 marker="o", label="model")
        plt.xlabel("Predicted probability"); plt.ylabel("Empirical frequency")
        plt.title(f"Calibration -- {name}\nBrier={cal['brier']} ECE={cal['ece']}")
        plt.grid(alpha=0.3); plt.legend(loc="upper left"); plt.tight_layout()
        plt.savefig(out_dir / "calibration.png", dpi=140); plt.close()


def full_suite(name, y, s, neg_rate=1.0, loss=None, cost_fn_scalar=None,
               extra=None, do_boot=True, is_proba=None, boot_neg_cap=None,
               n_boot=N_BOOT):
    """Run the entire protocol for one (y, score) detector and persist it."""
    y = np.asarray(y).astype(int)
    s = np.asarray(s, dtype=float)
    out_dir = EVAL / name
    out_dir.mkdir(parents=True, exist_ok=True)

    tf = threshold_free(y, s, neg_rate)
    ops, base_rate = operating_points(y, s, neg_rate)
    cal = calibration(y, s, neg_rate, is_proba=is_proba)
    cost = cost_analysis(y, s, neg_rate, loss=loss, cost_fn_scalar=cost_fn_scalar)
    boot = bootstrap(y, s, neg_rate, n=n_boot, boot_neg_cap=boot_neg_cap) if do_boot else {}

    P = int((y == 1).sum()); N = int((y == 0).sum())
    m = {
        "name": name,
        "n_pos": P, "n_neg_in_eval": N, "neg_sample_rate": neg_rate,
        "n_neg_full_pop": int(round(N / neg_rate)),
        "base_rate_full_pop": base_rate,
        "threshold_free": tf,
        "operating_points": ops,
        "calibration": cal if cal is not None
        else "score is an anomaly rank, not a probability -- Brier/ECE not applicable",
        "cost_weighted": cost,
        "bootstrap_ci_95": boot,
    }
    if extra:
        m.update(extra)
    with open(out_dir / "metrics_full.json", "w") as f:
        json.dump(m, f, indent=2)
    _plots(out_dir, name, y, s, neg_rate, cal=cal)
    return m


# =================================================================== runners
def _paysim_amounts(n_expected):
    feats = pd.read_parquet(PROC / "paysim_features.parquet",
                            columns=["step", "amount", "isFraud"])
    test = feats[feats.step > 600]
    return test


def run_paysim():
    df = pd.read_parquet(PROC / "paysim_scores.parquet")
    y = df.y.values
    # real per-transaction loss for the cost metric (align to score-table order)
    test = _paysim_amounts(len(df))
    assert len(test) == len(df) and int((test.isFraud.values == y).sum()) == len(y), \
        "paysim amount alignment check failed"
    loss = test.amount.values.astype(float)
    base = {"split": "temporal: train step<=600, test step>600",
            "test_fraud_rate_pct": round(float(y.mean() * 100), 3)}
    res = {}
    for label, col in [("paysim_full", "score_full"),
                       ("paysim_behavioral", "score_behavioral_only")]:
        res[label] = full_suite(label, y, df[col].values, loss=loss, extra=base, is_proba=True)
    # baseline lift: shipped isFlaggedFraud rule vs model at the rule's budget
    bl_recall = float(df.loc[y == 1, "flagged_baseline"].mean())
    bl_fpr = float(df.loc[y == 0, "flagged_baseline"].mean())
    fpr, tpr, thr, prec, P, Nf = _roc_pack(y, df.score_behavioral_only.values, 1.0)
    model_recall_at_blbudget = _recall_at(fpr, tpr, max(bl_fpr, 1e-9))
    for label in res:
        res[label]["baseline_lift"] = {
            "shipped_rule": "isFlaggedFraud",
            "rule_recall": round(bl_recall, 4), "rule_fpr": round(bl_fpr, 6),
            "model_recall_at_rule_budget": round(model_recall_at_blbudget, 4),
        }
        with open(EVAL / label / "metrics_full.json", "w") as f:
            json.dump(res[label], f, indent=2)
    return res


def run_ieee():
    df = pd.read_parquet(PROC / "ieee_scores.parquet")
    y = df.y.values
    # align TransactionAmt for the cost metric
    from train import IEEE_FEATS # reuse the exact split
    fe = pd.read_parquet(IEEE_FEATS)
    cut = fe.TransactionDT.quantile(0.8)
    test = fe[fe.TransactionDT > cut]
    loss = None
    if len(test) == len(df) and int((test.isFraud.values == y).sum()) == len(y):
        loss = test.TransactionAmt.values.astype(float)
    base = {"split": "temporal: train first 80% of TransactionDT, test last 20%",
            "test_fraud_rate_pct": round(float(y.mean() * 100), 3)}
    res = {}
    for label, col in [("ieee_txn_only", "score_txn_only"),
                       ("ieee_with_device", "score_with_device")]:
        res[label] = full_suite(label, y, df[col].values, loss=loss, extra=base, is_proba=True)
    sub = df[df.has_identity == 1]
    loss_sub = loss[df.has_identity.values == 1] if loss is not None else None
    res["ieee_with_device_identity_subset"] = full_suite(
        "ieee_with_device_identity_subset", sub.y.values, sub.score_with_device.values,
        loss=loss_sub, extra={"note": "transactions that carry device/identity data"}, is_proba=True)
    return res


def run_rba():
    df = pd.read_parquet(PROC / "rba_scores.parquet")
    y = df.y.values
    S = 0.02
    base = {"neg_sampling_note": "negatives uniformly sampled at ~2%; ROC/recall/FPR/KS "
            "unbiased, PR-AUC/precision/Brier/ECE corrected by reweighting negatives x50",
            "split": "temporal within sampled window; successful logins only"}
    res = {}
    for label, col in [("rba_noip", "score_noip"), ("rba_ip", "score_ip")]:
        # ATO loss is not in the data -> ratio-based cost; INR per ATO is an [A]
        res[label] = full_suite(label, y, df[col].values, neg_rate=S,
                                cost_fn_scalar=None, extra=base, is_proba=True)
    res["rba_ipreputation_baseline"] = full_suite(
        "rba_ipreputation_baseline", y, df.attack_ip_baseline.values, neg_rate=S,
        extra={"note": "IP-reputation flag alone as the score"}, do_boot=False)
    # baseline lift: IP blocklist vs behavioural model at a 2% step-up budget
    fpr, tpr, *_ = _roc_pack(y, df.score_noip.values, S)[:3]
    res["rba_noip"]["baseline_lift"] = {
        "shipped_rule": "Is Attack IP blocklist",
        "rule_recall_at_op": 0.0,
        "model_recall_at_2pct": round(_recall_at(*roc_curve(y, df.score_noip.values)[:2], 0.02), 4),
    }
    with open(EVAL / "rba_noip" / "metrics_full.json", "w") as f:
        json.dump(res["rba_noip"], f, indent=2)
    return res


def run_cmu():
    df = pd.read_parquet(PROC / "cmu_scores.parquet")
    out_dir = EVAL / "cmu_keystroke"
    out_dir.mkdir(parents=True, exist_ok=True)
    eers = []
    for u, g in df.groupby("user"):
        fpr, tpr, _ = roc_curve(g.y, g.score)
        i = int(np.argmin(np.abs(fpr - (1 - tpr))))
        eers.append({"user": u, "eer": float((fpr[i] + 1 - tpr[i]) / 2),
                     "roc_auc": float(roc_auc_score(g.y, g.score))})
    e = pd.DataFrame(eers)
    e.to_csv(out_dir / "per_user_eer.csv", index=False)
    # bootstrap over users for the mean-EER CI
    rng = np.random.default_rng(SEED)
    means = [e.eer.sample(len(e), replace=True, random_state=int(rng.integers(1e9))).mean()
             for _ in range(N_BOOT)]
    m = {
        "name": "cmu_keystroke_scaled_manhattan",
        "users": int(len(e)),
        "mean_eer": round(float(e.eer.mean()), 4),
        "median_eer": round(float(e.eer.median()), 4),
        "std_eer": round(float(e.eer.std()), 4),
        "mean_eer_ci95": [round(float(np.percentile(means, 2.5)), 4),
                          round(float(np.percentile(means, 97.5)), 4)],
        "global_roc_auc": round(float(roc_auc_score(df.y, df.score)), 4),
        "per_user_roc_auc_mean": round(float(e.roc_auc.mean()), 4),
        "literature_anchor": "Killourhy & Maxion DSN'09 scaled-Manhattan ~0.096 mean EER",
        "protocol": "train 200 reps/user; genuine=last 200; impostors=5 reps x 50 others",
        "calibration": "score is a distance, not a probability -- Brier/ECE not applicable",
    }
    json.dump(m, open(out_dir / "metrics_full.json", "w"), indent=2)
    plt.figure(figsize=(6, 4))
    plt.hist(e.eer, bins=20)
    plt.axvline(e.eer.mean(), ls="--", color="red", label=f"mean EER={e.eer.mean():.3f}")
    plt.axvline(0.096, ls=":", color="green", label="literature 0.096")
    plt.xlabel("per-user EER"); plt.ylabel("users")
    plt.title("CMU keystroke -- scaled-Manhattan")
    plt.legend(); plt.tight_layout()
    plt.savefig(out_dir / "eer_distribution.png", dpi=140); plt.close()
    return m


def run_cert():
    df = pd.read_parquet(PROC / "cert_scores.parquet")
    y = df.y.values
    res = {}
    for variant in ("score_iforest", "score_zsum"):
        res[f"cert_{variant}"] = full_suite(
            f"cert_{variant}", y, df[variant].values, neg_rate=1.0, is_proba=False,
            extra={"granularity": "user-day", "labels": "answers/insiders.csv r4.2 windows"})
    # analyst alert-budget sweep + per-scenario (top-k user-days/day)
    days = df.day.astype(str)
    ins = pd.read_csv(RAW / "cert_insider" / "answers" / "insiders.csv")
    ins = ins[ins.dataset.astype(str) == "4.2"][["user", "scenario"]].drop_duplicates()
    by_scen = {int(s): set(g.user) for s, g in ins.groupby("scenario")}
    insiders = set(df.loc[df.y == 1, "user"])
    sweep = []
    for k in (5, 10, 25, 50):
        flagged = df.assign(day=days).sort_values("score_iforest", ascending=False).groupby("day").head(k)
        det = set(flagged[flagged.y == 1].user)
        sweep.append({
            "budget_per_day": k,
            "userday_recall": round(float((flagged.y == 1).sum() / max(int(y.sum()), 1)), 4),
            "insider_users_detected": len(det), "insider_users_total": len(insiders),
            "user_detection_rate": round(len(det) / len(insiders), 4),
            "by_scenario": {f"scen{s}": f"{len(det & us)}/{len(us)}"
                            for s, us in sorted(by_scen.items())},
        })
    json.dump(sweep, open(EVAL / "cert_score_iforest" / "budget_sweep.json", "w"), indent=2)
    res["budget_sweep"] = sweep
    return res


def run_rba_full():
    """Full-dataset RBA: the ENTIRE held-out test split (all negatives, no 2%
    subsample) produced by train_rba_full.py -> population metrics, no correction."""
    df = pd.read_parquet(PROC / "rba_full_scores.parquet")
    y = df.y.values
    base = {"split": "temporal: cut = 0.6 quantile of ATO timestamps (successful logins); "
            "FULL held-out test, all negatives (neg_sample_rate=1.0, no correction)",
            "source": "rba-dataset.csv (31.3M logins) streamed from the verified zip"}
    res = {}
    for label, col in [("rba_full_noip", "score_noip"), ("rba_full_ip", "score_ip")]:
        res[label] = full_suite(label, y, df[col].values, neg_rate=1.0,
                                is_proba=True, extra=base,
                                boot_neg_cap=200_000, n_boot=500)
    res["rba_full_ipreputation_baseline"] = full_suite(
        "rba_full_ipreputation_baseline", y, df.attack_ip_baseline.values, neg_rate=1.0,
        extra={"note": "IP-reputation flag alone as the score"}, do_boot=False)
    return res


RUNNERS = {"paysim": run_paysim, "ieee_cis": run_ieee, "rba": run_rba,
           "rba_full": run_rba_full,
           "cmu_keystroke": run_cmu, "cert_insider": run_cert}


def write_leaderboard():
    rows = []
    for d in sorted(EVAL.iterdir()):
        f = d / "metrics_full.json"
        if not f.exists():
            continue
        m = json.loads(f.read_text())
        tf = m.get("threshold_free")
        if tf:
            op2 = m["operating_points"].get("stepup_2pct", {})
            boot = m.get("bootstrap_ci_95", {}).get("pr_auc", {})
            rows.append({
                "detection": m["name"], "n_pos": m.get("n_pos"),
                "pr_auc": tf["pr_auc"], "roc_auc": tf["roc_auc"],
                "pauc_1pct": tf["pauc_fpr_le_1pct_standardized"],
                "ks": tf["ks_statistic"], "gini": tf["gini"],
                "recall_at_2pct": op2.get("recall"), "f2_at_2pct": op2.get("f2"),
                "pr_auc_ci_lo": boot.get("ci95", [None, None])[0],
                "pr_auc_ci_hi": boot.get("ci95", [None, None])[1],
            })
        elif "mean_eer" in m:
            rows.append({"detection": m["name"], "n_pos": "51 users",
                         "pr_auc": None, "roc_auc": m.get("global_roc_auc"),
                         "pauc_1pct": None, "ks": None, "gini": None,
                         "recall_at_2pct": f"EER={m['mean_eer']}", "f2_at_2pct": None,
                         "pr_auc_ci_lo": None, "pr_auc_ci_hi": None})
    pd.DataFrame(rows).to_csv(EVAL / "leaderboard.csv", index=False)
    print(f"leaderboard -> {EVAL/'leaderboard.csv'} ({len(rows)} rows)")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    sys.path.insert(0, str(ROOT / "src"))
    names = list(RUNNERS) if which == "all" else [which]
    for n in names:
        print(f"== eval_full {n} ==")
        m = RUNNERS[n]()
        print(json.dumps(m, indent=2, default=str)[:900])
    write_leaderboard()
