"""
PRAMAAN shared evaluation library — the metric protocol.

Every detector is scored the same way, from a scores table (y_true, score):
  * ROC-AUC and PR-AUC (PR-AUC is the primary number — extreme class imbalance)
  * precision / recall @ fixed FPR (1% and 2%)
  * the detection-vs-step-up curve: recall (y) vs fraction of GENUINE events
    challenged (x). x is exactly the friction budget a bank chooses.
  * confusion matrix at the chosen operating point
  * optional negative-sampling correction: if only a fraction `neg_sample_rate`
    of genuine events is present (e.g. RBA: all 141 ATO kept, 2% of negatives),
    FPR/recall/ROC-AUC are unbiased as-is, while precision/PR-AUC must scale
    the false positives by 1/neg_sample_rate. We do that correction explicitly.

All outputs land in results/<name>/: metrics.json + curves (PNG).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve

RESULTS = Path(__file__).resolve().parents[1] / "results"


def _curves(y: np.ndarray, s: np.ndarray, neg_sample_rate: float = 1.0):
    """fpr, tpr (=recall), thresholds, and sampling-corrected precision."""
    fpr, tpr, thr = roc_curve(y, s)
    P = float((y == 1).sum())
    N_sampled = float((y == 0).sum())
    N_full = N_sampled / neg_sample_rate
    tp = tpr * P
    fp_full = fpr * N_full
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(tp + fp_full > 0, tp / (tp + fp_full), 1.0)
    return fpr, tpr, thr, precision


def _at_fpr(fpr, tpr, thr, precision, target_fpr):
    i = int(np.searchsorted(fpr, target_fpr, side="right") - 1)
    i = max(i, 0)
    return dict(
        fpr=float(fpr[i]), recall=float(tpr[i]),
        precision=float(precision[i]), threshold=float(thr[i]),
    )


def evaluate(name: str, y, s, neg_sample_rate: float = 1.0,
             op_fpr: float = 0.01, extra: dict | None = None,
             curve_label: str | None = None) -> dict:
    """Run the full protocol; write results/<name>/metrics.json + curves."""
    y = np.asarray(y).astype(int)
    s = np.asarray(s, dtype=float)
    out = RESULTS / name
    out.mkdir(parents=True, exist_ok=True)

    fpr, tpr, thr, precision = _curves(y, s, neg_sample_rate)
    pr_auc = float(np.trapz(np.nan_to_num(precision, nan=1.0), tpr))  # over recall axis
    roc = float(roc_auc_score(y, s))

    ops = {f"fpr_{int(t*100)}pct": _at_fpr(fpr, tpr, thr, precision, t)
           for t in (0.005, 0.01, 0.02, 0.05)}
    op = _at_fpr(fpr, tpr, thr, precision, op_fpr)
    P = int((y == 1).sum()); N = int((y == 0).sum())
    tp = int(round(op["recall"] * P)); fn = P - tp
    fp_sampled = int(round(op["fpr"] * N)); tn = N - fp_sampled

    m = {
        "name": name,
        "n_pos": P, "n_neg_in_eval": N, "neg_sample_rate": neg_sample_rate,
        "roc_auc": round(roc, 4), "pr_auc": round(pr_auc, 4),
        "operating_point": {"target_fpr": op_fpr, **{k: round(v, 4) for k, v in op.items()}},
        "confusion_at_op_sampled_negatives": {"tp": tp, "fn": fn, "fp": fp_sampled, "tn": tn},
        "at_fixed_fpr": {k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in ops.items()},
    }
    if extra:
        m.update(extra)

    # --- detection-vs-step-up curve (the signature plot) ---
    plt.figure(figsize=(6.4, 4.6))
    plt.plot(fpr * 100, tpr * 100, lw=2, label=curve_label or name)
    for t in (0.01, 0.02, 0.05):
        o = _at_fpr(fpr, tpr, thr, precision, t)
        plt.scatter([o["fpr"] * 100], [o["recall"] * 100], zorder=3)
        plt.annotate(f'{o["recall"]*100:.0f}% @ {t*100:.0f}%',
                     (o["fpr"] * 100, o["recall"] * 100),
                     textcoords="offset points", xytext=(8, -4), fontsize=9)
    plt.xscale("log")
    plt.xlabel("Step-up rate — % of genuine events challenged (log)")
    plt.ylabel("Detection — % of fraud caught")
    plt.title(f"Detection vs friction — {name}")
    plt.grid(alpha=0.3); plt.legend(loc="lower right"); plt.tight_layout()
    plt.savefig(out / "detection_vs_stepup.png", dpi=140); plt.close()

    # --- PR curve (sampling-corrected) ---
    plt.figure(figsize=(5.6, 4.4))
    plt.plot(tpr, precision, lw=2)
    plt.xlabel("Recall"); plt.ylabel("Precision (sampling-corrected)")
    plt.title(f"PR curve — {name}  (PR-AUC={pr_auc:.3f})")
    plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(out / "pr_curve.png", dpi=140); plt.close()

    # --- ROC ---
    plt.figure(figsize=(5.6, 4.4))
    plt.plot(fpr, tpr, lw=2); plt.plot([0, 1], [0, 1], "--", lw=1, color="grey")
    plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.title(f"ROC — {name}  (AUC={roc:.4f})")
    plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(out / "roc_curve.png", dpi=140); plt.close()

    with open(out / "metrics.json", "w") as f:
        json.dump(m, f, indent=2)
    return m


def compare_stepup(name: str, runs: list[tuple[str, np.ndarray, np.ndarray]],
                   neg_sample_rate: float = 1.0) -> None:
    """Overlay detection-vs-step-up curves (e.g. device-ablation A vs B)."""
    out = RESULTS / name
    out.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6.6, 4.8))
    for label, y, s in runs:
        fpr, tpr, _, _ = _curves(np.asarray(y).astype(int), np.asarray(s, float), neg_sample_rate)
        plt.plot(fpr * 100, tpr * 100, lw=2, label=label)
    plt.xscale("log")
    plt.xlabel("Step-up rate — % of genuine events challenged (log)")
    plt.ylabel("Detection — % of fraud caught")
    plt.title(f"Detection vs friction — {name}")
    plt.grid(alpha=0.3); plt.legend(loc="lower right"); plt.tight_layout()
    plt.savefig(out / "ablation_stepup.png", dpi=140); plt.close()
