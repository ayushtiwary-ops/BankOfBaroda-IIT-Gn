#!/usr/bin/env python3
"""
PRAMAAN -- evaluate_tuned.py: final locked-test evaluation of a tuned model.

    python src/evaluate_tuned.py ieee_cis

Reads the Optuna best params from results/evaluation/<det>/tuning/optuna_best.json,
applies probability calibration chosen on the VALIDATION slice, then reads the
locked temporal TEST split EXACTLY ONCE to report the full Phase-A metric suite
for the tuned model. Writes results/evaluation/<det>_tuned/metrics_full.json and
results/evaluation/old_vs_new.json comparing incumbent vs tuned with paired
bootstrap 95% CIs on the held-out test, and a keep/replace decision.

Discipline: model selection + calibration use TRAIN/VAL only; the test split is
untouched until the single final scoring below.
"""
from __future__ import annotations

import os
import sys

if not os.environ.get("_PRAMAAN_REEXEC"):
    extra = "/opt/homebrew/opt/libomp/lib:/usr/local/opt/libomp/lib"
    cur = os.environ.get("DYLD_LIBRARY_PATH", "")
    os.environ["DYLD_LIBRARY_PATH"] = extra + (":" + cur if cur else "")
    os.environ["PYTHONUNBUFFERED"] = "1"
    os.environ["_PRAMAAN_REEXEC"] = "1"
    try:
        os.execv(sys.executable, [sys.executable, "-u"] + sys.argv)
    except Exception:
        pass

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_curve

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "results" / "evaluation"
PROC = ROOT / "data" / "processed"
SEED = 42
sys.path.insert(0, str(ROOT / "src"))
import eval_full as EF
import tune as TU

try:
    import lightgbm as lgb
    HAVE_LGB = True
except Exception:
    HAVE_LGB = False


def _recall_at(y, s, t):
    fpr, tpr, _ = roc_curve(y, s)
    i = int(np.searchsorted(fpr, t, side="right") - 1)
    return float(tpr[max(i, 0)])


def _brier_ece(y, p, n_bins=10):
    p = np.clip(p, 0, 1)
    brier = float(np.mean((p - y) ** 2))
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.sum() == 0:
            continue
        ece += (m.mean()) * abs(y[m].mean() - p[m].mean())
    return round(brier, 5), round(float(ece), 5)


def paired_bootstrap_delta(y, s_old, s_new, n=1000, seed=SEED):
    """Paired bootstrap of (new - old) on the SAME test rows."""
    rng = np.random.default_rng(seed)
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    d_ap, d_r2 = [], []
    for _ in range(n):
        bi = np.concatenate([rng.choice(pos, len(pos), True), rng.choice(neg, len(neg), True)])
        yb = y[bi]
        d_ap.append(average_precision_score(yb, s_new[bi]) - average_precision_score(yb, s_old[bi]))
        d_r2.append(_recall_at(yb, s_new[bi], 0.02) - _recall_at(yb, s_old[bi], 0.02))
    def summ(a):
        a = np.array(a)
        return {"mean": round(float(a.mean()), 4),
                "ci95": [round(float(np.percentile(a, 2.5)), 4),
                         round(float(np.percentile(a, 97.5)), 4)]}
    return {"delta_pr_auc": summ(d_ap), "delta_recall_2pct": summ(d_r2)}


def run_ieee():
    best = json.load(open(EVAL / "ieee_with_device" / "tuning" / "optuna_best.json"))
    tr, val, train_all, test, cols, cat, num = TU.ieee_splits()
    ytr, yval, ytest = tr.isFraud.values, val.isFraud.values, test.isFraud.values
    print(f"splits: train_inner {len(tr):,} | val {len(val):,} | test {len(test):,} "
          f"(test fraud {int(ytest.sum())})")

    if not HAVE_LGB:
        print("lightgbm unavailable; cannot evaluate the tuned model"); return
    params = dict(best["best_params"]); params.update(objective="binary", verbose=-1,
                                                      seed=SEED, feature_pre_filter=False)
    n_round = int(best.get("best_iter") or 300)
    dtr = lgb.Dataset(tr[cols], label=ytr, categorical_feature=cat)
    booster = lgb.train(params, dtr, num_boost_round=max(n_round, 1))

    # ---- Round 3: probability calibration chosen on VALIDATION only ----
    p_val_raw = booster.predict(val[cols])
    iso = IsotonicRegression(out_of_bounds="clip").fit(p_val_raw, yval)
    platt = LogisticRegression(max_iter=1000).fit(p_val_raw.reshape(-1, 1), yval)
    cal_val = {
        "raw": _brier_ece(yval, p_val_raw),
        "isotonic": _brier_ece(yval, iso.predict(p_val_raw)),
        "platt": _brier_ece(yval, platt.predict_proba(p_val_raw.reshape(-1, 1))[:, 1]),
    }
    pick = min(cal_val, key=lambda k: cal_val[k][0]) # lowest val Brier
    print(f"calibration on val (brier,ece): {cal_val} -> pick '{pick}'")
    json.dump({"val_brier_ece": {k: {"brier": v[0], "ece": v[1]} for k, v in cal_val.items()},
               "chosen": pick}, open(EVAL / "ieee_with_device" / "tuning" / "calibration.json", "w"),
              indent=2)

    # ---- FINAL: read the locked TEST split ONCE ----
    p_test_raw = booster.predict(test[cols])
    if pick == "isotonic":
        p_test = iso.predict(p_test_raw)
    elif pick == "platt":
        p_test = platt.predict_proba(p_test_raw.reshape(-1, 1))[:, 1]
    else:
        p_test = p_test_raw

    m_new = EF.full_suite("ieee_tuned", ytest, p_test, is_proba=True,
                          extra={"split": "temporal: train first 80% TransactionDT, "
                                 "test last 20% (locked, read once)",
                                 "model": f"lightgbm (Optuna), calibration={pick}",
                                 "tuning": best})
    # feature importances of the tuned booster
    imp = dict(sorted(zip(cols, booster.feature_importance(importance_type="gain").tolist()),
                      key=lambda x: -x[1]))
    json.dump(imp, open(EVAL / "ieee_with_device" / "tuning" / "importances.json", "w"), indent=2)

    # ---- old vs new on the SAME locked test ----
    old = pd.read_parquet(PROC / "ieee_scores.parquet")
    assert len(old) == len(test) and int((old.y.values == ytest).sum()) == len(ytest), \
        "tuned-vs-incumbent test alignment failed"
    s_old = old.score_with_device.values
    old_ap = float(average_precision_score(ytest, s_old))
    old_r2 = _recall_at(ytest, s_old, 0.02)
    new_ap = m_new["threshold_free"]["pr_auc"]
    new_r2 = m_new["operating_points"]["stepup_2pct"]["recall"]
    delta = paired_bootstrap_delta(ytest, s_old, p_test)
    decision = ("replace" if delta["delta_pr_auc"]["ci95"][0] > 0 else "keep_incumbent")
    cmp = {
        "detection": "ieee_cis",
        "incumbent": {"model": "HistGradientBoosting (default)", "pr_auc": round(old_ap, 4),
                      "recall_2pct": round(old_r2, 4)},
        "tuned": {"model": f"lightgbm Optuna + {pick} calibration", "pr_auc": new_ap,
                  "recall_2pct": new_r2},
        "paired_delta_new_minus_old": delta,
        "decision": decision,
        "decision_rule": "replace only if the 95% CI lower bound of delta PR-AUC > 0 "
                         "on the paired held-out test bootstrap",
    }
    ovn_path = EVAL / "old_vs_new.json"
    allcmp = json.loads(ovn_path.read_text()) if ovn_path.exists() else {}
    allcmp["ieee_cis"] = cmp
    json.dump(allcmp, open(ovn_path, "w"), indent=2)
    print(json.dumps(cmp, indent=2))


RUNNERS = {"ieee_cis": run_ieee}
if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "ieee_cis"
    RUNNERS[which]()
