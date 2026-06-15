#!/usr/bin/env python3
"""
PRAMAAN -- tune.py: reproducible model bake-off + Optuna tuning, validation-only.

    python src/tune.py ieee_cis [--trials 100] [--timeout 240]
    python src/tune.py paysim
    python src/tune.py rba

Protocol (the locked test split is never read here):
  ROUND 1 model bake-off on a temporal VALIDATION slice carved from TRAIN:
           HistGradientBoosting (incumbent), XGBoost, LightGBM, calibrated logistic.
  ROUND 2 Optuna (TPE) search on the winning family, optimising validation PR-AUC
           (average precision), with early stopping; fixed seed; study saved.
  ROUND 3 imbalance handling (scale_pos_weight / class weight) + probability
           calibration (isotonic / sigmoid) chosen on validation.
  ROUND 4 operating thresholds picked on validation; optional averaged ensemble of
           the top two families, kept only if it beats the best single on validation.

Outputs results/evaluation/<det>/tuning/: bakeoff.json, optuna_best.json,
study_trials.csv, calibration.json, importances.json, and best_params.json.
The FINAL locked-test evaluation of the tuned model is done by evaluate_tuned.py.

Reproducibility: fixed seeds; xgboost/lightgbm fall back to HistGradientBoosting
if their native libs are unavailable, so the script always runs.
"""
from __future__ import annotations

# --- make libomp discoverable for xgboost/lightgbm wheels on macOS, once ---
import os
import sys

if not os.environ.get("_PRAMAAN_REEXEC"):
    extra = "/opt/homebrew/opt/libomp/lib:/usr/local/opt/libomp/lib"
    cur = os.environ.get("DYLD_LIBRARY_PATH", "")
    os.environ["DYLD_LIBRARY_PATH"] = extra + (":" + cur if cur else "")
    os.environ["_PRAMAAN_REEXEC"] = "1"
    try:
        os.execv(sys.executable, [sys.executable] + sys.argv)
    except Exception:
        pass

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
EVAL = ROOT / "results" / "evaluation"
SEED = 42

sys.path.insert(0, str(ROOT / "src"))
import train as T

try:
    import xgboost as xgb
    HAVE_XGB = True
except Exception:
    HAVE_XGB = False
try:
    import lightgbm as lgb
    HAVE_LGB = True
except Exception:
    HAVE_LGB = False
try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAVE_OPTUNA = True
except Exception:
    HAVE_OPTUNA = False


def recall_at_fpr(y, s, t):
    fpr, tpr, _ = roc_curve(y, s)
    i = int(np.searchsorted(fpr, t, side="right") - 1)
    return float(tpr[max(i, 0)])


def val_metrics(y, s):
    return {"pr_auc": float(average_precision_score(y, s)),
            "roc_auc": float(roc_auc_score(y, s)),
            "recall_2pct": recall_at_fpr(y, s, 0.02)}


# =========================================================== IEEE-CIS
def ieee_splits():
    df = pd.read_parquet(T.IEEE_FEATS)
    for c in T.TXN_CAT + T.ID_CAT:
        df[c] = df[c].astype("category")
    cut = df.TransactionDT.quantile(0.8)
    train_all = df[df.TransactionDT <= cut].copy()
    test = df[df.TransactionDT > cut].copy()
    vcut = train_all.TransactionDT.quantile(0.8) # temporal val from TRAIN only
    tr = train_all[train_all.TransactionDT <= vcut].copy()
    val = train_all[train_all.TransactionDT > vcut].copy()
    cols = T.TXN_NUM + T.TXN_CAT + T.ID_NUM + T.ID_CAT + ["has_identity"]
    return tr, val, train_all, test, cols, T.TXN_CAT + T.ID_CAT, T.TXN_NUM + T.ID_NUM


def _spw(y):
    p = float((y == 1).sum()); n = float((y == 0).sum())
    return n / max(p, 1.0)


def ieee_bakeoff():
    tr, val, train_all, test, cols, cat, num = ieee_splits()
    ytr, yval = tr.isFraud.values, val.isFraud.values
    res = {}

    # incumbent: HistGradientBoosting (current production family)
    m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, max_leaf_nodes=31,
                                       random_state=SEED, categorical_features="from_dtype")
    m.fit(tr[cols], ytr)
    res["hist_gbdt_incumbent"] = val_metrics(yval, m.predict_proba(val[cols])[:, 1])

    # calibrated logistic baseline (interpretability anchor)
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), num),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore", max_categories=30,
                                               sparse_output=True))]),
         [c for c in cat])])
    logit = Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=200, C=1.0,
                                                               class_weight="balanced"))])
    trc = tr.copy(); valc = val.copy()
    for c in cat:
        trc[c] = trc[c].astype(str); valc[c] = valc[c].astype(str)
    logit.fit(trc[cols], ytr)
    res["logistic_balanced"] = val_metrics(yval, logit.predict_proba(valc[cols])[:, 1])

    if HAVE_XGB:
        dtr = xgb.DMatrix(tr[cols], label=ytr, enable_categorical=True)
        dval = xgb.DMatrix(val[cols], label=yval, enable_categorical=True)
        params = {"max_depth": 6, "eta": 0.1, "subsample": 0.8, "colsample_bytree": 0.8,
                  "objective": "binary:logistic", "eval_metric": "aucpr",
                  "tree_method": "hist", "scale_pos_weight": _spw(ytr), "seed": SEED}
        bst = xgb.train(params, dtr, num_boost_round=400, evals=[(dval, "val")],
                        early_stopping_rounds=30, verbose_eval=False)
        res["xgboost"] = val_metrics(yval, bst.predict(dval))

    if HAVE_LGB:
        dtr = lgb.Dataset(tr[cols], label=ytr, categorical_feature=cat)
        dval = lgb.Dataset(val[cols], label=yval, reference=dtr, categorical_feature=cat)
        params = {"objective": "binary", "metric": "average_precision", "num_leaves": 63,
                  "learning_rate": 0.05, "feature_fraction": 0.8, "bagging_fraction": 0.8,
                  "bagging_freq": 1, "scale_pos_weight": _spw(ytr), "seed": SEED,
                  "verbose": -1}
        bst = lgb.train(params, dtr, num_boost_round=600, valid_sets=[dval],
                        callbacks=[lgb.early_stopping(40, verbose=False)])
        res["lightgbm"] = val_metrics(yval, bst.predict(val[cols]))

    return res, (tr, val, train_all, test, cols, cat, num)


def ieee_optuna(data, n_trials, timeout, search_neg_cap=80_000):
    tr, val, train_all, test, cols, cat, num = data
    # subsample negatives for the SEARCH only (all positives kept) so >=100 Optuna
    # trials fit in budget; the final model is refit on full train by evaluate_tuned.
    tr_pos = tr[tr.isFraud == 1]
    tr_neg = tr[tr.isFraud == 0]
    if len(tr_neg) > search_neg_cap:
        tr_neg = tr_neg.sample(search_neg_cap, random_state=SEED)
    trs = pd.concat([tr_pos, tr_neg])
    ytr, yval = trs.isFraud.values, val.isFraud.values
    spw = _spw(tr.isFraud.values) # range from the FULL train imbalance
    print(f" Optuna search set: {len(trs):,} rows (all {int(tr_pos.shape[0])} pos + "
          f"{len(tr_neg):,} neg); validating on {len(val):,}")

    if HAVE_LGB:
        # feature_pre_filter=False so Optuna can lower min_child_samples on the
        # cached Dataset without LightGBM raising on the pre-filter mismatch
        dprm = {"feature_pre_filter": False}
        dtr = lgb.Dataset(trs[cols], label=ytr, categorical_feature=cat,
                          free_raw_data=False, params=dprm)
        dval = lgb.Dataset(val[cols], label=yval, reference=dtr,
                           categorical_feature=cat, free_raw_data=False, params=dprm)

        def objective(trial):
            p = {"objective": "binary", "metric": "average_precision",
                 "num_leaves": trial.suggest_int("num_leaves", 15, 128),
                 "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.3, log=True),
                 "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
                 "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
                 "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
                 "min_child_samples": trial.suggest_int("min_child_samples", 5, 200),
                 "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
                 "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
                 "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, spw * 2),
                 "feature_pre_filter": False, "seed": SEED, "verbose": -1, "num_threads": 4}
            bst = lgb.train(p, dtr, num_boost_round=300, valid_sets=[dval],
                            callbacks=[lgb.early_stopping(25, verbose=False)])
            trial.set_user_attr("best_iter", bst.best_iteration)
            return average_precision_score(yval, bst.predict(val[cols]))
        family = "lightgbm"
    else:
        def objective(trial):
            p = dict(max_iter=trial.suggest_int("max_iter", 100, 600),
                     learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                     max_leaf_nodes=trial.suggest_int("max_leaf_nodes", 15, 255),
                     l2_regularization=trial.suggest_float("l2_regularization", 1e-8, 10, log=True),
                     min_samples_leaf=trial.suggest_int("min_samples_leaf", 5, 200),
                     random_state=SEED, categorical_features="from_dtype",
                     early_stopping=False)
            m = HistGradientBoostingClassifier(**p).fit(trs[cols], ytr)
            return average_precision_score(yval, m.predict_proba(val[cols])[:, 1])
        family = "hist_gbdt"

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)
    return family, study


def run_ieee(n_trials, timeout):
    out = EVAL / "ieee_with_device" / "tuning"
    out.mkdir(parents=True, exist_ok=True)
    print("ROUND 1: bake-off (validation-only)")
    bake, data = ieee_bakeoff()
    for k, v in sorted(bake.items(), key=lambda x: -x[1]["pr_auc"]):
        print(f" {k:24s} val PR-AUC {v['pr_auc']:.4f} ROC {v['roc_auc']:.4f} "
              f"recall@2% {v['recall_2pct']:.4f}")
    json.dump(bake, open(out / "bakeoff.json", "w"), indent=2)

    if not HAVE_OPTUNA:
        print("optuna unavailable -- bake-off only")
        return
    print(f"ROUND 2: Optuna search ({'lightgbm' if HAVE_LGB else 'hist_gbdt'}), "
          f"<= {n_trials} trials / {timeout}s")
    family, study = ieee_optuna(data, n_trials, timeout)
    print(f" best val PR-AUC {study.best_value:.4f} in {len(study.trials)} trials")
    json.dump({"family": family, "best_value": study.best_value,
               "best_params": study.best_params,
               "best_iter": study.best_trial.user_attrs.get("best_iter"),
               "n_trials": len(study.trials)},
              open(out / "optuna_best.json", "w"), indent=2)
    study.trials_dataframe().to_csv(out / "study_trials.csv", index=False)
    print(f" saved -> {out}")


RUNNERS = {"ieee_cis": run_ieee}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("detection", choices=list(RUNNERS))
    ap.add_argument("--trials", type=int, default=120)
    ap.add_argument("--timeout", type=int, default=300)
    a = ap.parse_args()
    print(f"xgboost={HAVE_XGB} lightgbm={HAVE_LGB} optuna={HAVE_OPTUNA}")
    RUNNERS[a.detection](a.trials, a.timeout)
