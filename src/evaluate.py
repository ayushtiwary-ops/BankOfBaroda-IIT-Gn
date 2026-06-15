#!/usr/bin/env python3
"""
PRAMAAN - evaluate.py: regenerate every reported number from the scores tables.

    python src/evaluate.py paysim|ieee_cis|rba|cmu_keystroke|cert_insider|all

Reads data/processed/<ds>_scores.parquet produced by train.py and writes
results/<run>/metrics.json + curves. No model retraining happens here.
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
from sklearn.metrics import roc_curve

import pramaan_eval as pe

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
RES = ROOT / "results"


def eval_paysim() -> dict:
    df = pd.read_parquet(PROC / "paysim_scores.parquet")
    y = df.y.values
    base_recall = float(df.loc[y == 1, "flagged_baseline"].mean())
    base_fpr = float(df.loc[y == 0, "flagged_baseline"].mean())
    extra = {"builtin_rule_baseline": {"recall": round(base_recall, 4),
                                       "fpr": round(base_fpr, 6)},
             "split": "temporal: train step<=600, test step>600",
             "test_fraud_rate_pct": round(float(y.mean() * 100), 3)}
    m_full = pe.evaluate("paysim_full", y, df.score_full, extra=extra)
    m_beh = pe.evaluate("paysim_behavioral", y, df.score_behavioral_only, extra=extra)
    pe.compare_stepup("paysim_ablation", [
        ("full features", y, df.score_full.values),
        ("behavioral-only (no balance cols)", y, df.score_behavioral_only.values)])
    return {"full": m_full, "behavioral_only": m_beh}


def eval_ieee() -> dict:
    df = pd.read_parquet(PROC / "ieee_scores.parquet")
    y = df.y.values
    extra = {"split": "temporal: train first 80% of TransactionDT, test last 20%",
             "test_fraud_rate_pct": round(float(y.mean() * 100), 3)}
    m_a = pe.evaluate("ieee_txn_only", y, df.score_txn_only, extra=extra)
    m_b = pe.evaluate("ieee_with_device", y, df.score_with_device, extra=extra)
    sub = df[df.has_identity == 1]
    m_bs = pe.evaluate("ieee_with_device_identity_subset", sub.y, sub.score_with_device,
                       extra={"note": "transactions that carry device/identity data"})
    pe.compare_stepup("ieee_ablation", [
        ("transaction-only", y, df.score_txn_only.values),
        ("+ device/identity columns", y, df.score_with_device.values)])
    return {"txn_only": m_a, "with_device": m_b, "device_subset": m_bs}


def eval_rba() -> dict:
    df = pd.read_parquet(PROC / "rba_scores.parquet")
    y = df.y.values
    S = 0.02  # negative sampling rate (uniform) - see train.py rba_fit docstring
    extra = {"neg_sampling_note": "negatives uniformly sampled at ~2%; FPR/recall/ROC "
                                  "unbiased, precision/PR-AUC corrected by the rate",
             "split": "temporal 70/30 within sampled window; successful logins only"}
    m_noip = pe.evaluate("rba_noip", y, df.score_noip, neg_sample_rate=S, extra=extra)
    m_ip = pe.evaluate("rba_ip", y, df.score_ip, neg_sample_rate=S, extra=extra)
    m_base = pe.evaluate("rba_ipreputation_baseline", y, df.attack_ip_baseline,
                         neg_sample_rate=S,
                         extra={"note": "IP-reputation flag alone as the score"})
    pe.compare_stepup("rba_ablation", [
        ("behavioral + IP reputation", y, df.score_ip.values),
        ("behavioral only (no IP feed)", y, df.score_noip.values),
        ("IP reputation alone", y, df.attack_ip_baseline.values)],
        neg_sample_rate=S)
    return {"noip": m_noip, "ip": m_ip, "baseline": m_base}


def eval_cmu() -> dict:
    df = pd.read_parquet(PROC / "cmu_scores.parquet")
    eers = []
    for u, g in df.groupby("user"):
        fpr, tpr, _ = roc_curve(g.y, g.score)
        i = int(np.argmin(np.abs(fpr - (1 - tpr))))
        eers.append({"user": u, "eer": float((fpr[i] + 1 - tpr[i]) / 2)})
    e = pd.DataFrame(eers)
    out = RES / "cmu_keystroke"
    out.mkdir(parents=True, exist_ok=True)
    e.to_csv(out / "per_user_eer.csv", index=False)
    m = {"name": "cmu_keystroke_scaled_manhattan",
         "users": int(len(e)),
         "mean_eer": round(float(e.eer.mean()), 4),
         "median_eer": round(float(e.eer.median()), 4),
         "std_eer": round(float(e.eer.std()), 4),
         "literature_anchor": "Killourhy & Maxion DSN'09 scaled-Manhattan ≈ 0.096 mean EER",
         "protocol": "train 200 reps/user; genuine=last 200; impostors=5 reps × 50 others"}
    json.dump(m, open(out / "metrics.json", "w"), indent=2)
    plt.figure(figsize=(6, 4))
    plt.hist(e.eer, bins=20)
    plt.axvline(e.eer.mean(), ls="--", color="red",
                label=f"mean EER = {e.eer.mean():.3f}")
    plt.xlabel("per-user Equal Error Rate"); plt.ylabel("users")
    plt.title("CMU keystroke - scaled-Manhattan detector")
    plt.legend(); plt.tight_layout()
    plt.savefig(out / "eer_distribution.png", dpi=140); plt.close()
    return m


def eval_cert() -> dict:
    df = pd.read_parquet(PROC / "cert_scores.parquet")
    y = df.y.values
    res = {}
    for variant in ("score_iforest", "score_zsum"):
        res[variant] = pe.evaluate(f"cert_{variant}", y, df[variant],
                                   extra={"granularity": "user-day",
                                          "labels": "answers/insiders.csv r4.2 windows"})
    # analyst alert-budget sweep (top-k user-days per day, unsupervised iforest)
    days = df.day.astype(str)
    insiders = set(df.loc[df.y == 1, "user"])
    # scenario map (1=data theft on departure, 2=IP theft, 3=IT sabotage)
    ins = pd.read_csv(ROOT / "data" / "raw" / "cert_insider" / "answers" / "insiders.csv")
    ins = ins[ins.dataset.astype(str) == "4.2"][["user", "scenario"]].drop_duplicates()
    by_scen = {int(s): set(g.user) for s, g in ins.groupby("scenario")}
    budgets = [5, 10, 25, 50]
    sweep = []
    for k in budgets:
        flagged = df.assign(day=days).sort_values("score_iforest", ascending=False) \
                    .groupby("day").head(k)
        det = set(flagged[flagged.y == 1].user)
        f_mal = flagged[flagged.y == 1]
        sweep.append({
            "budget_per_day": k,
            "userday_recall": round(float(len(f_mal) / max(int(df.y.sum()), 1)), 4),
            "insider_users_detected": len(det),
            "insider_users_total": len(insiders),
            "user_detection_rate": round(len(det) / len(insiders), 4),
            "by_scenario": {f"scen{s}": f"{len(det & us)}/{len(us)}"
                            for s, us in sorted(by_scen.items())},
            "alerts_per_day": k,
        })
    out = RES / "cert_budget"
    out.mkdir(parents=True, exist_ok=True)
    json.dump(sweep, open(out / "budget_sweep.json", "w"), indent=2)
    plt.figure(figsize=(6, 4))
    plt.plot([s["budget_per_day"] for s in sweep],
             [s["user_detection_rate"] * 100 for s in sweep], marker="o")
    plt.xlabel("analyst alert budget (user-days flagged / day)")
    plt.ylabel("% of insiders detected (≥1 malicious day flagged)")
    plt.title("CERT r4.2 - detection vs analyst budget (unsupervised)")
    plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(out / "detection_vs_budget.png", dpi=140); plt.close()
    res["budget_sweep"] = sweep
    return res


RUNNERS = {"paysim": eval_paysim, "ieee_cis": eval_ieee, "rba": eval_rba,
           "cmu_keystroke": eval_cmu, "cert_insider": eval_cert}

if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    names = list(RUNNERS) if which == "all" else [which]
    for n in names:
        print(f"== evaluate {n} ==")
        m = RUNNERS[n]()
        print(json.dumps(m, indent=2, default=str)[:1200])
