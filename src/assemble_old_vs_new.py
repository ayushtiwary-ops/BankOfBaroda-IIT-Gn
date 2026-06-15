#!/usr/bin/env python3
"""
PRAMAAN -- assemble_old_vs_new.py: collate the per-detection keep/replace decision.

Reads the artifacts written by evaluate_tuned.py, eval_full.py and phaseb_extra.py
and writes a single results/evaluation/old_vs_new.json summarising, per detection,
the incumbent metric, the tuned/alternative metric, and the decision. The IEEE
entry (with its paired bootstrap delta) is preserved as written by evaluate_tuned.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "results" / "evaluation"


def _load(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None


def main():
    ovn_path = EVAL / "old_vs_new.json"
    out = _load(ovn_path) or {}

    # RBA: full-data run supersedes the unreproducible sample
    full = _load(EVAL / "rba_full_noip" / "metrics_full.json")
    sample = _load(ROOT / "results" / "rba_noip" / "metrics.json")
    if full:
        out["rba"] = {
            "detection": "rba",
            "incumbent": {"model": "sample-based HGB (unreproducible 2% neg sample)",
                          "roc_auc": (sample or {}).get("roc_auc"),
                          "recall_1pct": ((sample or {}).get("at_fixed_fpr", {}).get("fpr_1pct", {}).get("recall"))},
            "replacement": {"model": "full-data HGB (6.79M-login test, no IP feed)",
                            "roc_auc": full["threshold_free"]["roc_auc"],
                            "recall_1pct": full["operating_points"]["stepup_1pct"]["recall"],
                            "recall_2pct": full["operating_points"]["stepup_2pct"]["recall"]},
            "decision": "replace",
            "decision_rule": "the committed sample scores did not reproduce from code; the "
            "full-data stream is reproducible AND improved the held-out numbers",
        }

    # PaySim: behaviour-only bake-off (incumbent kept unless clearly beaten)
    pb = _load(EVAL / "paysim_full" / "tuning" / "bakeoff.json")
    if pb:
        bake = pb.get("behavioral_only_bakeoff_val", {})
        inc = bake.get("hist_gbdt_incumbent", {}).get("pr_auc")
        lgbv = bake.get("lightgbm", {}).get("pr_auc")
        decision = ("keep_incumbent" if (lgbv is None or lgbv <= (inc or 0) + 0.01)
                    else "lightgbm wins on val; verify on test before replacing")
        out["paysim"] = {
            "detection": "paysim",
            "incumbent_val_pr_auc_behavioral": inc,
            "lightgbm_val_pr_auc_behavioral": lgbv,
            "full_features": "ROC and PR-AUC at separability ceiling (1.000); reported as upper "
            "bound, behaviour-only is the realistic arm",
            "leak_robustness": pb.get("leak_robustness_full"),
            "decision": decision,
        }

    # CMU: scaled-Manhattan vs Mahalanobis
    cm = _load(EVAL / "cmu_keystroke" / "tuning" / "bakeoff.json")
    if cm:
        out["cmu_keystroke"] = {
            "detection": "cmu_keystroke",
            "incumbent_scaled_manhattan_eer": cm.get("scaled_manhattan_mean_eer"),
            "alternative_mahalanobis_eer": cm.get("mahalanobis_mean_eer"),
            "decision": "keep_incumbent",
            "decision_rule": "scaled-Manhattan has lower mean EER and is the published anchor",
        }

    # CERT: unsupervised incumbent vs supervised upper bound
    ce = _load(EVAL / "cert_score_iforest" / "tuning" / "bakeoff.json")
    if ce:
        out["cert"] = {
            "detection": "cert",
            "incumbent_unsupervised": ce.get("incumbent_unsupervised_iforest"),
            "supervised_upper_bound_not_deployable": ce.get("supervised_user_grouped_upper_bound"),
            "decision": "keep_incumbent",
            "decision_rule": "production has no insider labels at train time; the supervised "
            "number is context only",
        }

    json.dump(out, open(ovn_path, "w"), indent=2)
    print(f"wrote {ovn_path} with detections: {sorted(out.keys())}")
    for k, v in out.items():
        print(f" {k}: decision={v.get('decision')}")


if __name__ == "__main__":
    main()
