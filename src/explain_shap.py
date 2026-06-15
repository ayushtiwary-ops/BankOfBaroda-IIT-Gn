#!/usr/bin/env python3
"""Real SHAP attribution for the serving IsolationForest (P5 explainability).

    python src/explain_shap.py

Uses the actual ``shap`` library to attribute the live anomaly model's score to
the 11 serving features (global mean |SHAP| + a few worked examples). The live
audit plane carries exact additive Shapley for the deterministic component
(cheap, per-decision); this script validates/visualises the heavier ML SHAP
offline.

Outputs:
  results/explainability/shap_summary.json
  results/explainability/shap_bar.png
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "src"))

from app.config import Settings  # noqa: E402
from app.features import FEATURE_NAMES  # noqa: E402
from app.model_loader import load_serving_model  # noqa: E402

from export_models import SAMPLE, build_serving_vectors  # noqa: E402

OUT = ROOT / "results" / "explainability"


def main(limit: int = 3000):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap

    prod = Settings(mode="prod", edge_secret=b"x", audit_signing_key=b"x",
                    stepup_pubkey="", attest_pubkey="", behavior_pubkey="",
                    api_keys={}, cors_origins=["x"],
                    model_dir=ROOT / "results" / "models", redis_url=None)
    model = load_serving_model(prod, FEATURE_NAMES)

    x = build_serving_vectors(SAMPLE, limit=limit)
    xs = model.scaler.transform(x)
    rng = np.random.default_rng(0)
    bg = xs[rng.choice(len(xs), size=min(50, len(xs)), replace=False)]
    explain = xs[rng.choice(len(xs), size=min(200, len(xs)), replace=False)]

    explainer = shap.Explainer(model.model.decision_function, bg)
    sv = explainer(explain, silent=True)
    mean_abs = np.abs(sv.values).mean(axis=0)
    order = np.argsort(mean_abs)[::-1]

    OUT.mkdir(parents=True, exist_ok=True)
    summary = {
        "detector": "IsolationForest (serving) - real SHAP via shap library",
        "n_background": int(len(bg)), "n_explained": int(len(explain)),
        "global_mean_abs_shap": {FEATURE_NAMES[i]: round(float(mean_abs[i]), 5)
                                 for i in order},
    }
    (OUT / "shap_summary.json").write_text(json.dumps(summary, indent=2))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh([FEATURE_NAMES[i] for i in order][::-1],
            [mean_abs[i] for i in order][::-1], color="#2dd4a7")
    ax.set_xlabel("mean |SHAP| (impact on anomaly score)")
    ax.set_title("Serving anomaly model - global feature importance (SHAP)")
    fig.tight_layout()
    fig.savefig(OUT / "shap_bar.png", dpi=110)

    top = [FEATURE_NAMES[i] for i in order[:3]]
    print(f"SHAP top features: {top}  -> {OUT}/shap_summary.json")
    return summary


if __name__ == "__main__":
    main()
