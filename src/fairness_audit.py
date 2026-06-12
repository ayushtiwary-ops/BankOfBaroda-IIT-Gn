#!/usr/bin/env python3
"""Fairness / disparate-impact audit across geo cohorts (P5) — HONEST version.

    python src/fairness_audit.py

Reports the per-cohort step-up (selection) rate + the disparate-impact ratio on
real RBA logins. It then DOES NOT claim a selection-rate-parity "fix": on a
security detector, equalizing selection rate by per-cohort thresholds is both a
quantile tautology (post-hoc rate ≡ target by construction) AND harmful (it
raises the threshold in higher-attack cohorts, dropping ATO recall). We
demonstrate that cost on a controlled simulation and instead recommend
equalized-odds monitoring + feature-level treatment (geo enters only as per-user
new_geo, never as an absolute country feature).

Output: results/fairness/report.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "src"))

from app.config import Settings  # noqa: E402
from app.features import FEATURE_NAMES  # noqa: E402
from app.model_loader import load_serving_model  # noqa: E402
from app.risk_engine import TrustEngine  # noqa: E402
from app.schemas import Decision  # noqa: E402

from export_models import SAMPLE, _row_to_event  # noqa: E402

OUT = ROOT / "results" / "fairness"
MIN_COHORT = 200
DI_THRESHOLD = 0.8  # 4/5ths rule


def _real_model():
    prod = Settings(mode="prod", edge_secret=b"x", audit_signing_key=b"x",
                    stepup_pubkey="", attest_pubkey="", behavior_pubkey="",
                    api_keys={}, cors_origins=["x"],
                    model_dir=ROOT / "results" / "models", redis_url=None)
    return load_serving_model(prod, FEATURE_NAMES)


def _parity_cost_simulation():
    """Show WHY selection-rate parity is rejected: equalizing the step-up rate
    raises the threshold in a higher-attack cohort and collapses its ATO recall."""
    rng = np.random.default_rng(0)
    # two cohorts, same detector, different true ATO base rate
    def cohort(n, ato_rate):
        y = (rng.random(n) < ato_rate).astype(int)
        risk = np.where(y == 1, rng.beta(7, 3, n), rng.beta(2, 8, n))  # attacks score higher
        return y, risk
    _, rs = cohort(5000, 0.02)       # "safe" cohort (labels unused here)
    yh, rh = cohort(5000, 0.20)      # "hot" cohort (more attackers)
    g_thr = 0.45                     # single global threshold

    def recall(y, r, t):
        return float(((r >= t) & (y == 1)).sum() / max((y == 1).sum(), 1))
    # parity: raise hot-cohort threshold to match the safe cohort's selection rate
    safe_sel = float((rs >= g_thr).mean())
    hot_thr = float(np.quantile(rh, 1 - safe_sel))
    return {
        "global_threshold": g_thr,
        "hot_cohort_ato_recall_global_thr": round(recall(yh, rh, g_thr), 3),
        "hot_cohort_threshold_after_parity": round(hot_thr, 3),
        "hot_cohort_ato_recall_after_parity": round(recall(yh, rh, hot_thr), 3),
        "interpretation": ("Equalizing selection rate raises the high-attack "
                           "cohort's threshold and drops its ATO recall — parity "
                           "buys a cosmetic 4/5ths pass by letting real "
                           "account-takeovers through. Rejected."),
    }


def main(limit: int | None = 20000):
    df = pd.read_csv(SAMPLE)
    df.columns = [c.strip() for c in df.columns]
    ts = pd.to_datetime(df["Login Timestamp"], errors="coerce")
    df = df.assign(_ts=ts).dropna(subset=["_ts"])
    df["_succ"] = df["Login Successful"].astype(str).str.lower().eq("true")
    df["_ato"] = df["Is Account Takeover"].astype(str).str.lower().eq("true")
    df["_hour"] = df["_ts"].dt.hour.fillna(0).astype(int)
    g = df[df["_succ"] & ~df["_ato"]].sort_values(["User ID", "_ts"])
    if limit:
        g = g.head(limit)
    g = g.rename(columns={
        "User ID": "user", "Device Type": "device_type",
        "Browser Name and Version": "browser", "OS Name and Version": "os",
        "Country": "country", "_hour": "hod"})

    s = Settings.from_env({"PRAMAAN_MODE": "demo_synthetic"})
    engine = TrustEngine(settings=s, serving_model=_real_model(), behavior_resolver=None)

    by_cohort: dict[str, list[int]] = {}
    for r in g[["user", "device_type", "browser", "os", "country", "hod"]].itertuples(index=False):
        e = _row_to_event(r.user, r.device_type, r.browser, r.os, r.country, r.hod)
        soc = engine.assess(e)
        by_cohort.setdefault(str(r.country), []).append(int(soc.decision != Decision.ALLOW))

    rates = {c: float(np.mean(v)) for c, v in by_cohort.items() if len(v) >= MIN_COHORT}
    di = (min(rates.values()) / max(rates.values())) if rates and max(rates.values()) > 0 else 1.0

    OUT.mkdir(parents=True, exist_ok=True)
    report = {
        "protected_attribute": "geo cohort (country)",
        "selection_rate_definition": "fraction of genuine logins challenged (STEP_UP/BLOCK)",
        "cohorts_audited": len(rates),
        "per_cohort_selection_rate": {c: round(r, 4) for c, r in sorted(rates.items())},
        "disparate_impact_ratio": round(di, 4),
        "passes_4_5ths_rule": bool(di >= DI_THRESHOLD),
        "rejected_mitigation": {
            "name": "per-cohort threshold calibration to equalize selection rate",
            "why_rejected_1_tautology": ("post-hoc rate ≡ target by construction "
                                         "(quantile identity) → 'after' DI is forced "
                                         "to ~1.0 and proves nothing"),
            "why_rejected_2_harmful": _parity_cost_simulation(),
        },
        "recommended": {
            "feature_level": ("geo enters the model ONLY as per-user new_geo (is this "
                              "bucket new for THIS user), never as an absolute country "
                              "feature — so the model does not key on nationality"),
            "metric": ("monitor equalized-odds: per-cohort ATO RECALL and FPR at a "
                       "SINGLE global threshold, not selection-rate parity"),
        },
        "note": ("Disparity is surfaced honestly. We deliberately do NOT equalize "
                 "selection rate on a security detector."),
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2))
    print(f"geo-cohort disparate-impact ratio = {di:.3f} over {len(rates)} cohorts; "
          f"selection-rate parity REJECTED (tautology + harmful) -> {OUT}/report.json")
    return report


if __name__ == "__main__":
    main()
