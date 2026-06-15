"""Smoke tests for the demonstration scripts - each runs on a small
slice and returns a sensible result. Skipped if the RBA sample is absent.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "data" / "samples" / "rba_sample.csv"
MODEL = ROOT / "results" / "models" / "serving_anomaly.joblib"
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "src"))

needs_data = pytest.mark.skipif(not SAMPLE.exists(), reason="RBA sample not present")
needs_model = pytest.mark.skipif(not MODEL.exists(), reason="serving artifact not present")


@needs_data
@needs_model
def test_coldstart_metric_reports_before_after():
    import coldstart_metric

    r = coldstart_metric.main(limit=600)
    ba = r["before_after"]
    assert 0.0 <= ba["after"] <= ba["before"] <= 1.0   # prior never increases friction


@needs_data
@needs_model
def test_fairness_audit_surfaces_disparity_and_rejects_parity():
    import fairness_audit

    r = fairness_audit.main(limit=4000)
    assert "disparate_impact_ratio" in r           # disparity surfaced honestly
    cost = r["rejected_mitigation"]["why_rejected_2_harmful"]
    # the rejected parity mitigation demonstrably LOWERS high-attack ATO recall
    assert (cost["hot_cohort_ato_recall_after_parity"]
            < cost["hot_cohort_ato_recall_global_thr"])


@needs_data
def test_impossible_travel_lift_runs_and_is_honest():
    import impossible_travel_lift

    r = impossible_travel_lift.main()
    assert "caveat" in r and r["n_ato"] > 0
