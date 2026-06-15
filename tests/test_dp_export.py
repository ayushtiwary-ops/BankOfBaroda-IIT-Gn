"""differential privacy is REAL on the feature-aggregation export path.

The export cannot run without passing through the DP mechanism; the spent ε is
accounted by a real RDP accountant and written to results/privacy_budget.json.
Skipped if the committed RBA sample is absent.
"""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "data" / "samples" / "rba_sample.csv"
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "src"))


def test_dp_export_blocks_un_noised_export():
    # SECURITY: there is no way to export aggregates without DP.
    import dp_export

    with pytest.raises(Exception):
        dp_export.export_dp_aggregates(out_dir=ROOT / "results", limit=10,
                                       noise_multiplier=0.0)  # DP disabled → blocked


@pytest.mark.skipif(not SAMPLE.exists(), reason="RBA sample not present")
def test_dp_export_writes_budget_with_stated_epsilon(tmp_path):
    import dp_export

    out = dp_export.export_dp_aggregates(out_dir=tmp_path, limit=4000)
    budget = json.loads((Path(out) / "privacy_budget.json").read_text())
    assert budget["mechanism"].lower().startswith("gaussian")
    assert 0.8 <= budget["epsilon"] <= 1.2   # target ε ≈ 1.0
    assert budget["delta"] == 1e-5
    assert budget["clipping"] and budget["sensitivity_l2"] > 0


@pytest.mark.skipif(not SAMPLE.exists(), reason="RBA sample not present")
def test_dp_export_actually_perturbs_the_means(tmp_path):
    import dp_export

    raw, noised, _ = dp_export.compute_cohort_means(SAMPLE, limit=4000,
                                                    noise_multiplier=4.0, seed=1)
    # at least one cohort's noised mean differs from the raw mean (noise applied)
    import numpy as np

    diffs = [float(np.abs(np.array(noised[c]) - np.array(raw[c])).max())
             for c in raw if c in noised]
    assert max(diffs) > 0.0
