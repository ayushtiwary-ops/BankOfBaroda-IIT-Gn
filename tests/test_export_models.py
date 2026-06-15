"""reproducibility - the serving artifact regenerates from REAL data.

Skipped automatically if the committed RBA sample is not present (e.g. a shallow
CI checkout); when present, it proves the offline → serving export produces a
loadable artifact trained on real logins, not np.random.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "data" / "samples" / "rba_sample.csv"
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "src"))


@pytest.mark.skipif(not SAMPLE.exists(), reason="RBA sample not present")
def test_export_builds_a_loadable_real_artifact(tmp_path):
    from app.config import Settings
    from app.features import FEATURE_NAMES
    from app.model_loader import load_serving_model

    import export_models

    out = export_models.export(out_dir=tmp_path, limit=3000)
    settings = Settings(
        mode="prod", edge_secret=b"x", audit_signing_key=b"x",
        stepup_pubkey="", attest_pubkey="", behavior_pubkey="",
        api_keys={}, cors_origins=["x"], model_dir=Path(out), redis_url=None,
   )
    model = load_serving_model(settings, FEATURE_NAMES)
    assert model.is_synthetic is False
    assert "rba" in model.provenance.lower()
    assert "rba" in model.card["dataset"].lower()
    assert model.card["n_train"] > 0
    assert 0.0 <= model.risk([0.0] * len(FEATURE_NAMES)) <= 1.0
