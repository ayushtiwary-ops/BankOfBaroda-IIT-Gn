"""Shared test fixtures for PRAMAAN.

These fixtures construct the *trusted-verifier* side (which holds private keys)
and the *engine* side (which holds only public keys) separately - mirroring the
real trust boundary, so a test can never accidentally let the engine mint its
own assertions.
"""
import sys
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parents[1] / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))


# --------------------------------------------------------------------------- #
# Key material - one Ed25519 keypair per trusted provider (step-up, device
# attestation, behavioral biometrics). The engine is configured with only the
# PUBLIC half of each.
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="session")
def keypairs():
    from app.verifier import Ed25519Signer

    return {
        "stepup": Ed25519Signer.generate(),
        "attest": Ed25519Signer.generate(),
        "behavior": Ed25519Signer.generate(),
    }


@pytest.fixture
def prod_env(keypairs):
    """A complete prod-mode environment dict (all required secrets present)."""
    import json

    return {
        "PRAMAAN_MODE": "prod",
        "PRAMAAN_EDGE_SECRET": "test-edge-secret-0123456789abcdef",
        "PRAMAAN_AUDIT_KEY": "test-audit-key-0123456789abcdef",
        "PRAMAAN_STEPUP_PUBKEY": keypairs["stepup"].public_key_b64,
        "PRAMAAN_ATTEST_PUBKEY": keypairs["attest"].public_key_b64,
        "PRAMAAN_BEHAVIOR_PUBKEY": keypairs["behavior"].public_key_b64,
        "PRAMAAN_API_KEYS": json.dumps({
            "edge-key-events": ["events:write"],
            "soc-key-readonly": ["audit:read", "identity:read"],
            "stepup-key": ["stepup:write"],
            "admin-key": ["events:write", "audit:read", "identity:read",
                          "stepup:write", "identity:erase"],
        }),
        "PRAMAAN_CORS_ORIGINS": "https://pramaan.demo,https://dashboard.local",
    }


@pytest.fixture
def stepup_provider(keypairs):
    """The trusted OTP/WebAuthn/video-KYC verifier service (holds private key)."""
    from app.verifier import TrustedVerifier

    return TrustedVerifier(keypairs["stepup"])


@pytest.fixture
def attest_provider(keypairs):
    """Models Play Integrity / App Attest - the device-attestation authority."""
    from app.attestation import DeviceAttestationProvider

    return DeviceAttestationProvider(keypairs["attest"])


@pytest.fixture
def behavior_provider(keypairs):
    """The on-device behavioural-biometrics signer (bound to an attested device)."""
    from app.attestation import BehaviorProvider

    return BehaviorProvider(keypairs["behavior"])


@pytest.fixture
def behavior_resolver(keypairs):
    """Engine-side resolver - holds only public keys; trusts nothing unsigned."""
    from app.attestation import BehaviorResolver
    from app.verifier import Ed25519Verifier, InMemoryNonceCache

    return BehaviorResolver(
        attest_verifier=Ed25519Verifier.from_b64(keypairs["attest"].public_key_b64),
        behavior_verifier=Ed25519Verifier.from_b64(keypairs["behavior"].public_key_b64),
        nonce_cache=InMemoryNonceCache(),
   )


@pytest.fixture
def stepup_validator(keypairs):
    """The engine-side validator (holds only the public key + a nonce cache)."""
    from app.verifier import AssertionValidator, Ed25519Verifier, InMemoryNonceCache

    return AssertionValidator(
        Ed25519Verifier.from_b64(keypairs["stepup"].public_key_b64),
        nonce_cache=InMemoryNonceCache(),
   )


# --------------------------------------------------------------------------- #
# A real (fitted) serving artifact written to a temp dir, with a model card +
# SHA-256 integrity digest - exercises the prod loader contract without needing
# the full RBA replay (that is covered by tests/test_export_models.py).
# --------------------------------------------------------------------------- #
@pytest.fixture
def serving_artifact(tmp_path):
    import numpy as np
    from app.features import FEATURE_NAMES
    from app.model_loader import save_artifact
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(7)
    x = rng.beta(1.5, 12, size=(800, len(FEATURE_NAMES)))  # benign-ish baseline
    scaler = StandardScaler().fit(x)
    model = IsolationForest(n_estimators=120, contamination=0.05,
                            random_state=7).fit(scaler.transform(x))
    out = tmp_path / "models"
    save_artifact(
        out, model=model, scaler=scaler, feature_names=FEATURE_NAMES,
        card={"name": "serving_anomaly", "dataset": "rba_wiefling_fixture",
              "dataset_url": "https://zenodo.org/records/6782156",
              "metric": {"roc_auc": 0.97}, "n_train": 800, "contamination": 0.05,
              "provenance": "rba_wiefling_fixture"},
   )
    return out


# --------------------------------------------------------------------------- #
# Engine fixtures - a demo-mode engine (real fitted synthetic model, in-memory
# store) and a "broken model" engine for resilience tests.
# --------------------------------------------------------------------------- #
@pytest.fixture
def demo_engine():
    from app.config import Settings
    from app.risk_engine import TrustEngine

    s = Settings.from_env({"PRAMAAN_MODE": "demo_synthetic"})
    # behavior_resolver=None → behaviour MISSING unless a test wires attestation.
    return TrustEngine(settings=s, behavior_resolver=None)


@pytest.fixture
def broken_model_engine():
    from app.config import Settings
    from app.risk_engine import TrustEngine

    class BrokenModel:
        provenance = "DEMO_SYNTHETIC"
        is_synthetic = True

        def risk(self, vec):
            raise RuntimeError("model unavailable")

    s = Settings.from_env({"PRAMAAN_MODE": "demo_synthetic"})
    return TrustEngine(settings=s, serving_model=BrokenModel(), behavior_resolver=None)


@pytest.fixture
def make_event():
    from app.schemas import Channel, EventType, IdentityEvent

    def _make(identity="u1", **kw):
        base = dict(
            identity_id=identity, event_type=EventType.LOGIN,
            channel=Channel.MOBILE_APP, device_id=f"dev_{identity}",
            geo="IN-GJ", hour_of_day=14,
       )
        base.update(kw)
        return IdentityEvent(**base)

    return _make


# --------------------------------------------------------------------------- #
# Full HTTP app under a real prod config (allowlisted CORS, scoped API keys,
# real model artifact). Reloads app.main so its module-level Settings/engine
# pick up this environment.
# --------------------------------------------------------------------------- #
@pytest.fixture
def api(prod_env, serving_artifact, monkeypatch):
    import importlib

    env = dict(prod_env)
    env["PRAMAAN_MODEL_DIR"] = str(serving_artifact)
    for k, v in env.items():
        monkeypatch.setenv(k, v)

    import app.main as main

    importlib.reload(main)
    from fastapi.testclient import TestClient

    return TestClient(main.app), main


def event_dict(identity="u1", **kw):
    base = dict(identity_id=identity, event_type="login", channel="mobile_app",
                device_id=f"dev_{identity}", geo="IN-GJ", hour_of_day=14)
    base.update(kw)
    return base
