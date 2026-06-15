"""Serving-model loader - connect the live engine to the REAL trained model.

SECURITY: the live engine no longer fits an IsolationForest on
``np.random``. ``src/export_models.py`` replays real RBA logins through the
SAME serving feature schema (``features.FEATURE_NAMES``) and persists a
versioned artifact here:

    results/models/serving_anomaly.joblib   (model + scaler + feature_names)
    results/models/model_card.json          (dataset, metric, train date,
                                             git SHA, SHA-256 of the joblib)

In ``prod`` mode the engine loads that artifact at startup and FAILS LOUD if it
is absent or tampered - there is no silent synthetic fallback. A clearly
labelled ``demo_synthetic`` mode (env flag) is the only way to run on a
synthetic baseline, and every such model is stamped ``DEMO_SYNTHETIC`` so the
provenance lands in every response and audit row.

Supply-chain note: ``joblib.load`` deserializes (pickle) and is therefore an
RCE surface. The AUTHORITATIVE integrity anchor is an OUT-OF-BAND pinned digest
(``PRAMAAN_MODEL_SHA256``, set at deploy time from config/KMS): the loader
refuses any artifact whose SHA-256 != the pin, BEFORE ``joblib.load``. The
hash recorded inside the (co-located) model card is only a corruption check -
it is written by whoever wrote the artifact, so it is NOT a trust boundary on
its own. Operators should set the pin in prod and mount the model dir read-only.
"""
from __future__ import annotations

import hashlib
import hmac
import json
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np

ARTIFACT_NAME = "serving_anomaly.joblib"
CARD_NAME = "model_card.json"


class ModelArtifactMissing(RuntimeError):
    """Prod mode: required model artifact is not present."""


class ModelArtifactInvalid(RuntimeError):
    """Artifact is present but corrupt / tampered / schema-mismatched."""


@dataclass
class ServingModel:
    model: object
    scaler: object
    feature_names: list[str]
    card: dict = field(default_factory=dict)
    provenance: str = "DEMO_SYNTHETIC"

    @property
    def is_synthetic(self) -> bool:
        return self.provenance == "DEMO_SYNTHETIC"

    def risk(self, vec) -> float:
        """Map a serving feature vector to an anomaly risk in [0, 1].

        IsolationForest.decision_function is + for normal, - for anomalous; the
        same affine map the original engine used keeps risk-band semantics."""
        x = self.scaler.transform([list(vec)])
        raw = float(self.model.decision_function(x)[0])
        return float(np.clip(0.5 - raw * 2.5, 0.0, 1.0))


# --------------------------------------------------------------------------- #
def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _consttime_eq(a: str, b: str) -> bool:
    return hmac.compare_digest(a, b)


def save_artifact(model_dir, *, model, scaler, feature_names, card: dict) -> Path:
    """Persist a serving artifact + a model card carrying its SHA-256 digest."""
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    artifact = model_dir / ARTIFACT_NAME
    joblib.dump(
        {"model": model, "scaler": scaler, "feature_names": list(feature_names)},
        artifact,
   )
    card = dict(card)
    card["feature_names"] = list(feature_names)
    card["artifact_sha256"] = _sha256(artifact)
    card.setdefault("provenance", card.get("name", "unknown"))
    (model_dir / CARD_NAME).write_text(json.dumps(card, indent=2, default=str))
    return artifact


def load_serving_model(settings, feature_names: list[str]) -> ServingModel:
    if settings.mode == "demo_synthetic":
        return _build_synthetic(feature_names)

    model_dir = Path(settings.model_dir)
    artifact = model_dir / ARTIFACT_NAME
    card_path = model_dir / CARD_NAME
    if not artifact.exists() or not card_path.exists():
        raise ModelArtifactMissing(
            f"prod mode requires a trained model at {artifact} (+ {CARD_NAME}); "
            f"regenerate it with `python src/export_models.py`. "
            f"Refusing to start on a synthetic fallback."
       )
    card = json.loads(card_path.read_text())

    # Integrity gate BEFORE deserializing the (pickle-backed) artifact.
    digest = _sha256(artifact)
    pinned = getattr(settings, "model_sha256", None)
    if pinned:
        # AUTHORITATIVE: the pin comes from config/KMS, not the model dir, so an
        # attacker who can write the dir still cannot match it .
        if not _consttime_eq(digest, pinned):
            raise ModelArtifactInvalid(
                f"artifact SHA-256 {digest} != pinned PRAMAAN_MODEL_SHA256 {pinned} "
                f"- refusing to load an unpinned/tampered model."
           )
    elif card.get("artifact_sha256") != digest:
        # No pin configured → fall back to the card's self-hash (corruption check
        # only; NOT tamper-proof - set PRAMAAN_MODEL_SHA256 in prod).
        raise ModelArtifactInvalid(
            f"artifact SHA-256 mismatch: card says {card.get('artifact_sha256')}, "
            f"file is {digest} - refusing to load a corrupt model."
       )

    bundle = joblib.load(artifact)
    if list(bundle.get("feature_names", [])) != list(feature_names):
        raise ModelArtifactInvalid(
            "artifact feature schema does not match the serving schema; "
            f"got {bundle.get('feature_names')}"
       )
    return ServingModel(
        model=bundle["model"],
        scaler=bundle["scaler"],
        feature_names=list(feature_names),
        card=card,
        provenance=card.get("provenance", card.get("name", "unknown")),
   )


def _build_synthetic(feature_names: list[str]) -> ServingModel:
    """The original np.random baseline - allowed ONLY in demo mode, stamped."""
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(42)

    def behavior(n):
        # Half of normal traffic carries an attested owner-like score (low
        # anomaly); the other half has MISSING behaviour (0.5 neutral) - so the
        # baseline treats "no behavioural signal yet" as normal, not anomalous.
        attested = rng.beta(1.5, 12, n)
        missing = np.full(n, 0.5)
        return np.where(rng.random(n) < 0.5, missing, attested)

    def block(n, criticality, channel, new_device, new_geo, hour):
        return np.column_stack([
            new_device(n), new_geo(n), hour(n),
            rng.beta(1.2, 10, n), behavior(n),
            criticality(n), channel(n),
            rng.binomial(1, 0.05, n), rng.binomial(1, 0.005, n),
            rng.beta(1.5, 15, n), rng.binomial(1, 0.02, n) * 0.33,
        ])

    retail = block(3500,
                   criticality=lambda n: rng.choice([0.2, 0.4, 0.5], n),
                   channel=lambda n: rng.choice([0.1, 0.3, 0.4], n),
                   new_device=lambda n: rng.binomial(1, 0.03, n),
                   new_geo=lambda n: rng.binomial(1, 0.02, n),
                   hour=lambda n: rng.beta(1.2, 8, n))
    admin = block(800,
                  criticality=lambda n: np.full(n, 0.9),
                  channel=lambda n: np.full(n, 0.7),
                  new_device=lambda n: rng.binomial(1, 0.02, n),
                  new_geo=lambda n: rng.binomial(1, 0.01, n),
                  hour=lambda n: rng.beta(1.2, 10, n))
    cold = block(300,
                 criticality=lambda n: rng.choice([0.2, 0.4, 0.6], n),
                 channel=lambda n: rng.choice([0.3, 0.4], n),
                 new_device=lambda n: np.full(n, 0.4),
                 new_geo=lambda n: np.full(n, 0.5),
                 hour=lambda n: np.full(n, 0.5))
    x = np.vstack([retail, admin, cold])
    scaler = StandardScaler().fit(x)
    model = IsolationForest(n_estimators=200, contamination=0.05,
                            random_state=42).fit(scaler.transform(x))
    return ServingModel(
        model=model, scaler=scaler, feature_names=list(feature_names),
        card={"name": "synthetic", "dataset": "np.random (DEMO ONLY)",
              "provenance": "DEMO_SYNTHETIC"},
        provenance="DEMO_SYNTHETIC",
   )
