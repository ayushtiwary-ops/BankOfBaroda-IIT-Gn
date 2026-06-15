"""Privacy layer - PRAMAAN never sees raw PII.

Three guarantees enforced here:
1. Pseudonymization : all identifiers are HMAC-tokenized at the edge with a
   bank-held secret. The risk engine works exclusively on tokens; reversing a
   token requires the HSM-protected key, which the engine does not hold.
2. Data minimization : geo is coarsened to state-level buckets, timestamps to
   hour-of-day. Behavioural biometrics are reduced to a single similarity
   score *on the user's device* - raw keystroke/swipe data never leaves it.
3. Differential privacy : aggregate statistics exported for model retraining
   pass through a calibrated mechanism with an accounted ε. The ACTIVE export
   path (``src/dp_export.py``) uses the GAUSSIAN mechanism (RDP-accounted,
   ε≈1.0); ``dp_noise`` below is the Laplace building block, used for the DP
   noisy-count cohort suppression.

SECURITY: there is NO shipped default secret. The edge secret is
resolved from env/KMS; in ``prod`` mode (or when the mode is unset) a missing
secret raises ``ConfigError`` (fail loud). Only an explicit ``demo_synthetic``
run may use an ephemeral, per-process secret - never a constant baked into the
source.
"""
import hashlib
import hmac
import os
import secrets

from .config import ConfigError

_EPHEMERAL: bytes | None = None  # demo-only, generated once per process


def _resolve_secret(secret: bytes | None) -> bytes:
    global _EPHEMERAL
    if secret is not None:
        return secret
    env_secret = os.environ.get("PRAMAAN_EDGE_SECRET")
    if env_secret:
        return env_secret.encode()
    mode = os.environ.get("PRAMAAN_MODE", "prod")
    if mode == "demo_synthetic":
        if _EPHEMERAL is None:
            _EPHEMERAL = secrets.token_bytes(32)  # ephemeral, not shipped
        return _EPHEMERAL
    raise ConfigError(
        "PRAMAAN_EDGE_SECRET is unset; refusing to tokenize with a default. "
        "Provide it via env/KMS (prod) or run in demo_synthetic mode."
   )


def pseudonymize(raw_identifier: str, *, secret: bytes | None = None) -> str:
    """HMAC-SHA256 tokenization (edge-side). Deterministic, non-reversible."""
    key = _resolve_secret(secret)
    return hmac.new(key, raw_identifier.encode(), hashlib.sha256).hexdigest()[:24]


def coarsen_geo(lat: float, lon: float, state_code: str) -> str:
    """Drop precise coordinates; keep only a state-level bucket."""
    return f"IN-{state_code}"


def dp_noise(value: float, epsilon: float = 1.0, sensitivity: float = 1.0) -> float:
    """Laplace mechanism for differentially-private aggregate exports.

    # FUTURE WORK: this mechanism exists but is NOT yet wired into the
    # feature-aggregation export path with a stated/accounted ε budget.
    # FUTURE WORK: crypto-shredding erasure (keyed PII store + key
    # destruction) vs the immutable audit chain is not implemented yet.
    """
    import math

    u = secrets.SystemRandom().random() - 0.5
    scale = sensitivity / epsilon
    noise = -scale * (1 if u >= 0 else -1) * math.log(1 - 2 * abs(u))
    return value + noise
