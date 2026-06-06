"""Central configuration — secrets come from env/KMS, never from defaults.

SECURITY: KS4(d) — every required secret is read from the environment with
NO default. In prod mode a process refuses to start if any is missing, so a
service can never silently run on a shipped secret (the old
``PRAMAAN_EDGE_SECRET="demo-edge-secret"`` default is gone).

A clearly-labelled ``demo_synthetic`` mode is the ONLY way to run without real
secrets; it self-identifies (``is_synthetic``) so a synthetic run can never be
mistaken for a trustworthy one, and that label is stamped into every response
and audit row downstream (KS3).
"""
from __future__ import annotations

import json
import os
import secrets
from dataclasses import dataclass
from pathlib import Path


class ConfigError(RuntimeError):
    """A required secret/config value is missing or invalid."""


# var -> human description (order = report order on failure)
REQUIRED_PROD = {
    "PRAMAAN_EDGE_SECRET": "edge PII-tokenization secret",
    "PRAMAAN_AUDIT_KEY": "audit-chain signing key",
    "PRAMAAN_STEPUP_PUBKEY": "step-up verifier public key",
    "PRAMAAN_ATTEST_PUBKEY": "device-attestation public key",
    "PRAMAAN_BEHAVIOR_PUBKEY": "behavioral-assertion public key",
    "PRAMAAN_API_KEYS": "API key -> scopes registry (JSON)",
    "PRAMAAN_CORS_ORIGINS": "CORS allowlist (comma-separated)",
}

VALID_SCOPES = frozenset(
    {"events:write", "audit:read", "identity:read", "stepup:write", "identity:erase"}
)

_DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[2] / "results" / "models"


@dataclass
class Settings:
    mode: str  # "prod" | "demo_synthetic"
    edge_secret: bytes
    audit_signing_key: bytes
    stepup_pubkey: str
    attest_pubkey: str
    behavior_pubkey: str
    api_keys: dict[str, frozenset[str]]
    cors_origins: list[str]
    model_dir: Path
    redis_url: str | None
    is_synthetic: bool = False
    # SECURITY: R2 — out-of-band pinned model digest (the AUTHORITATIVE
    # integrity anchor; a co-located card hash is only a corruption check).
    model_sha256: str | None = None

    # ---------------------------------------------------------------- #
    @classmethod
    def from_env(cls, env: dict | None = None) -> "Settings":
        env = dict(os.environ if env is None else env)
        mode = env.get("PRAMAAN_MODE", "prod")
        if mode not in ("prod", "demo_synthetic"):
            raise ConfigError(
                f"PRAMAAN_MODE must be 'prod' or 'demo_synthetic', got {mode!r}"
            )

        synthetic = False
        if mode == "prod":
            missing = [k for k in REQUIRED_PROD if not env.get(k)]
            if missing:
                detail = ", ".join(f"{k} ({REQUIRED_PROD[k]})" for k in missing)
                raise ConfigError(
                    "prod mode requires every secret via env/KMS — missing: " + detail
                )
        else:
            synthetic = True
            env.setdefault("PRAMAAN_EDGE_SECRET", "DEMO-" + secrets.token_hex(16))
            env.setdefault("PRAMAAN_AUDIT_KEY", "DEMO-" + secrets.token_hex(16))
            for k in ("PRAMAAN_STEPUP_PUBKEY", "PRAMAAN_ATTEST_PUBKEY",
                      "PRAMAAN_BEHAVIOR_PUBKEY"):
                if not env.get(k):
                    from .verifier import Ed25519Signer

                    env[k] = Ed25519Signer.generate().public_key_b64
            env.setdefault("PRAMAAN_API_KEYS", json.dumps(
                {"demo-key": sorted(VALID_SCOPES)}))
            env.setdefault("PRAMAAN_CORS_ORIGINS", "http://localhost:8000")

        try:
            api_keys_raw = json.loads(env["PRAMAAN_API_KEYS"])
        except json.JSONDecodeError as exc:
            raise ConfigError(f"PRAMAAN_API_KEYS is not valid JSON: {exc}") from None
        api_keys: dict[str, frozenset[str]] = {}
        for key, scopes in api_keys_raw.items():
            bad = set(scopes) - VALID_SCOPES
            if bad:
                raise ConfigError(f"unknown scope(s) {sorted(bad)} for an API key")
            api_keys[key] = frozenset(scopes)

        cors = [o.strip() for o in env["PRAMAAN_CORS_ORIGINS"].split(",") if o.strip()]
        if "*" in cors:
            # SECURITY: KS4(a) — wildcard CORS is forbidden, full stop.
            raise ConfigError('CORS wildcard "*" is forbidden; set an explicit allowlist')

        # SECURITY: R2 — a malformed verifier public key must fail loud, not
        # silently downgrade Ed25519 → symmetric HMAC (asymmetric trust collapse).
        from .verifier import _HAS_ED25519, Ed25519Verifier

        if _HAS_ED25519:
            for name in ("PRAMAAN_STEPUP_PUBKEY", "PRAMAAN_ATTEST_PUBKEY",
                         "PRAMAAN_BEHAVIOR_PUBKEY"):
                try:
                    Ed25519Verifier.from_b64(env[name])
                except Exception:
                    raise ConfigError(
                        f"{name} is not a valid Ed25519 public key (32 bytes, base64url)"
                    ) from None

        model_dir = Path(env.get("PRAMAAN_MODEL_DIR", str(_DEFAULT_MODEL_DIR)))

        return cls(
            mode=mode,
            edge_secret=env["PRAMAAN_EDGE_SECRET"].encode(),
            audit_signing_key=env["PRAMAAN_AUDIT_KEY"].encode(),
            stepup_pubkey=env["PRAMAAN_STEPUP_PUBKEY"],
            attest_pubkey=env["PRAMAAN_ATTEST_PUBKEY"],
            behavior_pubkey=env["PRAMAAN_BEHAVIOR_PUBKEY"],
            api_keys=api_keys,
            cors_origins=cors,
            model_dir=model_dir,
            redis_url=env.get("PRAMAAN_REDIS_URL") or None,
            is_synthetic=synthetic,
            model_sha256=env.get("PRAMAAN_MODEL_SHA256") or None,
        )

    # ---- engine-side factories (hold only public keys) ------------------- #
    def stepup_validator(self, nonce_cache=None):
        from .verifier import AssertionValidator, build_verifier

        return AssertionValidator(build_verifier(self.stepup_pubkey),
                                  nonce_cache=nonce_cache)
