"""Signed step-up assertions — the trusted-verifier trust boundary.

SECURITY: KS1 — the step-up outcome must arrive as a cryptographically
SIGNED assertion minted by a trusted verifier (OTP / WebAuthn / video-KYC
provider). The scoring engine holds ONLY the public verify key, so it can
*check* an assertion but can never *mint* one. A client can never set its own
trust outcome.

Trust boundary, drawn in code
-----------------------------
    TrustedVerifier   (issue)   ── holds the PRIVATE key ──  outside the engine
    AssertionValidator(validate)── holds the PUBLIC key  ──  inside the engine

Assertion payload (signed, exactly as transmitted):
    {challenge_id, identity_id, method, result, issued_at, nonce, exp, alg}

The signature is computed over the *exact base64url payload segment* that
travels in the token (mini-JWS), never over a re-serialized copy — this removes
canonicalization-drift and signature-stripping classes of bug.

Ed25519 (asymmetric) is preferred; an HMAC-SHA256 (symmetric, KMS-style key)
backend is provided as a portable fallback for environments without
`cryptography`.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time
from dataclasses import dataclass

try:  # asymmetric, preferred — engine cannot forge what it can only verify
    from cryptography.exceptions import InvalidSignature as _InvalidSignature
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PrivateKey,
        Ed25519PublicKey,
    )

    _HAS_ED25519 = True
except Exception:  # pragma: no cover - exercised only on minimal installs
    _HAS_ED25519 = False


# --------------------------------------------------------------------------- #
# Rejection taxonomy — every failure mode is a distinct, catchable subclass.
# --------------------------------------------------------------------------- #
class AssertionRejected(Exception):
    """Base: the assertion is not trustworthy and must NOT move trust."""


class MalformedAssertion(AssertionRejected):
    """Token could not be parsed / required fields absent."""


class BadSignature(AssertionRejected):
    """Signature did not verify against the trusted public key."""


class ExpiredAssertion(AssertionRejected):
    """Outside the freshness window (expired or issued in the future)."""


class ReplayedNonce(AssertionRejected):
    """Nonce already consumed — anti-replay."""


class IdentityMismatch(AssertionRejected):
    """Assertion is bound to a different identity than the one being acted on."""


# --------------------------------------------------------------------------- #
# base64url without padding (JWS-style)
# --------------------------------------------------------------------------- #
def _b64e(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


def _b64d(seg: str) -> bytes:
    return base64.urlsafe_b64decode(seg + "=" * (-len(seg) % 4))


# --------------------------------------------------------------------------- #
# Signing backends
# --------------------------------------------------------------------------- #
class Ed25519Signer:
    """Holds the PRIVATE key. Lives in the trusted verifier, never the engine."""

    alg = "ed25519"

    def __init__(self, private_key: "Ed25519PrivateKey"):
        if not _HAS_ED25519:  # pragma: no cover
            raise RuntimeError("cryptography is required for Ed25519")
        self._priv = private_key

    @classmethod
    def generate(cls) -> "Ed25519Signer":
        return cls(Ed25519PrivateKey.generate())

    @classmethod
    def from_private_b64(cls, b64: str) -> "Ed25519Signer":
        """Reconstruct the signer from its 32-byte private seed (verifier service)."""
        return cls(Ed25519PrivateKey.from_private_bytes(_b64d(b64)))

    def sign(self, data: bytes) -> bytes:
        return self._priv.sign(data)

    @property
    def private_key_b64(self) -> str:
        return _b64e(self._priv.private_bytes_raw())

    @property
    def public_key_b64(self) -> str:
        return _b64e(self._priv.public_key().public_bytes_raw())


class Ed25519Verifier:
    """Holds ONLY the public key. Lives inside the engine."""

    alg = "ed25519"

    def __init__(self, public_key: "Ed25519PublicKey"):
        self._pub = public_key

    @classmethod
    def from_b64(cls, b64: str) -> "Ed25519Verifier":
        return cls(Ed25519PublicKey.from_public_bytes(_b64d(b64)))

    def verify(self, data: bytes, sig: bytes) -> bool:
        try:
            self._pub.verify(sig, data)
            return True
        except _InvalidSignature:
            return False


class HmacSigner:
    """Symmetric fallback. NOTE: symmetric → holder can both sign and verify;
    only use where an asymmetric backend is unavailable."""

    alg = "hmac-sha256"

    def __init__(self, key: bytes):
        self._key = key

    @classmethod
    def generate(cls) -> "HmacSigner":
        return cls(os.urandom(32))

    def sign(self, data: bytes) -> bytes:
        return hmac.new(self._key, data, hashlib.sha256).digest()

    @property
    def public_key_b64(self) -> str:  # symmetric: the "verify key" IS the key
        return _b64e(self._key)


class HmacVerifier:
    alg = "hmac-sha256"

    def __init__(self, key: bytes):
        self._key = key

    @classmethod
    def from_b64(cls, b64: str) -> "HmacVerifier":
        return cls(_b64d(b64))

    def verify(self, data: bytes, sig: bytes) -> bool:
        expected = hmac.new(self._key, data, hashlib.sha256).digest()
        return hmac.compare_digest(expected, sig)  # constant-time


def build_verifier(public_key_b64: str):
    """Pick an asymmetric verifier when possible, else HMAC."""
    if _HAS_ED25519:
        try:
            return Ed25519Verifier.from_b64(public_key_b64)
        except Exception:  # pragma: no cover - malformed/legacy key
            pass
    return HmacVerifier.from_b64(public_key_b64)


# --------------------------------------------------------------------------- #
# Reusable signed-token core (mini-JWS). Shared by step-up assertions AND by
# device-attestation / behavioral assertions (attestation.py) so there is one
# audited signing/verification path, not several hand-rolled ones.
# --------------------------------------------------------------------------- #
def sign_token(signer, payload: dict) -> str:
    seg = _b64e(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode())
    return f"{seg}.{_b64e(signer.sign(seg.encode()))}"


def open_token(verifier, token: str) -> dict:
    """Verify signature over the exact transmitted bytes; return the payload.
    Raises BadSignature/MalformedAssertion — never returns unverified content."""
    try:
        seg, sig_b64 = token.split(".")
        sig = _b64d(sig_b64)
        payload = json.loads(_b64d(seg))
    except Exception as exc:
        raise MalformedAssertion(f"unparseable token: {exc}") from None
    if not verifier.verify(seg.encode(), sig):
        raise BadSignature("signature does not verify against trusted key")
    return payload


# --------------------------------------------------------------------------- #
# Anti-replay nonce cache
# --------------------------------------------------------------------------- #
class InMemoryNonceCache:
    """Single-process nonce store with TTL. In prod, back this with Redis so the
    anti-replay guard holds across horizontally-scaled scoring pods."""

    def __init__(self):
        self._seen: dict[str, float] = {}

    def _prune(self, now: float) -> None:
        for n, exp in list(self._seen.items()):
            if exp < now:
                del self._seen[n]

    def seen(self, nonce: str) -> bool:
        self._prune(time.time())
        return nonce in self._seen

    def add(self, nonce: str, exp: float) -> None:
        self._seen[nonce] = exp


# --------------------------------------------------------------------------- #
# The signed assertion
# --------------------------------------------------------------------------- #
_FIELDS = ("challenge_id", "identity_id", "method", "result",
           "issued_at", "nonce", "exp")


@dataclass(frozen=True)
class StepUpAssertion:
    challenge_id: str
    identity_id: str
    method: str
    result: str
    issued_at: float
    nonce: str
    exp: float

    @property
    def passed(self) -> bool:
        return self.result == "pass"


class TrustedVerifier:
    """The OTP / WebAuthn / video-KYC provider. Holds the private signer."""

    def __init__(self, signer):
        self._signer = signer

    def issue(self, *, challenge_id: str, identity_id: str, method: str,
              result: str, ttl_seconds: float = 120.0,
              nonce: str | None = None, issued_at: float | None = None) -> str:
        now = time.time() if issued_at is None else issued_at
        payload = {
            "challenge_id": challenge_id,
            "identity_id": identity_id,
            "method": method,
            "result": result,
            "issued_at": now,
            "nonce": nonce or os.urandom(12).hex(),
            "exp": now + ttl_seconds,
            "alg": self._signer.alg,
        }
        return sign_token(self._signer, payload)


class AssertionValidator:
    """Engine-side. Holds only the public verifier + a nonce cache."""

    def __init__(self, verifier, nonce_cache: InMemoryNonceCache | None = None,
                 max_skew: float = 5.0):
        self._verifier = verifier
        self._nonce = nonce_cache or InMemoryNonceCache()
        self._max_skew = max_skew

    def validate(self, token: str, *, expected_identity_id: str,
                 now: float | None = None) -> StepUpAssertion:
        now = time.time() if now is None else now
        # 1+2) parse + verify signature BEFORE using any field
        payload = open_token(self._verifier, token)
        # 3) required fields present
        if any(f not in payload for f in _FIELDS):
            raise MalformedAssertion("assertion missing required fields")
        # 4) identity binding
        if payload["identity_id"] != expected_identity_id:
            raise IdentityMismatch(
                f"assertion bound to {payload['identity_id']!r}, "
                f"not {expected_identity_id!r}")
        # 5) freshness (expired OR issued in the future beyond clock skew)
        if now > float(payload["exp"]):
            raise ExpiredAssertion("assertion expired")
        if float(payload["issued_at"]) > now + self._max_skew:
            raise ExpiredAssertion("assertion issued in the future")
        # 6) anti-replay (single use)
        nonce = payload["nonce"]
        if self._nonce.seen(nonce):
            raise ReplayedNonce("nonce already consumed")
        self._nonce.add(nonce, float(payload["exp"]))
        return StepUpAssertion(**{k: payload[k] for k in _FIELDS})
