"""Device attestation + signed behavioural assertions — the behaviour trust boundary.

SECURITY: KS2 — the behavioural-biometrics signal is no longer a float the
client sends. It is trusted ONLY when it arrives as:

  (1) a DEVICE-ATTESTATION token (models Play Integrity / App Attest) signed by
      the platform attestation authority, asserting the device passed integrity
      checks and binding a device_id; AND
  (2) a BEHAVIORAL-SIMILARITY assertion signed by the on-device biometrics
      provider, bound to that SAME attested device_id and to the identity.

The engine holds only the public verify keys. If either is absent, unsigned,
forged, stale, integrity-failed, or device-unbound, the behavioural signal is
treated as MISSING (cold-start neutral) — never as a trusted 0.99.

A server-side recompute path (``recompute_behavior_from_telemetry``) is the
explicit alternative trust boundary: score similarity server-side from raw
(already privacy-reduced) telemetry, so the server never has to believe a
client claim at all.

    ┌─────────── outside the trust boundary (client/device) ───────────┐
    │  raw keystroke/swipe timings  ──reduce on-device──►  similarity   │
    │  Play Integrity / App Attest  ──sign──►  attestation token        │
    └──────────────────────────────────────────────────────────────────┘
                              │ signed tokens only
    ┌──────────────── inside (server / engine) ────────────────────────┐
    │  verify signatures + freshness + device binding + verdict        │
    │  OR recompute_behavior_from_telemetry(reduced_telemetry)         │
    └──────────────────────────────────────────────────────────────────┘
"""
from __future__ import annotations

import math
import os
import time

from .verifier import (
    AssertionRejected,
    InMemoryNonceCache,
    open_token,
    sign_token,
)

INTEGRITY_PASS = "MEETS_DEVICE_INTEGRITY"


class DeviceAttestationProvider:
    """The platform attestation authority (Play Integrity / App Attest)."""

    def __init__(self, signer):
        self._signer = signer

    def issue(self, *, device_id: str, verdict: str = INTEGRITY_PASS,
              ttl_seconds: float = 300.0, nonce: str | None = None,
              issued_at: float | None = None) -> str:
        now = time.time() if issued_at is None else issued_at
        return sign_token(self._signer, {
            "kind": "device_attestation",
            "device_id": device_id,
            "verdict": verdict,
            "issued_at": now,
            "nonce": nonce or os.urandom(12).hex(),
            "exp": now + ttl_seconds,
        })


class BehaviorProvider:
    """On-device behavioural-biometrics signer, bound to an attested device."""

    def __init__(self, signer):
        self._signer = signer

    def issue(self, *, device_id: str, identity_id: str, similarity: float,
              ttl_seconds: float = 120.0, nonce: str | None = None,
              issued_at: float | None = None) -> str:
        now = time.time() if issued_at is None else issued_at
        return sign_token(self._signer, {
            "kind": "behavior_similarity",
            "device_id": device_id,
            "identity_id": identity_id,
            "similarity": float(similarity),
            "issued_at": now,
            "nonce": nonce or os.urandom(12).hex(),
            "exp": now + ttl_seconds,
        })


class BehaviorResolver:
    """Engine-side. Holds only public keys; returns a trusted similarity or None.

    None == MISSING == cold-start. The resolver NEVER raises into the scoring
    path on a bad token — an attacker should not be able to crash the engine,
    and a forged signal must simply be ignored (cold-start), not believed.
    """

    def __init__(self, *, attest_verifier, behavior_verifier,
                 nonce_cache: InMemoryNonceCache | None = None,
                 max_skew: float = 5.0):
        self._attest = attest_verifier
        self._behavior = behavior_verifier
        self._nonce = nonce_cache or InMemoryNonceCache()
        self._max_skew = max_skew

    def _fresh(self, payload: dict, now: float) -> bool:
        try:
            if now > float(payload["exp"]):
                return False
            if float(payload["issued_at"]) > now + self._max_skew:
                return False
        except (KeyError, TypeError, ValueError):
            return False
        return True

    def resolve(self, *, attestation_token: str | None, behavior_token: str | None,
                expected_identity_id: str, expected_device_id: str,
                now: float | None = None) -> float | None:
        now = time.time() if now is None else now
        if not attestation_token or not behavior_token:
            return None  # MISSING → cold-start
        try:
            att = open_token(self._attest, attestation_token)
            beh = open_token(self._behavior, behavior_token)
        except AssertionRejected:
            return None  # forged/unparseable → MISSING, never trusted

        # SECURITY: R2 — domain separation: each token must be its own kind
        # (prevents cross-protocol token confusion).
        if att.get("kind") != "device_attestation" or beh.get("kind") != "behavior_similarity":
            return None
        # freshness
        if not self._fresh(att, now) or not self._fresh(beh, now):
            return None
        # device integrity verdict must pass
        if att.get("verdict") != INTEGRITY_PASS:
            return None
        # binding: attestation device == behavior device == the event's device
        if att.get("device_id") != expected_device_id:
            return None
        if beh.get("device_id") != expected_device_id:
            return None
        if beh.get("identity_id") != expected_identity_id:
            return None
        # anti-replay on BOTH nonces (single-use, namespaced) — SECURITY: R2.
        att_nonce, beh_nonce = att.get("nonce"), beh.get("nonce")
        if not att_nonce or not beh_nonce:
            return None
        if self._nonce.seen(f"att:{att_nonce}") or self._nonce.seen(f"beh:{beh_nonce}"):
            return None

        sim = beh.get("similarity")
        try:
            sim = float(sim)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(sim):  # SECURITY: R2 — NaN/Inf must not clamp to 1.0
            return None

        self._nonce.add(f"att:{att_nonce}", float(att["exp"]))
        self._nonce.add(f"beh:{beh_nonce}", float(beh["exp"]))
        return max(0.0, min(1.0, sim))


def recompute_behavior_from_telemetry(template: list[float],
                                      observed: list[float]) -> float:
    """Server-side recompute path (explicit trust boundary, no client claim).

    Scaled-Manhattan similarity between an enrolled template and observed
    (privacy-reduced) telemetry — the same family of detector validated offline
    on the CMU keystroke dataset. Returns a similarity in [0, 1] where 1.0 is a
    perfect match. This is a stub of the on-server scorer; in production the
    template is held under the per-identity key (crypto-shredding).
    """
    if not template or len(template) != len(observed):
        return 0.0
    denom = sum(abs(t) for t in template) or 1.0
    dist = sum(abs(t - o) for t, o in zip(template, observed))
    return max(0.0, 1.0 - dist / denom)
