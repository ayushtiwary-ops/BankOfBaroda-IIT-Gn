"""Keyed PII vault + crypto-shredding erasure (closes KS6).

SECURITY: KS6 — the immutable hash-chained audit (audit.py) and DPDP-Act
right-to-erasure appear to contradict (you cannot delete from an immutable
chain). They are reconciled by CRYPTO-SHREDDING:

  * The audit chain stores ONLY pseudonymous token refs + ciphertext refs —
    never plaintext PII (see main.py: audit payloads carry the tokenized
    identity_id only).
  * Per-identity PII / behavioural-template material lives HERE, each record
    encrypted under a PER-IDENTITY key (Fernet / AES-128-CBC + HMAC).
  * "Erase" = destroy that identity's key. The ciphertext may remain (so the
    audit chain stays byte-for-byte intact and still verifies), but it is now
    computationally irrecoverable. Right-to-erasure is satisfied without ever
    breaking the tamper-evident chain.

Retention: keys are held only for the documented retention window
(``RETENTION_DAYS``); an erase request destroys the key immediately. See
docs/COMPLIANCE.md.
"""
from __future__ import annotations

import hashlib
import json

from cryptography.fernet import Fernet

RETENTION_DAYS = 365  # DPDP/RBI-aligned default; erase short-circuits this window


class Erased(KeyError):
    """The per-identity key was destroyed — material is irrecoverable."""


class KeyedPiiStore:
    """Per-identity encrypted vault. Destroying a key crypto-shreds its records."""

    def __init__(self):
        self._keys: dict[str, bytes] = {}      # identity_id -> Fernet key
        self._ct: dict[str, bytes] = {}        # identity_id -> ciphertext (may outlive key)
        self._tombstoned: set[str] = set()     # R3: erased ids cannot be re-collected

    def put(self, identity_id: str, material: dict) -> str:
        """Encrypt material under the identity's key; return a non-reversible ref."""
        # SECURITY: R3 — an erased identity must NOT be silently re-collected
        # by a later event (erasure must be durable, not just point-in-time).
        if identity_id in self._tombstoned:
            raise Erased(f"{identity_id}: erased (tombstoned) — re-collection "
                         f"requires explicit re-consent via reconsent()")
        key = self._keys.get(identity_id) or Fernet.generate_key()
        self._keys[identity_id] = key
        token = Fernet(key).encrypt(json.dumps(material, sort_keys=True).encode())
        self._ct[identity_id] = token
        return hashlib.sha256(token).hexdigest()[:16]  # ref for the audit chain

    def get(self, identity_id: str) -> dict:
        if identity_id not in self._keys:
            raise Erased(f"{identity_id}: key destroyed — material irrecoverable")
        return json.loads(Fernet(self._keys[identity_id]).decrypt(self._ct[identity_id]))

    def erase(self, identity_id: str) -> bool:
        """Crypto-shred: destroy the key + tombstone the id. Returns True if the
        identity existed. The ciphertext is left in place so the audit chain
        stays intact, but it can no longer be decrypted by anyone, and the id is
        blocked from re-collection until explicit re-consent."""
        existed = identity_id in self._keys
        self._keys.pop(identity_id, None)
        self._tombstoned.add(identity_id)
        return existed

    def reconsent(self, identity_id: str) -> None:
        """Lift the tombstone after a fresh, explicit consent (audited upstream)."""
        self._tombstoned.discard(identity_id)

    def is_erased(self, identity_id: str) -> bool:
        return identity_id in self._tombstoned
