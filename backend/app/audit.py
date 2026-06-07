"""Tamper-evident audit trail — the SOC / regulator plane.

Every assessment is appended to a hash chain (each record carries the SHA-256
of its predecessor). Any retroactive edit breaks the chain.

HARDENING (audit hardening): a *keyless* SHA chain is only tamper-evident
against a lazy attacker — anyone who can rewrite the store can recompute every
subsequent hash and forge a consistent chain. When constructed with a SOC
``signing_key`` each record also carries ``hmac = HMAC-SHA256(key, record_hash)``,
so a recompute attack fails without the key. The plain (keyless) chain remains
supported for the in-memory demo.

This is also the ONLY plane that carries detector reason codes + feature
contributions (KS8): rich explanations go to analysts/audit, never to the
client.
"""
import hashlib
import hmac
import json
import time


class AuditLog:
    GENESIS = "0" * 64

    def __init__(self, signing_key: bytes | None = None):
        self.records: list[dict] = []
        self._key = signing_key

    def _hash(self, body: dict) -> str:
        return hashlib.sha256(
            json.dumps(body, sort_keys=True, default=str).encode()
        ).hexdigest()

    def _mac(self, record_hash: str) -> str:
        return hmac.new(self._key, record_hash.encode(), hashlib.sha256).hexdigest()

    def append(self, payload: dict) -> dict:
        prev_hash = self.records[-1]["hash"] if self.records else self.GENESIS
        record = {
            "seq": len(self.records),
            "ts": time.time(),
            "payload": payload,
            "prev_hash": prev_hash,
        }
        record["hash"] = self._hash(record)  # body excludes hash/hmac (not yet set)
        if self._key is not None:
            record["hmac"] = self._mac(record["hash"])
        self.records.append(record)
        return record

    def verify_chain(self) -> bool:
        prev = self.GENESIS
        for idx, r in enumerate(self.records):
            if r.get("seq") != idx:  # R2: seq must equal index (reorder/gap guard)
                return False
            if r["prev_hash"] != prev:
                return False
            body = {k: v for k, v in r.items() if k not in ("hash", "hmac")}
            if self._hash(body) != r["hash"]:
                return False
            if self._key is not None:
                expected = self._mac(r["hash"])
                if not hmac.compare_digest(expected, r.get("hmac", "")):
                    return False
            prev = r["hash"]
        return True

    def head_checkpoint(self) -> str:
        """A signed commitment to the chain LENGTH + head hash.

        SECURITY: R2 — ``verify_chain`` alone cannot detect tail-truncation
        (a genuine prefix is itself a valid chain). The SOC persists this
        checkpoint out-of-band; ``verify_against_checkpoint`` then catches a
        dropped/emptied tail that no attacker can re-forge without the key."""
        last = self.records[-1]["hash"] if self.records else self.GENESIS
        body = f"{len(self.records)}|{last}"
        if self._key is None:
            return body
        return body + "|" + hmac.new(self._key, body.encode(), hashlib.sha256).hexdigest()

    def verify_against_checkpoint(self, expected: str) -> bool:
        return self.verify_chain() and hmac.compare_digest(self.head_checkpoint(), expected)

    def tail(self, n: int = 50) -> list[dict]:
        return self.records[-n:]
