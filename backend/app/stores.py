"""Postgres-backed audit chain + keyed PII vault (executable-architecture).

These are the PRODUCTION implementations of the same interfaces the in-memory
``AuditLog`` (audit.py) and ``KeyedPiiStore`` (keystore.py) expose — so the
docker-compose topology runs the REAL durable stores while tests keep using the
in-memory ones. Selected by env (``PRAMAAN_DATABASE_URL``).

Crypto is identical to the in-memory versions: the audit chain is SHA-256 +
HMAC(SOC key); the PII vault is per-identity Fernet with crypto-shredding +
tombstone (KS6 / R3). The audit chain stores only token refs, never plaintext.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import time

from cryptography.fernet import Fernet

from .keystore import Erased

GENESIS = "0" * 64


class PostgresAuditStore:
    def __init__(self, dsn: str, signing_key: bytes):
        self._dsn = dsn
        self._key = signing_key
        self._init()

    def _conn(self):
        import psycopg2

        return psycopg2.connect(self._dsn)

    def _init(self) -> None:
        with self._conn() as c, c.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS audit_log (
                    seq        INTEGER PRIMARY KEY,
                    ts         DOUBLE PRECISION NOT NULL,
                    payload    JSONB NOT NULL,
                    prev_hash  TEXT NOT NULL,
                    hash       TEXT NOT NULL,
                    hmac       TEXT NOT NULL
                )""")

    def _hash(self, body: dict) -> str:
        return hashlib.sha256(
            json.dumps(body, sort_keys=True, default=str).encode()).hexdigest()

    def _mac(self, record_hash: str) -> str:
        return hmac.new(self._key, record_hash.encode(), hashlib.sha256).hexdigest()

    def append(self, payload: dict) -> dict:
        with self._conn() as c, c.cursor() as cur:
            cur.execute("SELECT seq, hash FROM audit_log ORDER BY seq DESC LIMIT 1")
            row = cur.fetchone()
            seq = (row[0] + 1) if row else 0
            prev_hash = row[1] if row else GENESIS
            body = {"seq": seq, "ts": time.time(), "payload": payload,
                    "prev_hash": prev_hash}
            h = self._hash(body)
            m = self._mac(h)
            cur.execute(
                "INSERT INTO audit_log (seq, ts, payload, prev_hash, hash, hmac) "
                "VALUES (%s,%s,%s,%s,%s,%s)",
                (seq, body["ts"], json.dumps(payload), prev_hash, h, m))
            return {**body, "hash": h, "hmac": m}

    def verify_chain(self) -> bool:
        with self._conn() as c, c.cursor() as cur:
            cur.execute("SELECT seq, ts, payload, prev_hash, hash, hmac "
                        "FROM audit_log ORDER BY seq ASC")
            prev = GENESIS
            for idx, (seq, ts, payload, prev_hash, h, m) in enumerate(cur.fetchall()):
                if seq != idx or prev_hash != prev:
                    return False
                body = {"seq": seq, "ts": ts, "payload": payload, "prev_hash": prev_hash}
                if self._hash(body) != h or not hmac.compare_digest(self._mac(h), m):
                    return False
                prev = h
        return True

    def tail(self, n: int = 50) -> list[dict]:
        with self._conn() as c, c.cursor() as cur:
            cur.execute("SELECT seq, ts, payload, prev_hash, hash, hmac FROM "
                        "(SELECT * FROM audit_log ORDER BY seq DESC LIMIT %s) t "
                        "ORDER BY seq ASC", (n,))
            return [{"seq": s, "ts": t, "payload": p, "prev_hash": ph,
                     "hash": h, "hmac": m} for s, t, p, ph, h, m in cur.fetchall()]

    def count(self) -> int:
        with self._conn() as c, c.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM audit_log")
            return int(cur.fetchone()[0])

    def head_checkpoint(self) -> str:
        with self._conn() as c, c.cursor() as cur:
            cur.execute("SELECT hash FROM audit_log ORDER BY seq DESC LIMIT 1")
            row = cur.fetchone()
            cur.execute("SELECT COUNT(*) FROM audit_log")
            n = int(cur.fetchone()[0])
        last = row[0] if row else GENESIS
        body = f"{n}|{last}"
        return body + "|" + hmac.new(self._key, body.encode(), hashlib.sha256).hexdigest()


class PostgresDecisionStore:
    """Async-flow decision sink: the scoring pod writes the client decision here;
    the ingress polls it by event_id (the Kafka→pod→decision→audit path)."""

    def __init__(self, dsn: str):
        self._dsn = dsn
        self._init()

    def _conn(self):
        import psycopg2

        return psycopg2.connect(self._dsn)

    def _init(self) -> None:
        with self._conn() as c, c.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS decisions (
                    event_id   TEXT PRIMARY KEY,
                    identity_id TEXT NOT NULL,
                    decision   TEXT NOT NULL,
                    body       JSONB NOT NULL,
                    ts         DOUBLE PRECISION NOT NULL
                )""")

    def put(self, event_id: str, identity_id: str, decision: str, body: dict) -> None:
        with self._conn() as c, c.cursor() as cur:
            cur.execute(
                "INSERT INTO decisions (event_id, identity_id, decision, body, ts) "
                "VALUES (%s,%s,%s,%s,%s) ON CONFLICT (event_id) DO NOTHING",
                (event_id, identity_id, decision, json.dumps(body), time.time()))

    def get(self, event_id: str) -> dict | None:
        with self._conn() as c, c.cursor() as cur:
            cur.execute("SELECT body FROM decisions WHERE event_id=%s", (event_id,))
            row = cur.fetchone()
            return row[0] if row else None


class PostgresPiiVault:
    """Durable keyed PII vault with crypto-shredding + tombstone (KS6 / R3)."""

    def __init__(self, dsn: str):
        self._dsn = dsn
        self._init()

    def _conn(self):
        import psycopg2

        return psycopg2.connect(self._dsn)

    def _init(self) -> None:
        with self._conn() as c, c.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS pii_vault (
                    identity_id TEXT PRIMARY KEY,
                    ciphertext  BYTEA,
                    fkey        BYTEA,
                    tombstoned  BOOLEAN NOT NULL DEFAULT FALSE
                )""")

    def put(self, identity_id: str, material: dict) -> str:
        with self._conn() as c, c.cursor() as cur:
            cur.execute("SELECT tombstoned, fkey FROM pii_vault WHERE identity_id=%s",
                        (identity_id,))
            row = cur.fetchone()
            if row and row[0]:
                raise Erased(f"{identity_id}: erased (tombstoned) — re-consent required")
            key = bytes(row[1]) if row and row[1] else Fernet.generate_key()
            token = Fernet(key).encrypt(json.dumps(material, sort_keys=True).encode())
            cur.execute(
                "INSERT INTO pii_vault (identity_id, ciphertext, fkey, tombstoned) "
                "VALUES (%s,%s,%s,FALSE) ON CONFLICT (identity_id) DO UPDATE SET "
                "ciphertext=EXCLUDED.ciphertext, fkey=EXCLUDED.fkey",
                (identity_id, token, key))
            return hashlib.sha256(token).hexdigest()[:16]

    def get(self, identity_id: str) -> dict:
        with self._conn() as c, c.cursor() as cur:
            cur.execute("SELECT ciphertext, fkey FROM pii_vault WHERE identity_id=%s",
                        (identity_id,))
            row = cur.fetchone()
        if not row or not row[1]:
            raise Erased(f"{identity_id}: key destroyed — material irrecoverable")
        return json.loads(Fernet(bytes(row[1])).decrypt(bytes(row[0])))

    def erase(self, identity_id: str) -> bool:
        with self._conn() as c, c.cursor() as cur:
            cur.execute("SELECT 1 FROM pii_vault WHERE identity_id=%s AND fkey IS NOT NULL",
                        (identity_id,))
            existed = cur.fetchone() is not None
            cur.execute(
                "INSERT INTO pii_vault (identity_id, ciphertext, fkey, tombstoned) "
                "VALUES (%s, NULL, NULL, TRUE) ON CONFLICT (identity_id) DO UPDATE SET "
                "fkey=NULL, tombstoned=TRUE", (identity_id,))
            return existed

    def is_erased(self, identity_id: str) -> bool:
        with self._conn() as c, c.cursor() as cur:
            cur.execute("SELECT tombstoned FROM pii_vault WHERE identity_id=%s",
                        (identity_id,))
            row = cur.fetchone()
            return bool(row and row[0])
