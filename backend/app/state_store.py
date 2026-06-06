"""Externalized per-identity state — the engine becomes stateless.

SECURITY: KS7 — per-identity trust + behavioural profile + the
cross-identity device graph live behind a ``StateStore`` interface instead of
process-global ``dict``s. Two implementations:

  * ``InMemoryStateStore`` — for tests / single-process demo. Per-key locks +
    an optimistic-concurrency version so it has the SAME contract as Redis.
  * ``RedisStateStore``   — for prod. State is shared across horizontally
    scaled scoring pods; writes use Redis ``WATCH``/``MULTI`` optimistic
    concurrency so concurrent same-identity requests cannot lose updates.

Because state is keyed by ``identity_id`` and never held in the process, the
"stateless pods" claim in the README/diagram becomes literally true.
"""
from __future__ import annotations

import copy
import json
import threading
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field

DEFAULT_TRUST = 700  # base trust for an unseen identity


@dataclass
class IdentityState:
    """All mutable per-identity state, in one place."""

    trust: int = DEFAULT_TRUST
    devices: set[str] = field(default_factory=set)
    geos: set[str] = field(default_factory=set)
    hours: list[int] = field(default_factory=list)
    amounts: list[float] = field(default_factory=list)
    event_count: int = 0
    burst: float = 0.0
    last_event_ts: float = 0.0
    # rolling recovery budget (x-factor: capped/rate-limited trust recovery)
    recovered_in_window: int = 0
    recovery_window_start: float = 0.0
    # KS9: rolling event-risk window for drift detection (low-and-slow)
    risk_window: list[float] = field(default_factory=list)
    # KS9 (R3): persistent baseline + CUSUM — catches arbitrarily-slow drift
    risk_ewma: float = 0.0
    cusum: float = 0.0
    # KS9: sticky secondary-review flag — set on drift, cleared by a verified
    # step-up, so a flagged identity stays under review instead of reverting.
    under_review: bool = False
    # KS10 (R3): cold-start prior is one-shot — consumed on first assessment so
    # it cannot be re-probed across non-committed retries.
    cold_prior_used: bool = False
    # KS10: impossible-travel / geo-velocity needs the previous geo + wall time
    last_geo: str = ""
    last_geo_ts: float = 0.0
    version: int = 0

    # Schema-explicit (de)serialization. We deliberately AVOID pickle here:
    # state is round-tripped through Redis and a pickle.loads on attacker-
    # writable Redis would be an RCE (CWE-502) — exactly the class of bug this
    # project exists to close. JSON only reconstructs declared primitive types.
    def to_json(self) -> bytes:
        d = asdict(self)
        d["devices"] = sorted(self.devices)
        d["geos"] = sorted(self.geos)
        return json.dumps(d).encode()

    @classmethod
    def from_json(cls, raw: bytes) -> "IdentityState":
        d = json.loads(raw)
        d["devices"] = set(d.get("devices", []))
        d["geos"] = set(d.get("geos", []))
        return cls(**d)


class StateStore(ABC):
    @abstractmethod
    def load(self, identity_id: str) -> IdentityState: ...

    @abstractmethod
    def commit(self, identity_id: str, state: IdentityState,
               expected_version: int) -> bool:
        """Compare-and-set. Returns False (no write) if the stored version no
        longer matches ``expected_version`` — caller should reload + retry."""

    @abstractmethod
    @contextmanager
    def lock(self, identity_id: str): ...

    @abstractmethod
    def device_add(self, device_id: str, identity_id: str) -> int:
        """Record identity on device; return distinct-identity count (mule signal)."""

    @abstractmethod
    def device_count(self, device_id: str) -> int: ...


class InMemoryStateStore(StateStore):
    def __init__(self):
        self._data: dict[str, IdentityState] = {}
        self._devices: dict[str, set[str]] = {}
        self._global = threading.RLock()
        self._key_locks: dict[str, threading.RLock] = {}

    def _key_lock(self, identity_id: str) -> threading.RLock:
        with self._global:
            lk = self._key_locks.get(identity_id)
            if lk is None:
                lk = self._key_locks[identity_id] = threading.RLock()
            return lk

    @contextmanager
    def lock(self, identity_id: str):
        lk = self._key_lock(identity_id)
        lk.acquire()
        try:
            yield
        finally:
            lk.release()

    def load(self, identity_id: str) -> IdentityState:
        with self._global:
            stored = self._data.get(identity_id)
            return copy.deepcopy(stored) if stored is not None else IdentityState()

    def commit(self, identity_id: str, state: IdentityState,
               expected_version: int) -> bool:
        with self._global:
            current = self._data.get(identity_id)
            current_version = current.version if current is not None else 0
            if current_version != expected_version:
                return False  # someone else wrote first — caller retries
            new = copy.deepcopy(state)
            new.version = expected_version + 1
            self._data[identity_id] = new
            return True

    def device_add(self, device_id: str, identity_id: str) -> int:
        with self._global:
            ids = self._devices.setdefault(device_id, set())
            ids.add(identity_id)
            return len(ids)

    def device_count(self, device_id: str) -> int:
        with self._global:
            return len(self._devices.get(device_id, ()))


class RedisStateStore(StateStore):  # pragma: no cover - needs a live Redis
    """Prod store. State shared across pods; CAS via WATCH/MULTI.

    Constructed only when ``PRAMAAN_REDIS_URL`` is set; kept dependency-light so
    test runs never need a Redis server.
    """

    def __init__(self, url: str, namespace: str = "pramaan"):
        import redis  # local import: only prod pulls the client in

        self._r = redis.Redis.from_url(url, decode_responses=False)
        self._ns = namespace

    def _key(self, identity_id: str) -> str:
        return f"{self._ns}:id:{identity_id}"

    def _dev_key(self, device_id: str) -> str:
        return f"{self._ns}:dev:{device_id}"

    @contextmanager
    def lock(self, identity_id: str):
        # Redlock-style single-key lock; CAS in commit() is the real guard, the
        # lock just reduces wasted retries under contention.
        with self._r.lock(f"{self._ns}:lock:{identity_id}", timeout=5, blocking_timeout=5):
            yield

    def load(self, identity_id: str) -> IdentityState:
        raw = self._r.get(self._key(identity_id))
        return IdentityState.from_json(raw) if raw else IdentityState()

    def commit(self, identity_id: str, state: IdentityState,
               expected_version: int) -> bool:
        key = self._key(identity_id)
        with self._r.pipeline() as pipe:
            try:
                pipe.watch(key)
                raw = pipe.get(key)
                current_version = IdentityState.from_json(raw).version if raw else 0
                if current_version != expected_version:
                    pipe.unwatch()
                    return False
                state.version = expected_version + 1
                pipe.multi()
                pipe.set(key, state.to_json())
                pipe.execute()
                return True
            except Exception:
                return False

    def device_add(self, device_id: str, identity_id: str) -> int:
        self._r.sadd(self._dev_key(device_id), identity_id)
        return int(self._r.scard(self._dev_key(device_id)))

    def device_count(self, device_id: str) -> int:
        return int(self._r.scard(self._dev_key(device_id)))


def build_state_store(redis_url: str | None) -> StateStore:
    """Pick the prod store when a Redis URL is configured, else in-memory."""
    if redis_url:
        return RedisStateStore(redis_url)
    return InMemoryStateStore()
