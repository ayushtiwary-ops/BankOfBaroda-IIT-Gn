"""Stateless scoring pod — consumes Kafka events, scores, persists.

Flow:  Kafka(pramaan.events) → assess (Redis state, KS7) → decision → Postgres
       (audit chain KS6/KS8 + decisions sink). N replicas form a consumer group
       partitioned by identity_id, so the pods are horizontally scalable and
       hold NO per-identity state (it lives in Redis).
"""
from __future__ import annotations

import os
import time

from .bus import make_consumer
from .config import Settings
from .risk_engine import TrustEngine
from .schemas import IdentityEvent
from .stores import PostgresAuditStore, PostgresDecisionStore


def main() -> None:
    settings = Settings.from_env()              # prod mode → real model, Redis state
    engine = TrustEngine(settings=settings)
    dsn = os.environ["PRAMAAN_DATABASE_URL"]
    audit = PostgresAuditStore(dsn, settings.audit_signing_key)
    decisions = PostgresDecisionStore(dsn)
    consumer = make_consumer(os.environ["PRAMAAN_KAFKA_BROKERS"], group="scoring-pods")
    pod = os.environ.get("HOSTNAME", "pod")
    print(f"[scoring_pod {pod}] ready (model={engine.model_mode})", flush=True)

    while True:
        for msg in consumer:                    # consumer_timeout_ms makes this yield
            rec = msg.value
            try:
                event = IdentityEvent(**rec["event"])
                soc = engine.assess(event)
                payload = soc.to_audit_payload()
                payload["pod"] = pod
                audit.append(payload)            # full SOC plane → durable audit chain
                decisions.put(rec["event_id"], event.identity_id,
                              soc.decision.value, soc.to_client().model_dump())
            except Exception as exc:             # never let one bad event kill the pod
                print(f"[scoring_pod {pod}] error: {exc}", flush=True)
        time.sleep(0.1)


if __name__ == "__main__":
    main()
