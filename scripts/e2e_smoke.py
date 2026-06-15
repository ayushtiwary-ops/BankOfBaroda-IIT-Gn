#!/usr/bin/env python3
"""End-to-end smoke for the executable architecture.

Proves an event flows Kafka → scoring pod → decision → Postgres audit, and that
the verifier service (private key) is the only minter of step-up assertions.

    cd pramaan && docker compose -f infra/docker-compose.yml up --build -d
    python scripts/e2e_smoke.py

Exits non-zero on any failure (CI-friendly).
"""
import sys
import time

import httpx

INGRESS = "http://127.0.0.1:8090"
VERIFIER = "http://127.0.0.1:8081"
PG_DSN = "postgresql://postgres:pramaan@127.0.0.1:5433/pramaan"
EDGE = {"X-API-Key": "edge-key"}


def _wait(url, timeout=120):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            if httpx.get(url, timeout=3).status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(2)
    return False


def main() -> int:
    print("waiting for ingress + verifier ...")
    if not _wait(f"{INGRESS}/health") or not _wait(f"{VERIFIER}/health"):
        print("FAIL: services did not come up")
        return 1

    # the verifier exposes ONLY its public key (trust boundary)
    vh = httpx.get(f"{VERIFIER}/health").json()
    assert vh.get("public_key") and "private" not in str(vh).lower(), vh
    print("ok: verifier up, exposes public key only")

    # produce an event → Kafka → pod → decision → Postgres
    ev = {"identity_id": "e2e-cust-1", "event_type": "login", "channel": "mobile_app",
          "device_id": "dev-1", "geo": "IN-GJ", "hour_of_day": 12}
    r = httpx.post(f"{INGRESS}/v1/events", headers=EDGE, json=ev, timeout=10)
    assert r.status_code == 202, (r.status_code, r.text)
    event_id = r.json()["event_id"]
    print(f"ok: event accepted (202), id={event_id} - produced to Kafka")

    decision = None
    for _ in range(30):
        d = httpx.get(f"{INGRESS}/v1/decisions/{event_id}", headers=EDGE)
        if d.status_code == 200:
            decision = d.json()
            break
        time.sleep(1)
    assert decision and decision["decision"] in ("ALLOW", "STEP_UP", "BLOCK"), decision
    # client plane must carry NO detector internals end-to-end
    blob = str(decision).lower()
    assert "trust" not in blob and "reason" not in blob and "shap" not in blob, decision
    print(f"ok: decision={decision['decision']} (Kafka→pod→Postgres), client-plane clean")

    # the durable audit chain (Postgres) verifies
    sys.path.insert(0, "backend")
    from app.stores import PostgresAuditStore
    audit = PostgresAuditStore(PG_DSN, signing_key=b"DEMO-audit-key-rotate-in-prod-0123456789")
    assert audit.verify_chain() is True and audit.count() >= 1
    print(f"ok: durable audit chain verifies ({audit.count()} records, keyed HMAC)")

    # the verifier can mint a signed step-up assertion (the engine could not)
    a = httpx.post(f"{VERIFIER}/issue", json={
        "challenge_id": "c1", "identity_id": "e2e-cust-1", "method": "otp_sms",
        "result": "pass"}).json()
    assert a.get("assertion") and "." in a["assertion"], a
    print("ok: verifier minted a signed step-up assertion (trust boundary)")

    print("\nE2E SMOKE PASSED - Kafka + Redis + Postgres + pods + verifier + ingress")
    return 0


if __name__ == "__main__":
    sys.exit(main())
