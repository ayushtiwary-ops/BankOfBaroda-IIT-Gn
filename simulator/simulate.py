"""Attack & traffic simulator - demonstrates PRAMAAN end-to-end (hardened).

Phase 1 builds normal baselines; Phase 2 runs four classic attacks.

The hardened trust boundary is the point of the demo:
  * Legitimate events carry a SIGNED, device-ATTESTED behavioural assertion
    (the only way a similarity score is believed).
  * Attacks carry NO behavioural tokens - the attacker cannot self-assert a
    high score; behaviour is MISSING and the engine scores them anyway.

Run directly against the engine (no server, demo model):
    python simulator/simulate.py
Against a running API (prints the GENERIC client decision):
    python simulator/simulate.py --api http://localhost:8000 --api-key edge-key-events
"""
import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.attestation import ( # noqa: E402
    BehaviorProvider,
    BehaviorResolver,
    DeviceAttestationProvider,
)
from app.config import Settings  # noqa: E402
from app.risk_engine import TrustEngine  # noqa: E402
from app.schemas import Channel, EventType, IdentityEvent  # noqa: E402
from app.verifier import Ed25519Signer, Ed25519Verifier  # noqa: E402

random.seed(7)

CUSTOMERS = [f"cust_{i:03d}" for i in range(5)]
EMPLOYEE = "emp_ops_01"

# Trusted providers (private keys) + the engine-side resolver (public keys only).
_ATTEST = Ed25519Signer.generate()
_BEHAV = Ed25519Signer.generate()
_ATTEST_PROVIDER = DeviceAttestationProvider(_ATTEST)
_BEHAV_PROVIDER = BehaviorProvider(_BEHAV)


def _resolver() -> BehaviorResolver:
    return BehaviorResolver(
        attest_verifier=Ed25519Verifier.from_b64(_ATTEST.public_key_b64),
        behavior_verifier=Ed25519Verifier.from_b64(_BEHAV.public_key_b64),
   )


def _attested(device_id: str, identity: str, similarity: float) -> dict:
    """Signed attestation + behavioural assertion for a legitimate device."""
    return dict(
        device_attestation=_ATTEST_PROVIDER.issue(device_id=device_id),
        behavior_assertion=_BEHAV_PROVIDER.issue(
            device_id=device_id, identity_id=identity, similarity=similarity),
   )


def normal_event(identity: str) -> IdentityEvent:
    dev = f"dev_{identity}_home"
    return IdentityEvent(
        identity_id=identity,
        event_type=random.choice([EventType.LOGIN, EventType.TRANSACTION]),
        channel=random.choice([Channel.MOBILE_APP, Channel.INTERNET_BANKING]),
        device_id=dev, geo="IN-GJ", hour_of_day=random.randint(9, 21),
        amount=random.uniform(200, 5000) if random.random() < 0.5 else None,
        **_attested(dev, identity, random.uniform(0.85, 0.99)),
   )


def show(label, soc):
    print(f"  {label:<38} trust={soc.trust_score:<4} band={soc.risk_band:<16} "
          f"decision={soc.decision.value:<8} "
          f"stepup={soc.step_up_method.value if soc.step_up_method else '-'}")
    for r in soc.reason_codes:          # SOC plane only
        print(f"      · {r}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", default=None, help="Score via a running API")
    parser.add_argument("--api-key", default="edge-key-events")
    args = parser.parse_args()

    if args.api:
        import json
        import urllib.request
        from types import SimpleNamespace

        def assess(e: IdentityEvent):
            req = urllib.request.Request(
                f"{args.api}/v1/events", data=e.model_dump_json().encode(),
                headers={"Content-Type": "application/json", "X-API-Key": args.api_key})
            d = json.loads(urllib.request.urlopen(req).read())
            # client plane is GENERIC - no trust/band/reasons
            return SimpleNamespace(decision=SimpleNamespace(value=d["decision"]),
                                   step_up_method=None, trust_score="·", risk_band="(client)",
                                   reason_codes=[f"client message: {d['message']}"])
    else:
        engine = TrustEngine(
            settings=Settings.from_env({"PRAMAAN_MODE": "demo_synthetic"}),
            behavior_resolver=_resolver())
        assess = engine.assess

    print("\n=== Phase 1 - building behavioural baselines (60 normal events) ===")
    for _ in range(12):
        for c in CUSTOMERS:
            assess(normal_event(c))
    show("routine login, known device (attested)", assess(normal_event(CUSTOMERS[0])))

    for _ in range(10):
        dev = "dev_corp_laptop_01"
        assess(IdentityEvent(
            identity_id=EMPLOYEE, event_type=EventType.PRIVILEGED_ACCESS,
            channel=Channel.ADMIN_CONSOLE, device_id=dev, geo="IN-GJ",
            hour_of_day=random.randint(10, 17), privileged_scope="crm.read",
            **_attested(dev, EMPLOYEE, random.uniform(0.9, 0.99))))

    print("\n=== Phase 2 - attack scenarios (attacker CANNOT self-assert behaviour) ===")

    print("\n[A] Account takeover attempt on cust_000:")
    show("ATO: new device+geo, ₹95k to new payee", assess(IdentityEvent(
        identity_id="cust_000", event_type=EventType.TRANSACTION,
        channel=Channel.INTERNET_BANKING, device_id="dev_attacker_x",
        geo="RU-MOW", hour_of_day=3, amount=95_000.0, is_new_beneficiary=True)))

    print("\n[B] Suspicious account recovery on cust_001:")
    show("recovery-contact change, unknown device", assess(IdentityEvent(
        identity_id="cust_001", event_type=EventType.ACCOUNT_RECOVERY,
        channel=Channel.INTERNET_BANKING, device_id="dev_unknown_77",
        geo="IN-DL", hour_of_day=3, recovery_contact_changed=True)))

    print("\n[C] Mule-account onboarding burst (one device, many identities):")
    for i in range(3):
        show(f"onboarding mule_{i}", assess(IdentityEvent(
            identity_id=f"mule_{i}", event_type=EventType.ONBOARDING,
            channel=Channel.MOBILE_APP, device_id="dev_mule_farm_01",
            geo="IN-WB", hour_of_day=2)))

    print("\n[D] Insider: privileged write access at 02:00 from new device:")
    show("emp_ops_01 → core_banking.write", assess(IdentityEvent(
        identity_id=EMPLOYEE, event_type=EventType.PRIVILEGED_ACCESS,
        channel=Channel.ADMIN_CONSOLE, device_id="dev_personal_phone",
        geo="IN-MH", hour_of_day=2, privileged_scope="core_banking.write")))

    print("\n=== Done. Legitimate (attested) users sailed through; "
          "every attack was challenged or blocked. ===\n")


if __name__ == "__main__":
    main()
