"""Core behaviour tests for the PRAMAAN trust engine.

Updated for the hardened contract:
  * IdentityEvent no longer carries a trusted ``behavior_score``.
  * ``assess`` returns a SocAssessment; detector reasons live on that SOC plane,
    not on the client projection.
  * Per-identity state lives in the StateStore, not engine dicts.
  * Step-up is applied via ``apply_verified_step_up`` (called only after a
    signed assertion is validated upstream).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.audit import AuditLog  # noqa: E402
from app.privacy import pseudonymize  # noqa: E402
from app.schemas import Channel, Decision, EventType  # noqa: E402


def _warm(engine, make_event, identity="u1", n=15):
    for _ in range(n):
        engine.assess(make_event(identity))


def test_normal_behaviour_is_frictionless(demo_engine, make_event):
    _warm(demo_engine, make_event)
    a = demo_engine.assess(make_event())
    assert a.decision == Decision.ALLOW
    assert a.trust_score >= 650


def test_account_takeover_is_challenged(demo_engine, make_event):
    _warm(demo_engine, make_event)
    ato = make_event(
        event_type=EventType.TRANSACTION, channel=Channel.INTERNET_BANKING,
        device_id="dev_attacker", geo="RU-MOW", hour_of_day=3,
        amount=95_000.0, is_new_beneficiary=True,
   )
    a = demo_engine.assess(ato)
    assert a.decision in (Decision.STEP_UP, Decision.BLOCK)
    assert any("NEW_DEVICE" in r or "BEHAVIOUR" in r or "GEO" in r
               for r in a.reason_codes)


def test_suspicious_recovery_requires_strong_verification(demo_engine, make_event):
    _warm(demo_engine, make_event, "u2")
    a = demo_engine.assess(make_event(
        "u2", event_type=EventType.ACCOUNT_RECOVERY, device_id="dev_unknown",
        hour_of_day=3, recovery_contact_changed=True))
    assert a.decision in (Decision.STEP_UP, Decision.BLOCK)


def test_verified_step_up_restores_then_failure_craters(demo_engine):
    ident = "u3"
    with demo_engine.store.lock(ident):
        s = demo_engine.store.load(ident)
        s.trust = 300
        assert demo_engine.store.commit(ident, s, s.version)
    restored = demo_engine.apply_verified_step_up(ident, True)
    assert restored > 300
    failed = demo_engine.apply_verified_step_up(ident, False)
    assert failed < restored


def test_failed_events_do_not_poison_baseline(demo_engine, make_event):
    _warm(demo_engine, make_event, "u4")
    before = len(demo_engine.store.load("u4").devices)
    demo_engine.assess(make_event(
        "u4", device_id="dev_evil", geo="RU-MOW", hour_of_day=3,
        event_type=EventType.ACCOUNT_RECOVERY, recovery_contact_changed=True))
    # blocked/challenged event must NOT add the attacker device to the profile
    assert len(demo_engine.store.load("u4").devices) == before


def test_audit_chain_tamper_evident():
    log = AuditLog()
    for i in range(5):
        log.append({"i": i})
    assert log.verify_chain()
    log.records[2]["payload"]["i"] = 999       # tamper
    assert not log.verify_chain()


def test_pseudonymization_is_deterministic_and_opaque():
    t1, t2 = pseudonymize("CUST-9876", secret=b"k"), pseudonymize("CUST-9876", secret=b"k")
    assert t1 == t2
    assert "9876" not in t1
