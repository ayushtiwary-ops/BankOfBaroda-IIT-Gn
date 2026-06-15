"""Security regression tests for PRAMAAN - one red→green test per hardening item.

Each test (or block) is tagged with the hardening item it proves closed. Every test
here is written to FAIL against the original scaffold and PASS against the
hardened service (red→green discipline, per the hardened spec).

Layout:
"""
import sys
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parents[1] / "backend"
sys.path.insert(0, str(BACKEND))


# ====================================================================== # No default secrets; prod refuses to start when a required secret is unset.
def test_prod_mode_fails_loud_without_required_secret():
    # SECURITY: removing the "demo-edge-secret" default means a prod
    # process with no configured secret must refuse to start, not run insecure.
    from app.config import ConfigError, Settings

    with pytest.raises(ConfigError) as ei:
        Settings.from_env({"PRAMAAN_MODE": "prod"})  # nothing else set
    assert "PRAMAAN_EDGE_SECRET" in str(ei.value)


def test_prod_mode_starts_with_all_secrets(prod_env):
    from app.config import Settings

    s = Settings.from_env(prod_env)
    assert s.mode == "prod"
    assert s.edge_secret  # present, non-empty
    assert s.api_keys  # at least one caller registered


def test_demo_mode_synthesizes_but_is_flagged():
    # Demo mode is allowed, but it must be explicit and self-identifying so a
    # synthetic run can never be mistaken for a trustworthy one.
    from app.config import Settings

    s = Settings.from_env({"PRAMAAN_MODE": "demo_synthetic"})
    assert s.mode == "demo_synthetic"
    assert s.is_synthetic is True


# ======================================================================== # Step-up outcome must be a SIGNED assertion from a trusted verifier; the client
# can never assert its own success.
def test_valid_signed_assertion_is_accepted(stepup_provider, stepup_validator):
    # SECURITY: only a verifier-signed assertion can move trust.
    token = stepup_provider.issue(
        challenge_id="ch-1", identity_id="u1", method="otp_sms", result="pass"
   )
    assertion = stepup_validator.validate(token, expected_identity_id="u1")
    assert assertion.result == "pass"
    assert assertion.identity_id == "u1"


def test_self_asserted_token_is_rejected(stepup_validator):
    # The attacker's "I verified myself" - an unsigned/forged blob - is rejected.
    from app.verifier import AssertionRejected

    forged = "eyJyZXN1bHQiOiJwYXNzIn0.not-a-real-signature"
    with pytest.raises(AssertionRejected):
        stepup_validator.validate(forged, expected_identity_id="u1")


def test_tampered_result_breaks_signature(stepup_provider, stepup_validator):
    import base64
    import json

    from app.verifier import AssertionRejected

    token = stepup_provider.issue(
        challenge_id="ch-2", identity_id="u1", method="otp_sms", result="fail"
   )
    payload_b64, sig_b64 = token.split(".")
    payload = json.loads(base64.urlsafe_b64decode(payload_b64 + "=="))
    payload["result"] = "pass"  # flip fail → pass without re-signing
    tampered_b64 = base64.urlsafe_b64encode(
        json.dumps(payload).encode()
   ).rstrip(b"=").decode()
    with pytest.raises(AssertionRejected):
        stepup_validator.validate(f"{tampered_b64}.{sig_b64}", expected_identity_id="u1")


def test_identity_binding_enforced(stepup_provider, stepup_validator):
    # A valid assertion for victim u1 cannot be replayed against attacker u2.
    from app.verifier import AssertionRejected

    token = stepup_provider.issue(
        challenge_id="ch-3", identity_id="u1", method="otp_sms", result="pass"
   )
    with pytest.raises(AssertionRejected):
        stepup_validator.validate(token, expected_identity_id="u2")


def test_nonce_is_single_use_anti_replay(stepup_provider, stepup_validator):
    token = stepup_provider.issue(
        challenge_id="ch-4", identity_id="u1", method="otp_sms", result="pass"
   )
    stepup_validator.validate(token, expected_identity_id="u1")  # first use ok
    from app.verifier import AssertionRejected

    with pytest.raises(AssertionRejected):  # replay rejected
        stepup_validator.validate(token, expected_identity_id="u1")


def test_expired_assertion_is_rejected(stepup_provider, stepup_validator):
    from app.verifier import AssertionRejected

    token = stepup_provider.issue(
        challenge_id="ch-5", identity_id="u1", method="otp_sms",
        result="pass", ttl_seconds=-1,  # already expired
   )
    with pytest.raises(AssertionRejected):
        stepup_validator.validate(token, expected_identity_id="u1")


# ======================================================================== # Per-identity state externalized behind a StateStore; concurrent same-identity
# requests cannot race the trust value.
def test_state_is_isolated_per_identity():
    from app.state_store import DEFAULT_TRUST, InMemoryStateStore

    store = InMemoryStateStore()
    a = store.load("a")
    a.trust = 123
    assert store.commit("a", a, a.version) is True
    assert store.load("b").trust == DEFAULT_TRUST  # untouched identity at baseline
    assert store.load("a").trust == 123


def test_optimistic_concurrency_detects_stale_write():
    # SECURITY: two readers of the same version cannot both win.
    from app.state_store import InMemoryStateStore

    store = InMemoryStateStore()
    first = store.load("u")
    second = store.load("u")  # same version as `first`
    first.trust = 10
    assert store.commit("u", first, first.version) is True
    second.trust = 20
    assert store.commit("u", second, second.version) is False  # stale → rejected


def test_no_lost_updates_under_concurrency():
    # SECURITY: N threads each commit one event to the SAME identity;
    # with per-key locking the final event_count must be exactly N (no lost
    # updates). The old global-dict-without-locks design loses writes here.
    import threading

    from app.state_store import InMemoryStateStore

    store = InMemoryStateStore()
    n = 64

    def worker():
        with store.lock("hot"):
            s = store.load("hot")
            s.event_count += 1
            assert store.commit("hot", s, s.version) is True

    threads = [threading.Thread(target=worker) for _ in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert store.load("hot").event_count == n


def test_device_graph_counts_distinct_identities():
    from app.state_store import InMemoryStateStore

    store = InMemoryStateStore()
    assert store.device_add("dev", "i1") == 1
    assert store.device_add("dev", "i2") == 2
    assert store.device_add("dev", "i1") == 2  # re-add same identity = no growth


# ======================================================================== # The live engine must load a REAL exported artifact; prod refuses to start
# without it; demo_synthetic is allowed but stamped.
def test_prod_refuses_to_start_without_artifact(prod_env, tmp_path):
    # SECURITY: prod must FAIL LOUD with no real model (no silent
    # np.random fallback).
    from app.config import Settings
    from app.features import FEATURE_NAMES
    from app.model_loader import ModelArtifactMissing, load_serving_model

    env = dict(prod_env)
    env["PRAMAAN_MODEL_DIR"] = str(tmp_path / "absent")
    with pytest.raises(ModelArtifactMissing):
        load_serving_model(Settings.from_env(env), FEATURE_NAMES)


def test_loads_real_artifact_with_model_card(prod_env, serving_artifact):
    from app.config import Settings
    from app.features import FEATURE_NAMES
    from app.model_loader import load_serving_model

    env = dict(prod_env)
    env["PRAMAAN_MODEL_DIR"] = str(serving_artifact)
    m = load_serving_model(Settings.from_env(env), FEATURE_NAMES)
    assert m.provenance != "DEMO_SYNTHETIC"
    assert m.card["dataset"]  # model card names the dataset it trained on
    assert 0.0 <= m.risk([0.0] * len(FEATURE_NAMES)) <= 1.0


def test_demo_mode_is_stamped_synthetic():
    from app.config import Settings
    from app.features import FEATURE_NAMES
    from app.model_loader import load_serving_model

    m = load_serving_model(Settings.from_env({"PRAMAAN_MODE": "demo_synthetic"}),
                           FEATURE_NAMES)
    assert m.is_synthetic is True
    assert m.provenance == "DEMO_SYNTHETIC"


def test_tampered_artifact_is_rejected(prod_env, serving_artifact):
    # Flip a byte in the joblib → SHA-256 in the card no longer matches → refuse.
    from app.config import Settings
    from app.features import FEATURE_NAMES
    from app.model_loader import ModelArtifactInvalid, load_serving_model

    joblib_path = serving_artifact / "serving_anomaly.joblib"
    data = bytearray(joblib_path.read_bytes())
    data[len(data) // 2] ^= 0xFF
    joblib_path.write_bytes(bytes(data))
    env = dict(prod_env)
    env["PRAMAAN_MODEL_DIR"] = str(serving_artifact)
    with pytest.raises(ModelArtifactInvalid):
        load_serving_model(Settings.from_env(env), FEATURE_NAMES)


# ======================================================================== # Behavioural signal must be attested + signed; unsigned/forged input is MISSING
# (cold-start), never trusted as 0.99.
def test_attested_signed_behavior_is_trusted(
        attest_provider, behavior_provider, behavior_resolver):
    # SECURITY: a similarity score is trusted ONLY when it rides a
    # signed assertion bound to an attested device.
    att = attest_provider.issue(device_id="devA")
    beh = behavior_provider.issue(device_id="devA", identity_id="u1", similarity=0.93)
    sim = behavior_resolver.resolve(
        attestation_token=att, behavior_token=beh,
        expected_identity_id="u1", expected_device_id="devA")
    assert sim == pytest.approx(0.93)


def test_unsigned_high_score_is_treated_missing(behavior_resolver):
    # The attacker's bare "behavior_score=0.99" has no signed assertion at all.
    sim = behavior_resolver.resolve(
        attestation_token=None, behavior_token=None,
        expected_identity_id="u1", expected_device_id="devA")
    assert sim is None  # MISSING → cold-start, NOT 0.99


def test_forged_behavior_token_is_not_trusted(behavior_provider, behavior_resolver):
    # A behavioral assertion with NO valid device attestation is rejected.
    beh = behavior_provider.issue(device_id="devA", identity_id="u1", similarity=0.99)
    sim = behavior_resolver.resolve(
        attestation_token="forged.attestation", behavior_token=beh,
        expected_identity_id="u1", expected_device_id="devA")
    assert sim is None


def test_behavior_must_be_bound_to_attested_device(
        attest_provider, behavior_provider, behavior_resolver):
    # Attestation is for devA but the behavior assertion claims devB → reject.
    att = attest_provider.issue(device_id="devA")
    beh = behavior_provider.issue(device_id="devB", identity_id="u1", similarity=0.95)
    sim = behavior_resolver.resolve(
        attestation_token=att, behavior_token=beh,
        expected_identity_id="u1", expected_device_id="devA")
    assert sim is None


def test_failed_integrity_verdict_is_not_trusted(
        attest_provider, behavior_provider, behavior_resolver):
    att = attest_provider.issue(device_id="devA", verdict="FAILS_DEVICE_INTEGRITY")
    beh = behavior_provider.issue(device_id="devA", identity_id="u1", similarity=0.95)
    sim = behavior_resolver.resolve(
        attestation_token=att, behavior_token=beh,
        expected_identity_id="u1", expected_device_id="devA")
    assert sim is None


def test_serverside_recompute_path_exists():
    # Explicit trust boundary: the server can recompute similarity from raw
    # (privacy-reduced) telemetry instead of trusting any client assertion.
    from app.attestation import recompute_behavior_from_telemetry

    template = [100.0, 120.0, 90.0, 110.0]
    near = recompute_behavior_from_telemetry(template, [101.0, 119.0, 91.0, 109.0])
    far = recompute_behavior_from_telemetry(template, [300.0, 20.0, 400.0, 5.0])
    assert 0.0 <= far < near <= 1.0  # closer telemetry → higher similarity


# ===================================================================== X-AUDIT
# Keyed audit chain resists a recompute attack a plain SHA chain would miss.
def test_keyed_audit_chain_resists_recompute_attack():
    # HARDENING: an attacker edits a record and recomputes EVERY subsequent
    # SHA hash (a plain chain would now verify). The HMAC keyed to the SOC key
    # still catches it.
    import hashlib
    import json

    from app.audit import AuditLog

    log = AuditLog(signing_key=b"soc-signing-key")
    for i in range(5):
        log.append({"i": i})
    assert log.verify_chain()

    recs = log.records
    recs[2]["payload"]["i"] = 999  # tamper
    prev = recs[1]["hash"]
    for r in recs[2:]:  # recompute the SHA chain forward (attacker without key)
        r["prev_hash"] = prev
        body = {k: v for k, v in r.items() if k not in ("hash", "hmac")}
        r["hash"] = hashlib.sha256(
            json.dumps(body, sort_keys=True, default=str).encode()).hexdigest()
        prev = r["hash"]
    assert log.verify_chain() is False  # HMAC mismatch still trips


def test_plain_audit_chain_still_supported_without_key():
    from app.audit import AuditLog

    log = AuditLog()
    for i in range(3):
        log.append({"i": i})
    assert log.verify_chain()


# ===================================================================== # privacy.pseudonymize has no shipped default secret; prod without one fails.
def test_pseudonymize_requires_a_secret_in_prod(monkeypatch):
    from app import privacy
    from app.config import ConfigError

    monkeypatch.setenv("PRAMAAN_MODE", "prod")
    monkeypatch.delenv("PRAMAAN_EDGE_SECRET", raising=False)
    with pytest.raises(ConfigError):
        privacy.pseudonymize("CUST-1")


def test_no_demo_edge_secret_string_in_source():
    # The literal default secret must be gone from the codebase entirely.
    from pathlib import Path

    privacy_src = (Path(__file__).resolve().parents[1] / "backend" / "app"
                   / "privacy.py").read_text()
    assert "demo-edge-secret" not in privacy_src


# ================================================================ (schema)
def test_event_rejects_a_trusted_behavior_score():
    # SECURITY: the old trusted float is gone; smuggling it is a 422.
    import pydantic
    from app.schemas import Channel, EventType, IdentityEvent

    with pytest.raises(pydantic.ValidationError):
        IdentityEvent(identity_id="u", event_type=EventType.LOGIN,
                      channel=Channel.MOBILE_APP, device_id="d", geo="IN-GJ",
                      hour_of_day=12, behavior_score=0.99)


def test_cold_start_when_behavior_missing(demo_engine, make_event):
    # No attestation/assertion → MISSING → engine still scores (cold-start), it
    # does NOT crash and does NOT assume a trusted 0.99.
    soc = demo_engine.assess(make_event("newbie"))
    assert soc.decision is not None  # produced a decision from cold-start path


# ======================================================================== # Detector reason codes + trust score live only on the SOC plane.
DETECTOR_LEAKS = [
    "NEW_DEVICE", "BEHAVIOUR", "GEO_ANOMALY", "TIME_ANOMALY", "AMOUNT_ANOMALY",
    "VELOCITY", "DEVICE_SHARING", "SENSITIVE_ACTION", "CHANNEL_RISK",
    "NEW_BENEFICIARY", "RECOVERY_CHANGE", "NORMAL",
]


def test_reasons_only_on_soc_never_on_client(demo_engine, make_event):
    # SECURITY: client gets a generic decision; SOC gets the reasons.
    from app.schemas import Channel, EventType

    soc = demo_engine.assess(make_event(
        "atk", event_type=EventType.TRANSACTION, channel=Channel.INTERNET_BANKING,
        device_id="dev_x", geo="RU-MOW", hour_of_day=3, amount=95_000.0,
        is_new_beneficiary=True))
    assert soc.reason_codes  # SOC plane carries detector reasons

    client_blob = soc.to_client().model_dump_json()
    for frag in DETECTOR_LEAKS:
        assert frag not in client_blob
    assert "trust" not in client_blob.lower()  # no trust-score oracle either
    assert "risk_band" not in client_blob


def test_audit_payload_carries_full_reasons(demo_engine, make_event):
    from app.schemas import EventType

    soc = demo_engine.assess(make_event(
        "atk2", event_type=EventType.ACCOUNT_RECOVERY, device_id="dev_y",
        geo="RU-MOW", hour_of_day=2, recovery_contact_changed=True))
    payload = soc.to_audit_payload()
    assert payload["reasons"]  # SOC/audit plane has them
    assert "trust_score" in payload


# ============================================================= X-CAPPED-RECOVERY
def test_passive_trust_recovery_is_rate_limited(demo_engine, make_event):
    # HARDENING: a cratered identity cannot be slowly laundered back to full
    # trust by a stream of benign-looking events within one window.
    from app.risk_engine import RECOVERY_CAP_PER_WINDOW

    ident = "slow"
    for _ in range(8):  # warm: device + geo become known → low-risk events
        demo_engine.assess(make_event(ident, device_id="dd", geo="IN-GJ"))
    with demo_engine.store.lock(ident):
        s = demo_engine.store.load(ident)
        s.trust, s.recovered_in_window, s.recovery_window_start = 100, 0, 0.0
        assert demo_engine.store.commit(ident, s, s.version)
    for _ in range(40):
        demo_engine.assess(make_event(ident, device_id="dd", geo="IN-GJ"))
    final = demo_engine.get_trust(ident)
    assert 100 < final <= 100 + RECOVERY_CAP_PER_WINDOW  # recovered, but capped


# ================================================================= X-RESILIENCE
def test_degraded_engine_fails_closed_on_privileged(broken_model_engine, make_event):
    # HARDENING: model down → never silently ALLOW a privileged action.
    from app.schemas import Channel, Decision, EventType

    soc = broken_model_engine.assess(make_event(
        "emp", event_type=EventType.PRIVILEGED_ACCESS,
        channel=Channel.ADMIN_CONSOLE, privileged_scope="core_banking.write"))
    assert soc.degraded is True
    assert soc.decision in (Decision.STEP_UP, Decision.BLOCK)


def test_degraded_engine_fails_open_on_routine_login(broken_model_engine, make_event):
    from app.schemas import Decision

    for _ in range(6):  # warm a known device/geo
        broken_model_engine.assess(make_event("cust", device_id="dk", geo="IN-GJ"))
    soc = broken_model_engine.assess(make_event("cust", device_id="dk", geo="IN-GJ"))
    assert soc.degraded is True
    assert soc.decision == Decision.ALLOW  # availability preserved for low risk


# ============================================================  (HTTP)
from conftest import event_dict  # noqa: E402


def test_endpoint_self_asserted_verified_true_is_impossible(api):
    # DoD: POST /v1/stepup/<id>?verified=true must be IMPOSSIBLE.
    client, _ = api
    r = client.post("/v1/stepup/victim?verified=true",
                    headers={"X-API-Key": "stepup-key"})
    assert r.status_code in (401, 422)  # no signed assertion body → rejected


def test_endpoint_accepts_only_a_valid_signed_assertion(api, stepup_provider):
    client, _ = api
    token = stepup_provider.issue(challenge_id="c1", identity_id="victim",
                                  method="otp_sms", result="pass")
    r = client.post("/v1/stepup/victim", headers={"X-API-Key": "stepup-key"},
                    json={"assertion": token})
    assert r.status_code == 200
    assert r.json()["decision"] == "ALLOW"


def test_endpoint_rejects_forged_assertion(api):
    client, _ = api
    r = client.post("/v1/stepup/victim", headers={"X-API-Key": "stepup-key"},
                    json={"assertion": "forged.token"})
    assert r.status_code == 401


def test_unauthenticated_call_is_rejected(api):
    client, _ = api
    r = client.post("/v1/events", json=event_dict())
    assert r.status_code == 401


def test_wrong_scope_is_forbidden(api):
    client, _ = api
    # edge-key-events only has events:write, not audit:read
    r = client.get("/v1/audit", headers={"X-API-Key": "edge-key-events"})
    assert r.status_code == 403


def test_idor_identity_lookup_requires_soc_scope(api):
    client, _ = api
    assert client.get("/v1/identity/victim").status_code == 401  # anonymous
    assert client.get("/v1/identity/victim",
                      headers={"X-API-Key": "edge-key-events"}).status_code == 403
    ok = client.get("/v1/identity/victim", headers={"X-API-Key": "soc-key-readonly"})
    assert ok.status_code == 200
    assert "trust_score" in ok.json()  # SOC-scoped minimal snapshot


def test_cors_is_an_explicit_allowlist(api):
    _, main = api
    assert "*" not in main.settings.cors_origins
    assert main.settings.cors_origins  # non-empty allowlist


def test_client_event_response_carries_no_detector_internals(api):
    client, _ = api
    ev = event_dict("atk", event_type="transaction", channel="internet_banking",
                    device_id="dev_x", geo="RU-MOW", hour_of_day=3,
                    amount=95_000.0, is_new_beneficiary=True)
    r = client.post("/v1/events", headers={"X-API-Key": "edge-key-events"}, json=ev)
    assert r.status_code == 200
    body = r.text
    for frag in DETECTOR_LEAKS:
        assert frag not in body
    assert "trust" not in body.lower()


def test_idempotency_dedupes_replayed_event(api):
    client, _ = api
    ev = event_dict("dup", idempotency_key="key-abc")
    r1 = client.post("/v1/events", headers={"X-API-Key": "edge-key-events"}, json=ev)
    r2 = client.post("/v1/events", headers={"X-API-Key": "edge-key-events"}, json=ev)
    assert r1.status_code == r2.status_code == 200
    assert r1.json()["event_id"] == r2.json()["event_id"]  # not re-scored


def test_stepup_bombing_is_rate_limited(api, stepup_provider):
    client, _ = api
    last = None
    for i in range(6):
        token = stepup_provider.issue(challenge_id=f"c{i}", identity_id="bomb",
                                      method="otp_sms", result="pass")
        last = client.post("/v1/stepup/bomb", headers={"X-API-Key": "stepup-key"},
                           json={"assertion": token})
    assert last.status_code == 429  # 6th attempt in the window is blocked


def test_audit_endpoint_chain_is_keyed_and_intact(api, stepup_provider):
    client, _ = api
    client.post("/v1/events", headers={"X-API-Key": "edge-key-events"},
                json=event_dict("auditcust"))
    r = client.get("/v1/audit/verify", headers={"X-API-Key": "soc-key-readonly"})
    assert r.status_code == 200
    assert r.json()["chain_intact"] is True


# ============================================================================ #
# ROUND 2 - fixes for the adversarial-review findings (red→green).
# Each was a real bypass the first-pass tests missed.
# ============================================================================ #

# CRITICAL - idempotency cache poisoning. A benign key must NOT replay an
# ALLOW for a later DIFFERENT (malicious) event under the same key.
def test_r2_idempotency_key_reuse_with_different_payload_is_rejected(api):
    client, _ = api
    benign = event_dict("victim", idempotency_key="K")
    r1 = client.post("/v1/events", headers={"X-API-Key": "edge-key-events"}, json=benign)
    assert r1.status_code == 200
    malicious = event_dict("victim", idempotency_key="K", event_type="privileged_access",
                           channel="admin_console", device_id="dev_evil", geo="RU-MOW",
                           hour_of_day=3, privileged_scope="core_banking.write",
                           is_new_beneficiary=True)
    r2 = client.post("/v1/events", headers={"X-API-Key": "edge-key-events"}, json=malicious)
    assert r2.status_code == 409  # conflict - NOT a replayed ALLOW


def test_r2_idempotency_is_namespaced_per_caller(api):
    client, _ = api
    ev = event_dict("v", idempotency_key="shared")
    r1 = client.post("/v1/events", headers={"X-API-Key": "edge-key-events"}, json=ev)
    # a different caller reusing the same token+payload is not a conflict
    r2 = client.post("/v1/events", headers={"X-API-Key": "admin-key"}, json=ev)
    assert r1.status_code == 200 and r2.status_code == 200


# HIGH - model integrity must rest on an OUT-OF-BAND pinned digest, not the
# card's self-written hash.
def test_r2_pinned_model_digest_mismatch_is_rejected(prod_env, serving_artifact):
    from app.config import Settings
    from app.features import FEATURE_NAMES
    from app.model_loader import ModelArtifactInvalid, load_serving_model

    env = dict(prod_env)
    env["PRAMAAN_MODEL_DIR"] = str(serving_artifact)
    env["PRAMAAN_MODEL_SHA256"] = "00" * 32  # wrong pin
    with pytest.raises(ModelArtifactInvalid):
        load_serving_model(Settings.from_env(env), FEATURE_NAMES)


def test_r2_pinned_model_digest_match_loads(prod_env, serving_artifact):
    import hashlib

    from app.config import Settings
    from app.features import FEATURE_NAMES
    from app.model_loader import load_serving_model

    real = hashlib.sha256(
        (serving_artifact / "serving_anomaly.joblib").read_bytes()).hexdigest()
    env = dict(prod_env)
    env["PRAMAAN_MODEL_DIR"] = str(serving_artifact)
    env["PRAMAAN_MODEL_SHA256"] = real
    m = load_serving_model(Settings.from_env(env), FEATURE_NAMES)
    assert m.provenance != "DEMO_SYNTHETIC"


# MEDIUM - failed step-ups must NOT lock a victim out of their real one.
def test_r2_failed_stepups_do_not_block_a_valid_one(api, stepup_provider):
    client, _ = api
    for _ in range(5):  # garbage assertions (each 401)
        client.post("/v1/stepup/victim", headers={"X-API-Key": "stepup-key"},
                    json={"assertion": "garbage.token"})
    token = stepup_provider.issue(challenge_id="ok", identity_id="victim",
                                  method="otp_sms", result="pass")
    r = client.post("/v1/stepup/victim", headers={"X-API-Key": "stepup-key"},
                    json={"assertion": token})
    assert r.status_code == 200  # the valid step-up is not blocked by prior failures


# MEDIUM - audit tail-truncation is detectable via an out-of-band checkpoint.
def test_r2_audit_truncation_detected_by_head_checkpoint():
    from app.audit import AuditLog

    log = AuditLog(signing_key=b"soc-key")
    for i in range(4):
        log.append({"i": i})
    checkpoint = log.head_checkpoint()  # SOC persists this out-of-band
    log.records = log.records[:3]       # attacker drops the last record
    assert log.verify_against_checkpoint(checkpoint) is False


# LOW - NaN behavioural similarity must not slip past the [0,1] clamp as 1.0.
def test_r2_nan_similarity_is_not_trusted(attest_provider, behavior_provider, behavior_resolver):
    att = attest_provider.issue(device_id="devN")
    beh = behavior_provider.issue(device_id="devN", identity_id="u1", similarity=float("nan"))
    sim = behavior_resolver.resolve(attestation_token=att, behavior_token=beh,
                                    expected_identity_id="u1", expected_device_id="devN")
    assert sim is None


# LOW - wrong-kind token (a behavior token used as the attestation) is rejected.
def test_r2_attestation_kind_is_enforced(behavior_provider, behavior_resolver):
    not_an_attestation = behavior_provider.issue(device_id="devK", identity_id="u1",
                                                 similarity=0.95)
    sim = behavior_resolver.resolve(attestation_token=not_an_attestation,
                                    behavior_token=not_an_attestation,
                                    expected_identity_id="u1", expected_device_id="devK")
    assert sim is None


# LOW - a malformed (non-Ed25519) step-up pubkey must fail config, not
# silently downgrade to symmetric HMAC.
def test_r2_malformed_ed25519_pubkey_fails_config(prod_env):
    from app.config import ConfigError, Settings

    env = dict(prod_env)
    env["PRAMAAN_STEPUP_PUBKEY"] = "not-a-valid-32-byte-key"
    with pytest.raises(ConfigError):
        Settings.from_env(env)


# ======================================================================== # Crypto-shredding: erasure makes material irrecoverable WITHOUT breaking the
# immutable audit chain.
def test_crypto_shred_makes_material_irrecoverable():
    from app.keystore import Erased, KeyedPiiStore

    store = KeyedPiiStore()
    store.put("id1", {"pan": "XXXX1234", "name": "secret"})
    assert store.get("id1")["name"] == "secret"
    assert store.erase("id1") is True
    with pytest.raises(Erased):           # key destroyed → irrecoverable
        store.get("id1")
    assert store.is_erased("id1") is True  # ciphertext remains but is unreadable


def test_audit_chain_still_verifies_after_erasure():
    from app.audit import AuditLog

    log = AuditLog(signing_key=b"soc-key")
    log.append({"type": "assessment", "identity_id": "token-abc", "pii_ref": "deadbeef"})
    log.append({"type": "erasure", "identity_id": "token-abc", "method": "crypto_shred"})
    assert log.verify_chain() is True      # chain holds; only token refs inside


def test_erase_endpoint_is_soc_scoped_and_keeps_chain_intact(api):
    client, _ = api
    client.post("/v1/events", headers={"X-API-Key": "edge-key-events"},
                json=event_dict("erase-me"))
    # wrong scope is forbidden
    assert client.delete("/v1/identity/erase-me/erase",
                         headers={"X-API-Key": "edge-key-events"}).status_code == 403
    # identity:erase scope succeeds
    r = client.delete("/v1/identity/erase-me/erase", headers={"X-API-Key": "admin-key"})
    assert r.status_code == 200 and r.json()["erased"] is True
    assert r.json()["chain_intact"] is True
    av = client.get("/v1/audit/verify", headers={"X-API-Key": "soc-key-readonly"}).json()
    assert av["chain_intact"] is True


# ======================================================================== # Drift detection on the per-identity risk window catches a sustained, low-and-
# slow upward shift even when every single event stayed sub-threshold.
def test_drift_detector_flags_sustained_subthreshold_rise():
    from app.drift import DriftDetector

    d = DriftDetector()
    creeping = [0.10, 0.12, 0.15, 0.18, 0.22, 0.26, 0.30, 0.34, 0.38, 0.42]
    assert d.detect(creeping) is True   # rising trend, each value still < 0.45


def test_drift_detector_ignores_stable_low_risk():
    from app.drift import DriftDetector

    d = DriftDetector()
    stable = [0.12, 0.10, 0.13, 0.11, 0.12, 0.10, 0.11, 0.12, 0.10, 0.13]
    assert d.detect(stable) is False


def test_engine_drift_triggers_secondary_review(demo_engine, make_event):
    # A sequence of slowly-escalating sub-threshold events on a warmed identity
    # must eventually be escalated to a step-up (secondary review) by drift -
    # not silently allowed (which would let the profile be poisoned).
    from app.schemas import Decision

    ident = "creep"
    for _ in range(6):  # warm: known device/geo
        demo_engine.assess(make_event(ident, device_id="dk", geo="IN-GJ"))
    decisions = []
    for amt in range(10):  # creeping amounts on transactions, each modest
        from app.schemas import Channel, EventType

        soc = demo_engine.assess(make_event(
            ident, event_type=EventType.TRANSACTION, channel=Channel.INTERNET_BANKING,
            device_id="dk", geo="IN-GJ", amount=2000.0 + amt * 1500))
        decisions.append(soc.decision)
    assert Decision.STEP_UP in decisions or Decision.BLOCK in decisions


# ======================================================================= # Impossible-travel: a different country within an implausibly short interval is
# flagged and escalated.
def test_impossible_travel_is_flagged_and_escalated(demo_engine, make_event):
    from app.schemas import Decision

    ident = "traveler"
    for _ in range(4):  # establish presence in India (known device)
        demo_engine.assess(make_event(ident, device_id="dk", geo="IN-GJ"))
    soc = demo_engine.assess(make_event(ident, device_id="dk", geo="RU-MOW"))
    assert any("IMPOSSIBLE_TRAVEL" in r for r in soc.reason_codes)
    assert soc.decision in (Decision.STEP_UP, Decision.BLOCK)


def test_cold_start_prior_lowers_new_user_risk(make_event):
    # A genuine NEW user's first event scores LOWER risk with the cold-start
    # prior on than off (less onboarding friction).
    from app.config import Settings
    from app.risk_engine import TrustEngine

    s = Settings.from_env({"PRAMAAN_MODE": "demo_synthetic"})
    with_prior = TrustEngine(settings=s, behavior_resolver=None, cold_start_prior=True)
    without = TrustEngine(settings=s, behavior_resolver=None, cold_start_prior=False)
    r_on = with_prior.assess(make_event("newbie_a", device_id="d1", geo="IN-GJ")).event_risk
    r_off = without.assess(make_event("newbie_b", device_id="d1", geo="IN-GJ")).event_risk
    assert r_on < r_off  # the population prior dampens cold-start friction


# =========================================================== EXPLAINABILITY (P5)
def test_p5_shap_and_counterfactual_on_audit_plane_not_client(demo_engine, make_event):
    from app.schemas import Channel, EventType

    soc = demo_engine.assess(make_event(
        "xai", event_type=EventType.TRANSACTION, channel=Channel.INTERNET_BANKING,
        device_id="newdev", geo="RU-MOW", hour_of_day=3, amount=95_000.0,
        is_new_beneficiary=True))
    assert soc.shap_values and soc.counterfactual          # computed
    payload = soc.to_audit_payload()
    assert payload["shap_values"] and payload["counterfactual"]  # on the SOC plane
    blob = soc.to_client().model_dump_json().lower()
    assert "shap" not in blob and "counterfactual" not in blob   # never on client


# ============================================================================ #
# ROUND 3 - fixes for the internal security review (red→green).
# ============================================================================ #

# SECURITY REGRESSION - the cold-start prior must NOT soften a malicious
# "look new" first contact.
def test_r3_cold_start_prior_does_not_help_malicious_first_contact(make_event):
    from app.config import Settings
    from app.risk_engine import TrustEngine
    from app.schemas import Channel, Decision, EventType

    s = Settings.from_env({"PRAMAAN_MODE": "demo_synthetic"})
    mal = dict(event_type=EventType.TRANSACTION, channel=Channel.INTERNET_BANKING,
               device_id="atk", geo="RU-MOW", hour_of_day=3, amount=50_000.0,
               is_new_beneficiary=True)
    on = TrustEngine(settings=s, behavior_resolver=None,
                     cold_start_prior=True).assess(make_event("a", **mal))
    off = TrustEngine(settings=s, behavior_resolver=None,
                      cold_start_prior=False).assess(make_event("b", **mal))
    assert on.event_risk == pytest.approx(off.event_risk)   # prior had NO effect
    assert on.decision in (Decision.STEP_UP, Decision.BLOCK)


def test_r3_cold_start_prior_is_one_shot(demo_engine, make_event):
    # The prior is consumed on the FIRST cold assessment (even if the event is
    # stepped-up and never commits), so it can't be re-probed across retries.
    from app.schemas import Channel, EventType

    ident = "probe"
    # an attack-shaped first contact stays cold (STEP_UP → no commit), but the
    # one-shot flag must still be consumed.
    demo_engine.assess(make_event(
        ident, event_type=EventType.ACCOUNT_RECOVERY, channel=Channel.INTERNET_BANKING,
        device_id="atk", geo="RU-MOW", hour_of_day=3, recovery_contact_changed=True))
    st = demo_engine.store.load(ident)
    assert st.cold_prior_used is True       # consumed on first assessment
    assert st.event_count == 0              # and it never committed (stayed cold)


# SECURITY REGRESSION - impossible-travel reference must not be poisoned by
# un-allowed events; repeated attempts from the foreign country keep flagging.
def test_r3_impossible_travel_keeps_flagging_repeated_attempts(demo_engine, make_event):
    ident = "rt"
    for _ in range(4):  # known-good presence in India
        demo_engine.assess(make_event(ident, device_id="dk", geo="IN-GJ"))
    first = demo_engine.assess(make_event(ident, device_id="dk", geo="RU-MOW"))
    second = demo_engine.assess(make_event(ident, device_id="dk", geo="RU-MOW"))
    assert any("IMPOSSIBLE_TRAVEL" in r for r in first.reason_codes)
    assert any("IMPOSSIBLE_TRAVEL" in r for r in second.reason_codes)  # not poisoned


# drift CUSUM catches an arbitrarily-slow ramp the sliding window misses.
def test_r3_drift_cusum_catches_slow_ramp():
    from app.drift import DriftDetector

    d = DriftDetector()
    ewma = cusum = 0.0
    fired = False
    for i in range(40):
        risk = min(0.05 + 0.0133 * i, 0.44)   # slope below the window min_shift/6
        ewma, cusum, drift = d.step(ewma, cusum, risk)
        fired = fired or drift
    assert fired


def test_r3_drift_cusum_ignores_stable_benign():
    from app.drift import DriftDetector

    d = DriftDetector()
    ewma = cusum = 0.0
    fired = False
    for _ in range(60):
        ewma, cusum, drift = d.step(ewma, cusum, 0.12)
        fired = fired or drift
    assert not fired


# crypto-shredding erasure is durable: a post-erase event cannot resurrect.
def test_r3_erased_identity_is_not_resurrected():
    from app.keystore import Erased, KeyedPiiStore

    s = KeyedPiiStore()
    s.put("id", {"x": 1})
    s.erase("id")
    with pytest.raises(Erased):
        s.put("id", {"x": 2})        # tombstoned → refuses re-collection
    assert s.is_erased("id") is True


# the amount counterfactual is no longer the vacuous "₹0".
def test_r3_amount_counterfactual_is_not_vacuous_zero():
    from app.explain import counterfactual
    from app.risk_engine import WEIGHTS

    vec = [0, 0, 0.1, 1.0, 0.5, 0.4, 0.4, 0, 0, 0.1, 0]  # amount_zscore dominates

    class _E:
        amount = 500_000.0

    cf = counterfactual(vec, WEIGHTS, _E(), amount_ref=2000.0)
    assert "₹0 " not in cf and "₹0." not in cf and "≈ ₹0" not in cf
