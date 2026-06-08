"""PRAMAAN API — FastAPI service exposing the Identity Trust Engine.

Endpoints (every one is authenticated + scoped — KS4):
POST   /v1/events                 events:write   score an event → generic decision
POST   /v1/stepup/{identity_id}   stepup:write   apply a SIGNED step-up assertion
GET    /v1/identity/{identity_id} identity:read  SOC-scoped minimal snapshot (no IDOR)
GET    /v1/audit                  audit:read     tamper-evident audit tail (SOC plane)
GET    /v1/audit/verify           audit:read     verify hash-chain integrity
GET    /health                    public         liveness + mode/provenance stamp
GET    /                          public         static dashboard

Trust boundaries enforced here:
  * KS1 — step-up arrives ONLY as a signed verifier assertion (request body),
    validated server-side. A self-asserted ``?verified=true`` cannot move trust.
  * KS8 — clients get a generic decision; the rich SocAssessment (reasons,
    trust score, feature contributions) goes ONLY to the audit/SOC plane.
  * KS4 — CORS is an explicit allowlist, never "*"; secrets come from config.
"""
import hashlib
import hmac
import json
import time
from collections import OrderedDict, defaultdict
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from .audit import AuditLog
from .auth import make_auth
from .config import Settings
from .keystore import KeyedPiiStore
from .risk_engine import TrustEngine
from .schemas import (
    Decision,
    IdentityEvent,
    RiskAssessment,
    StepUpAssertionRequest,
    StepUpResult,
)
from .verifier import AssertionRejected, InMemoryNonceCache

# Startup config — fails LOUD in prod if a required secret is unset (KS4d).
settings = Settings.from_env()
# Engine load — fails LOUD in prod if the real model artifact is missing (KS3).
engine = TrustEngine(settings=settings)
# Keyed audit chain (KS4 secret + audit x-factor).
audit = AuditLog(signing_key=settings.audit_signing_key)
# Step-up validator holds only the verifier PUBLIC key + an anti-replay cache.
stepup_validator = settings.stepup_validator(nonce_cache=InMemoryNonceCache())
# KS6 keyed PII vault — per-identity material, crypto-shreddable on erasure.
pii_vault = KeyedPiiStore()
authenticate, require = make_auth(settings)

DEMO_DIR = Path(__file__).resolve().parents[2] / "demo"

# x-factor in-process caches (documented to move to Redis in prod alongside KS7).
# SECURITY: R2 — bounded LRU; key = (caller, idempotency_key) → (fingerprint, response).
_IDEM_MAX = 10_000
_idem_cache: "OrderedDict[tuple[str, str], tuple[str, RiskAssessment]]" = OrderedDict()
# SECURITY: R2 — accepted step-ups (anti-fatigue) tracked SEPARATELY from
# failed ones, so garbage assertions cannot lock a victim out of a real step-up.
_stepup_accepted: dict[str, list[float]] = defaultdict(list)
_stepup_failures: dict[str, list[float]] = defaultdict(list)
STEPUP_ACCEPTED_MAX = 5      # anti-MFA-fatigue: cap successful step-ups per window
STEPUP_FAILURE_MAX = 20      # higher cap; only flags flooding, won't lock a few typos
STEPUP_WINDOW_SECONDS = 300.0
_STEPUP_MESSAGE = {
    Decision.ALLOW: "Verification successful.",
    Decision.BLOCK: "Verification failed.",
}


def _event_fingerprint(event: IdentityEvent) -> str:
    body = event.model_dump(mode="json", exclude={"idempotency_key"})
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()


def _rate_count(bucket: dict[str, list[float]], identity_id: str, now: float) -> int:
    """Prune the window and reclaim empty buckets (no unbounded growth)."""
    q = bucket.get(identity_id)
    if q is None:
        return 0
    q[:] = [t for t in q if now - t < STEPUP_WINDOW_SECONDS]
    if not q:
        del bucket[identity_id]
        return 0
    return len(q)

app = FastAPI(
    title="PRAMAAN — Identity Trust Engine",
    description="Privacy-first, Risk-Adaptive, Multi-channel Authentication & Anomaly Network",
    version="1.0.0",
)
# SECURITY: KS4(a) — explicit allowlist from config, never "*".
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["X-API-Key", "Content-Type"],
)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "pramaan",
        "mode": settings.mode,
        "model": engine.model_mode,          # provenance honesty stamp (KS3)
        "audit_chain_intact": audit.verify_chain(),
    }


@app.post("/v1/events", response_model=RiskAssessment)
def ingest_event(event: IdentityEvent,
                 ctx=Depends(require("events:write"))) -> RiskAssessment:
    # SECURITY: R2 — idempotency is bound to (caller, key, PAYLOAD fingerprint).
    # A replayed IDENTICAL event returns the prior decision; the same key reused
    # with a DIFFERENT payload is a 409 conflict — never a laundered ALLOW.
    cache_key = fingerprint = None
    if event.idempotency_key:
        cache_key = (ctx.client_id, event.idempotency_key)
        fingerprint = _event_fingerprint(event)
        cached = _idem_cache.get(cache_key)
        if cached is not None:
            stored_fp, stored_resp = cached
            if not hmac.compare_digest(stored_fp, fingerprint):
                raise HTTPException(
                    status_code=409,
                    detail="idempotency key reused with a different payload")
            _idem_cache.move_to_end(cache_key)
            return stored_resp

    soc = engine.assess(event)
    # KS6: stash per-identity material in the keyed vault under a non-reversible
    # ref; the audit chain carries only the token + ref, never plaintext.
    # R3: a tombstoned (erased) identity is scored but NOT re-vaulted — erasure
    # is durable; re-collection needs explicit re-consent.
    pii_ref = None
    if not pii_vault.is_erased(event.identity_id):
        pii_ref = pii_vault.put(event.identity_id,
                                {"identity_token": event.identity_id, "geo": event.geo,
                                 "device": event.device_id})
    payload = soc.to_audit_payload()
    payload["pii_ref"] = pii_ref
    audit.append(payload)                     # FULL reasons → SOC plane (KS8)
    client = soc.to_client()                 # generic decision only → client
    if cache_key is not None:
        _idem_cache[cache_key] = (fingerprint, client)
        _idem_cache.move_to_end(cache_key)
        while len(_idem_cache) > _IDEM_MAX:
            _idem_cache.popitem(last=False)  # evict oldest (bounded)
    return client


@app.post("/v1/stepup/{identity_id}", response_model=StepUpResult)
def step_up_outcome(identity_id: str, body: StepUpAssertionRequest,
                    ctx=Depends(require("stepup:write"))) -> StepUpResult:
    # SECURITY: KS1 — the outcome must be a SIGNED verifier assertion.
    # There is no `verified` query param to set; `?verified=true` does nothing.
    now = time.time()
    # Failure flood guard (high cap) — does NOT consume the accepted budget.
    if _rate_count(_stepup_failures, identity_id, now) >= STEPUP_FAILURE_MAX:
        raise HTTPException(status_code=429, detail="too many failed step-up attempts")
    try:
        assertion = stepup_validator.validate(body.assertion,
                                              expected_identity_id=identity_id)
    except AssertionRejected as exc:
        _stepup_failures.setdefault(identity_id, []).append(now)
        audit.append({"type": "step_up_rejected", "identity_id": identity_id,
                      "reason": type(exc).__name__})
        raise HTTPException(status_code=401, detail="invalid step-up assertion")

    # SECURITY: R2 — anti-fatigue cap applies ONLY to VALIDATED step-ups, so
    # garbage submissions can never lock a victim out of their real step-up.
    if _rate_count(_stepup_accepted, identity_id, now) >= STEPUP_ACCEPTED_MAX:
        raise HTTPException(status_code=429, detail="too many step-up attempts")
    _stepup_accepted.setdefault(identity_id, []).append(now)

    new_trust = engine.apply_verified_step_up(identity_id, assertion.passed)
    audit.append({"type": "step_up", "identity_id": identity_id,
                  "method": assertion.method, "result": assertion.result,
                  "challenge_id": assertion.challenge_id,
                  "new_trust_score": new_trust})  # trust score → SOC plane only
    decision = Decision.ALLOW if assertion.passed else Decision.BLOCK
    return StepUpResult(challenge_id=assertion.challenge_id, decision=decision,
                        message=_STEPUP_MESSAGE[decision])


@app.get("/v1/identity/{identity_id}")
def identity_state(identity_id: str, ctx=Depends(require("identity:read"))):
    # SECURITY: KS4(c) — formerly public IDOR/enumeration endpoint, now
    # gated behind an internal SOC scope and returning only non-sensitive fields.
    return {"identity_id": identity_id, **engine.identity_snapshot(identity_id)}


@app.delete("/v1/identity/{identity_id}/erase")
def erase_identity(identity_id: str, ctx=Depends(require("identity:erase"))):
    # SECURITY: KS6 — crypto-shredding. Destroy the per-identity key so the
    # vault material is irrecoverable, while the hash chain stays intact and
    # verifiable. The erasure itself is recorded (token only) for accountability.
    existed = pii_vault.erase(identity_id)
    audit.append({"type": "erasure", "identity_id": identity_id,
                  "method": "crypto_shred", "existed": existed})
    return {"identity_id": identity_id, "erased": existed,
            "method": "crypto_shred", "chain_intact": audit.verify_chain()}


@app.get("/v1/audit")
def audit_tail(n: int = 50, ctx=Depends(require("audit:read"))):
    return {"chain_intact": audit.verify_chain(), "records": audit.tail(n)}


@app.get("/v1/audit/verify")
def audit_verify(ctx=Depends(require("audit:read"))):
    # head_checkpoint (R2) lets the SOC detect tail-truncation out-of-band.
    return {"chain_intact": audit.verify_chain(), "records": len(audit.records),
            "head_checkpoint": audit.head_checkpoint()}


@app.get("/")
def dashboard():
    return FileResponse(DEMO_DIR / "dashboard.html")
