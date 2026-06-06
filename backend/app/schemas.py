"""Pydantic schemas — the API contract for PRAMAAN.

Trust-boundary note (KS2 + KS8):
  * IdentityEvent NO LONGER accepts a client ``behavior_score``. Behaviour is
    trusted only via signed, device-attested assertions (``device_attestation``
    + ``behavior_assertion``); ``extra="forbid"`` rejects any attempt to sneak
    the old field back in.
  * The CLIENT-facing ``RiskAssessment`` carries a generic decision only — no
    detector reason codes, no numeric trust score (both are attacker oracles).
    The rich ``SocAssessment`` goes to the audit/SOC plane exclusively.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class EventType(str, Enum):
    LOGIN = "login"
    TRANSACTION = "transaction"
    ONBOARDING = "onboarding"
    ACCOUNT_RECOVERY = "account_recovery"
    PRIVILEGED_ACCESS = "privileged_access"
    PROFILE_CHANGE = "profile_change"


class Channel(str, Enum):
    MOBILE_APP = "mobile_app"
    INTERNET_BANKING = "internet_banking"
    BRANCH = "branch"
    API = "api"
    ADMIN_CONSOLE = "admin_console"


class Decision(str, Enum):
    ALLOW = "ALLOW"
    STEP_UP = "STEP_UP"
    BLOCK = "BLOCK"


class StepUpMethod(str, Enum):
    DEVICE_BIOMETRIC = "device_biometric"     # lowest friction
    OTP_SMS = "otp_sms"
    SECURITY_QUESTIONS = "security_questions"
    VIDEO_KYC = "video_kyc"                   # highest assurance
    MANAGER_APPROVAL = "manager_approval"     # for privileged access


class IdentityEvent(BaseModel):
    """A single identity-relevant event from any channel.

    Identifiers arriving here are already pseudonymized at the edge (privacy.py).
    No raw PII / biometrics ever reach the engine.
    """
    model_config = ConfigDict(extra="forbid")  # SECURITY: KS2 — no smuggled fields

    identity_id: str = Field(..., description="Pseudonymous identity token")
    event_type: EventType
    channel: Channel
    device_id: str = Field(..., description="Pseudonymous device fingerprint hash")
    geo: str = Field(..., description="Coarse geo bucket, e.g. 'IN-GJ' (never exact location)")
    hour_of_day: int = Field(..., ge=0, le=23)
    amount: Optional[float] = Field(None, description="Txn amount (transactions only)")
    is_new_beneficiary: bool = False
    recovery_contact_changed: bool = False
    privileged_scope: Optional[str] = Field(None, description="e.g. 'core_banking.write'")

    # KS2: behavioural signal is trusted ONLY when it rides these signed tokens.
    # Absent → MISSING → cold-start neutral (never an assumed 0.99).
    device_attestation: Optional[str] = Field(
        None, description="Play Integrity / App Attest token (signed by the platform)")
    behavior_assertion: Optional[str] = Field(
        None, description="On-device behavioural-similarity assertion (signed, device-bound)")

    # x-factor: replay/duplicate protection at ingestion.
    idempotency_key: Optional[str] = Field(
        None, description="Caller-supplied key to dedupe retried/replayed events")


class RiskAssessment(BaseModel):
    """CLIENT plane — a generic decision only. No detector internals (KS8)."""
    event_id: str
    decision: Decision
    step_up_method: Optional[StepUpMethod] = None
    challenge_id: Optional[str] = None  # present iff a step-up is required
    message: str
    model_mode: str  # "prod" | "demo_synthetic" — provenance honesty stamp (KS3)
    latency_ms: float


class StepUpAssertionRequest(BaseModel):
    """The signed step-up token, submitted as a request BODY (never a query bool)."""
    assertion: str = Field(..., description="Signed verifier assertion token")


class StepUpResult(BaseModel):
    challenge_id: str
    decision: Decision
    message: str


_CLIENT_MESSAGE = {
    Decision.ALLOW: "Approved.",
    Decision.STEP_UP: "Additional verification is required to continue.",
    Decision.BLOCK: "This request could not be completed. Please contact support.",
}


@dataclass
class SocAssessment:
    """SOC / audit plane — the full, sensitive picture. Never sent to a client."""
    event_id: str
    identity_id: str
    trust_score: int
    risk_band: str
    decision: Decision
    step_up_method: Optional[StepUpMethod]
    challenge_id: Optional[str]
    reason_codes: list[str]
    feature_contributions: list[tuple] = field(default_factory=list)
    shap_values: list[tuple] = field(default_factory=list)   # exact additive Shapley
    counterfactual: str = ""                                  # one actionable change
    ml_risk: Optional[float] = None
    det_risk: float = 0.0
    event_risk: float = 0.0
    model_provenance: str = "unknown"
    model_mode: str = "prod"
    degraded: bool = False
    latency_ms: float = 0.0

    def to_client(self) -> "RiskAssessment":
        return to_client_assessment(self)

    def to_audit_payload(self) -> dict:
        """The rich record written to the tamper-evident audit chain (SOC only)."""
        return {
            "type": "assessment",
            "event_id": self.event_id,
            "identity_id": self.identity_id,  # pseudonymous token only
            "trust_score": self.trust_score,
            "risk_band": self.risk_band,
            "decision": self.decision.value,
            "reasons": self.reason_codes,
            "feature_contributions": [[round(c, 4), n] for c, n in self.feature_contributions],
            "shap_values": [[c, n] for c, n in self.shap_values],
            "counterfactual": self.counterfactual,
            "ml_risk": self.ml_risk,
            "det_risk": round(self.det_risk, 4),
            "event_risk": round(self.event_risk, 4),
            "model_provenance": self.model_provenance,
            "model_mode": self.model_mode,
            "degraded": self.degraded,
        }


def to_client_assessment(soc: SocAssessment) -> RiskAssessment:
    """Project the SOC assessment down to the client-safe subset."""
    return RiskAssessment(
        event_id=soc.event_id,
        decision=soc.decision,
        step_up_method=soc.step_up_method if soc.decision == Decision.STEP_UP else None,
        challenge_id=soc.challenge_id if soc.decision == Decision.STEP_UP else None,
        message=_CLIENT_MESSAGE[soc.decision],
        model_mode=soc.model_mode,
        latency_ms=soc.latency_ms,
    )
