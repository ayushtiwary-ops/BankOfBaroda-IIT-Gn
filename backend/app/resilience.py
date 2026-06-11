"""Degradation policy — explicit fail-open vs fail-closed per event type.

HARDENING (resilience): the original service had no answer for "what happens
when the model/state is unavailable?" — leaving either a fraud window
(fail-open everything) or a business outage (fail-closed everything). PRAMAAN
makes the choice explicit and risk-tiered:

  * High-assurance events (privileged access, account recovery, onboarding) →
    FAIL CLOSED: never silently ALLOW on a degraded engine; escalate to at
    least STEP_UP (BLOCK is preserved). Better friction than fraud.
  * Routine, low-risk events (e.g. login on a known device) → FAIL OPEN:
    preserve availability, but stamp ``degraded`` so the SOC sees reduced
    assurance and the decision is auditable.
"""
from .schemas import Decision, EventType, IdentityEvent, StepUpMethod

FAIL_CLOSED_EVENTS = frozenset({
    EventType.PRIVILEGED_ACCESS,
    EventType.ACCOUNT_RECOVERY,
    EventType.ONBOARDING,
})


class ResiliencePolicy:
    def apply(self, decision: Decision, band: str,
              step_up: StepUpMethod | None,
              e: IdentityEvent) -> tuple[Decision, str, StepUpMethod | None]:
        """Adjust a decision computed under degraded conditions."""
        if e.event_type in FAIL_CLOSED_EVENTS:
            if decision == Decision.ALLOW:
                # never auto-approve a sensitive action without a working model
                method = (StepUpMethod.MANAGER_APPROVAL
                          if e.event_type == EventType.PRIVILEGED_ACCESS
                          else StepUpMethod.VIDEO_KYC)
                return Decision.STEP_UP, "DEGRADED_FAILCLOSED", method
            return decision, "DEGRADED_FAILCLOSED", step_up
        # low-risk: keep availability, but the band records the degradation
        return decision, "DEGRADED_FAILOPEN", step_up
