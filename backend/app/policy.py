"""Risk Policy Orchestrator - converts (trust score, event risk, context)
into a friction-optimized decision.

Design principle: *verify only when risk is elevated*. The vast majority of
events sail through silently; step-up is targeted and the challenge chosen is
the LEAST intrusive method that provides sufficient assurance.
"""
from .schemas import Decision, EventType, IdentityEvent, StepUpMethod


class PolicyOrchestrator:
    def decide(
        self, trust_score: int, event_risk: float, e: IdentityEvent
   ) -> tuple[Decision, str, StepUpMethod | None]:

        # --- risk banding -------------------------------------------------
        if trust_score >= 650 and event_risk < 0.45:
            band = "LOW"
        elif trust_score >= 400 and event_risk < 0.65:
            band = "ELEVATED"
        elif trust_score >= 150:
            band = "HIGH"
        else:
            band = "CRITICAL"

        # --- privileged access is always least-privilege ------------------
        if e.event_type == EventType.PRIVILEGED_ACCESS:
            if band == "LOW":
                return Decision.ALLOW, band, None
            if band in ("ELEVATED", "HIGH"):
                return Decision.STEP_UP, band, StepUpMethod.MANAGER_APPROVAL
            return Decision.BLOCK, band, None

        # --- customer-facing decisions ------------------------------------
        if band == "LOW":
            return Decision.ALLOW, band, None

        if band == "ELEVATED":
            return Decision.STEP_UP, band, self._cheapest_sufficient(e, strong=False)

        if band == "HIGH":
            return Decision.STEP_UP, band, self._cheapest_sufficient(e, strong=True)

        return Decision.BLOCK, band, None

    # ---------------------------------------------------------------- #
    @staticmethod
    def _cheapest_sufficient(e: IdentityEvent, strong: bool) -> StepUpMethod:
        """Pick the lowest-friction challenge that still covers the risk."""
        if e.event_type in (EventType.ACCOUNT_RECOVERY, EventType.ONBOARDING):
            return StepUpMethod.VIDEO_KYC          # identity itself is in doubt
        if strong:
            return StepUpMethod.OTP_SMS
        return StepUpMethod.DEVICE_BIOMETRIC       # one-touch, ~2s of friction
