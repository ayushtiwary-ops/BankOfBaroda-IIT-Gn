"""The Identity Trust Engine - stateless scoring over externalized state.

Hybrid scoring:
  - Unsupervised ML  : a REAL anomaly model loaded from a versioned artifact(- trained offline on real RBA logins, not np.random).
  - Deterministic    : weighted risk features - explainable, auditable.
  - Continuous trust : every identity carries a 0–1000 trust score that decays
    on risk and recovers through verified, normal behaviour - with recovery
    RATE-LIMITED so a patient attacker cannot slowly launder a poisoned profile
    (HARDENING, seed).

The engine holds NO per-identity state: it loads/commits ``IdentityState``
through a ``StateStore`` under a per-key lock + version CAS, so concurrent
same-identity requests cannot race ``trust``.

Output is a ``SocAssessment`` (full reasons + contributions for the audit/SOC
plane). The client-facing projection is produced in ``main.py`` and
carries no detector internals.
"""
import time
import uuid

import numpy as np

from .config import Settings
from .features import FEATURE_NAMES, commit_features, compute_features
from .model_loader import load_serving_model
from .policy import PolicyOrchestrator
from .resilience import ResiliencePolicy
from .schemas import Channel, Decision, EventType, IdentityEvent, SocAssessment
from .state_store import DEFAULT_TRUST, build_state_store
from .verifier import InMemoryNonceCache

REASON_TEXT = {
    "new_device": "NEW_DEVICE: first time this device is used for this identity",
    "new_geo": "GEO_ANOMALY: activity from an unseen location bucket",
    "hour_deviation": "TIME_ANOMALY: activity far outside usual hours",
    "amount_zscore": "AMOUNT_ANOMALY: amount deviates strongly from history",
    "behavior_anomaly": "BEHAVIOUR_MISMATCH: typing/swipe pattern does not match owner",
    "event_criticality": "SENSITIVE_ACTION: inherently high-risk operation",
    "channel_risk": "CHANNEL_RISK: elevated-risk channel",
    "new_beneficiary": "NEW_BENEFICIARY: payment to a never-seen payee",
    "recovery_change": "RECOVERY_CHANGE: recovery contact being modified",
    "velocity": "VELOCITY: unusually rapid activity burst",
    "device_sharing": "DEVICE_SHARING: device linked to multiple identities (mule-farm pattern)",
}
DRIFT_REASON = "DRIFT_REVIEW: sustained low-and-slow risk increase - secondary review"
IMPOSSIBLE_TRAVEL_REASON = "IMPOSSIBLE_TRAVEL: geo changed faster than physically possible"

WEIGHTS = [0.14, 0.10, 0.05, 0.11, 0.15, 0.07, 0.04, 0.08, 0.12, 0.04, 0.10]

# HARDENING: capped / rate-limited passive trust recovery (anti slow-drift).
RECOVERY_CAP_PER_WINDOW = 120
RECOVERY_WINDOW_SECONDS = 3600.0
RISK_WINDOW_MAX = 16                 # per-identity risk history for drift
COLD_PRIOR_FACTOR = 0.5              # dampen new-device/new-geo for new users
COLD_PRIOR_MAX_AMOUNT = 25_000.0     # prior never applies above this amount
IMPOSSIBLE_TRAVEL_SECONDS = 3600.0   # different country within 1h = implausible
IMPOSSIBLE_TRAVEL_BOOST = 0.5
_CAS_RETRIES = 6

# the cold-start prior is allowed ONLY for benign-shaped first contacts, so
# it can never soften the "look new" attacker (new device + new geo + odd hour +
# big amount / new payee / recovery / privileged).
_COLD_PRIOR_OK_EVENTS = frozenset({EventType.LOGIN, EventType.TRANSACTION})
_COLD_PRIOR_OK_CHANNELS = frozenset(
    {Channel.MOBILE_APP, Channel.INTERNET_BANKING, Channel.BRANCH})


def _country(geo: str) -> str:
    return (geo or "").split("-")[0]


def _cold_prior_applies(e: IdentityEvent) -> bool:
    return (e.event_type in _COLD_PRIOR_OK_EVENTS
            and e.channel in _COLD_PRIOR_OK_CHANNELS
            and not e.is_new_beneficiary
            and not e.recovery_contact_changed
            and (e.amount is None or e.amount < COLD_PRIOR_MAX_AMOUNT))


class TrustEngine:
    BASE_TRUST = DEFAULT_TRUST

    def __init__(self, *, settings=None, store=None, serving_model=None,
                 behavior_resolver="auto", drift_enabled=True,
                 recovery_cap=RECOVERY_CAP_PER_WINDOW, cold_start_prior=True,
                 impossible_travel=True):
        self.settings = settings or Settings.from_env()
        self.store = store or build_state_store(self.settings.redis_url)
        # load the REAL model; in prod this raises if the artifact is absent.
        self.model = serving_model or load_serving_model(self.settings, FEATURE_NAMES)
        self.policy = PolicyOrchestrator()
        self.resilience = ResiliencePolicy()
        from .drift import DriftDetector

        self.drift = DriftDetector()
        #  toggles - set the "legacy" combination to reproduce the OLD
        # (poisonable, friction-bombing) engine for before/after demos.
        self.drift_enabled = drift_enabled
        self.recovery_cap = recovery_cap
        self.cold_start_prior = cold_start_prior
        self.impossible_travel = impossible_travel
        if behavior_resolver == "auto":
            behavior_resolver = self._build_resolver(self.settings)
        self.behavior_resolver = behavior_resolver

    @classmethod
    def legacy(cls, **kw):
        """The legacy engine: no drift detection, uncapped recovery, no
        cold-start prior - for adversarial before/after comparisons."""
        return cls(drift_enabled=False, recovery_cap=10 ** 9,
                   cold_start_prior=False, impossible_travel=False, **kw)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _build_resolver(settings):
        try:
            from .attestation import BehaviorResolver
            from .verifier import build_verifier

            return BehaviorResolver(
                attest_verifier=build_verifier(settings.attest_pubkey),
                behavior_verifier=build_verifier(settings.behavior_pubkey),
                nonce_cache=InMemoryNonceCache(),
           )
        except Exception:  # pragma: no cover - misconfigured keys → cold-start
            return None

    @property
    def model_mode(self) -> str:
        return "demo_synthetic" if self.model.is_synthetic else "prod"

    def _resolve_behavior(self, e: IdentityEvent):
        """trusted similarity from a signed/attested assertion, or None."""
        if self.behavior_resolver is None:
            return None
        try:
            return self.behavior_resolver.resolve(
                attestation_token=e.device_attestation,
                behavior_token=e.behavior_assertion,
                expected_identity_id=e.identity_id,
                expected_device_id=e.device_id,
           )
        except Exception:  # pragma: no cover - any failure → MISSING (safe)
            return None

    def _explain(self, vec):
        contributions = sorted(
            ((w * f, name) for w, f, name in zip(WEIGHTS, vec, FEATURE_NAMES)),
            reverse=True,
       )
        reasons = [REASON_TEXT[name] for c, name in contributions[:3] if c > 0.05]
        if not reasons:
            reasons = ["NORMAL: behaviour consistent with identity baseline"]
        return reasons, contributions[:5]

    def _update_trust(self, state, event_risk: float, now: float) -> int:
        prev = state.trust
        if event_risk > 0.45:
            return int(np.clip(prev - int((event_risk - 0.45) * 900), 0, 1000))
        # passive recovery - rate-limited within a rolling window
        gain = int((0.45 - event_risk) * 60)
        if now - state.recovery_window_start > RECOVERY_WINDOW_SECONDS:
            state.recovery_window_start = now
            state.recovered_in_window = 0
        allowed = max(0, self.recovery_cap - state.recovered_in_window)
        gain = min(gain, allowed)
        state.recovered_in_window += gain
        return int(np.clip(prev + gain, 0, 1000))

    def _impossible_travel(self, state, e: IdentityEvent, now: float) -> bool:
        # a different country within an implausibly short interval.
        if not self.impossible_travel or not state.last_geo:
            return False
        if _country(e.geo) == _country(state.last_geo):
            return False
        return (now - state.last_geo_ts) < IMPOSSIBLE_TRAVEL_SECONDS

    # ------------------------------------------------------------------ #
    def assess(self, e: IdentityEvent) -> SocAssessment:
        t0 = time.perf_counter()
        sim = self._resolve_behavior(e)
        behavior_anomaly = None if sim is None else (1.0 - sim)

        decision = band = step_up = None
        new_trust = ml_risk = det_risk = event_risk = None
        degraded = False
        reasons: list[str] = []
        contributions: list = []

        now_wall = time.time()
        for _ in range(_CAS_RETRIES):
            with self.store.lock(e.identity_id):
                state = self.store.load(e.identity_id)
                cold = state.event_count == 0
                dev_count = self.store.device_add(e.device_id, e.identity_id)
                vec = compute_features(e, state, dev_count, behavior_anomaly)
                # cold-start population prior - dampen new-device/
                # new-geo ONLY for benign-shaped first contacts, and only ONCE
                # (one-shot), so it cannot soften a "look new" attacker or be
                # re-probed across non-committed retries.
                if (cold and self.cold_start_prior and not state.cold_prior_used
                        and _cold_prior_applies(e)):
                    vec[0] *= COLD_PRIOR_FACTOR
                    vec[1] *= COLD_PRIOR_FACTOR
                if cold:
                    state.cold_prior_used = True

                degraded = False
                try:
                    ml_risk = self.model.risk(vec)  # real model
                except Exception:                    # HARDENING: degrade safely
                    ml_risk = None
                    degraded = True
                det_risk = float(np.clip(sum(w * f for w, f in zip(WEIGHTS, vec)),
                                         0.0, 1.0))
                event_risk = det_risk if ml_risk is None else (0.55 * ml_risk
                                                               + 0.45 * det_risk)

                reasons, contributions = self._explain(vec)
                # impossible-travel deterministic override.
                impossible = self._impossible_travel(state, e, now_wall)
                if impossible:
                    event_risk = float(np.clip(event_risk + IMPOSSIBLE_TRAVEL_BOOST,
                                               0.0, 1.0))
                    reasons = [IMPOSSIBLE_TRAVEL_REASON, *reasons]

                # drift - fast sliding-window mean-shift OR a persistent-
                # baseline CUSUM that catches arbitrarily-slow ramps .
                state.risk_window = (state.risk_window + [round(event_risk, 4)])[-RISK_WINDOW_MAX:]
                state.risk_ewma, state.cusum, cusum_fired = self.drift.step(
                    state.risk_ewma, state.cusum, event_risk)
                window_fired = not cold and self.drift.detect(state.risk_window)
                drifting = self.drift_enabled and (cusum_fired or window_fired)

                new_trust = self._update_trust(state, event_risk, now_wall)
                decision, band, step_up = self.policy.decide(new_trust, event_risk, e)
                if (drifting or state.under_review) and decision == Decision.ALLOW:
                    # secondary review - sticky until a verified step-up clears it;
                    # halts poisoning too (no ALLOW → no commit).
                    decision = Decision.STEP_UP
                    band = "DRIFT_REVIEW"
                    step_up = self.policy._cheapest_sufficient(e, strong=True)
                    reasons = [DRIFT_REASON, *reasons]
                    state.under_review = True
                if degraded:
                    decision, band, step_up = self.resilience.apply(
                        decision, band, step_up, e)

                state.trust = new_trust
                if decision == Decision.ALLOW:  # anti-poisoning: trusted events only
                    commit_features(state, e)
                    # only a TRUSTED event may move the impossible-travel
                    # reference (mirrors commit_features) - an un-allowed event
                    # from a new country must not poison it away.
                    state.last_geo, state.last_geo_ts = e.geo, now_wall
                if self.store.commit(e.identity_id, state, state.version):
                    break

        # DPDP/RBI explainability - exact additive SHAP + one counterfactual,
        # attached to the SOC/audit plane only (never the client;).
        from .explain import counterfactual as _cf
        from .explain import shap_values as _shap

        amount_ref = (sum(state.amounts) / len(state.amounts)) if state.amounts else None
        shap = _shap(vec, WEIGHTS)
        cf = _cf(vec, WEIGHTS, e, amount_ref=amount_ref)

        challenge_id = uuid.uuid4().hex[:12] if decision == Decision.STEP_UP else None
        return SocAssessment(
            event_id=uuid.uuid4().hex[:12],
            identity_id=e.identity_id,
            trust_score=new_trust,
            risk_band=band,
            decision=decision,
            step_up_method=step_up,
            challenge_id=challenge_id,
            reason_codes=reasons,
            feature_contributions=contributions,
            shap_values=shap,
            counterfactual=cf,
            ml_risk=ml_risk,
            det_risk=det_risk,
            event_risk=event_risk,
            model_provenance=self.model.provenance,
            model_mode=self.model_mode,
            degraded=degraded,
            latency_ms=round((time.perf_counter() - t0) * 1000, 2),
       )

    # ------------------------------------------------------------------ #
    def apply_verified_step_up(self, identity_id: str, passed: bool) -> int:
        """Apply a step-up outcome that has ALREADY been validated as a signed
        verifier assertion (see main.py / verifier.py). Verified → trust
        restored (bounded); failed → trust craters. This method must NEVER be
        called on a self-asserted client boolean - that is the whole of ."""
        new = self.BASE_TRUST
        for _ in range(_CAS_RETRIES):
            with self.store.lock(identity_id):
                state = self.store.load(identity_id)
                prev = state.trust
                new = min(prev + 250, 1000) if passed else max(prev - 400, 0)
                state.trust = new
                if passed:
                    state.under_review = False  # a verified step-up clears review
                    state.risk_window = []      # reset the drift window post-verification
                    state.cusum = 0.0           # reset CUSUM after verification
                if self.store.commit(identity_id, state, state.version):
                    break
        return new

    def get_trust(self, identity_id: str) -> int:
        return self.store.load(identity_id).trust

    def identity_snapshot(self, identity_id: str) -> dict:
        """Minimal, SOC-scoped view (no raw PII; gated behind identity:read)."""
        s = self.store.load(identity_id)
        return {
            "trust_score": s.trust,
            "known_devices": len(s.devices),
            "events_seen": s.event_count,
        }
