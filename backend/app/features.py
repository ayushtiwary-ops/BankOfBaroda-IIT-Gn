"""Feature extraction — turns an IdentityEvent into a numeric vector.

SECURITY: KS7 — these are now stateless helpers that operate on an injected
``IdentityState`` (loaded from the StateStore) instead of process-global dicts.
The cross-identity device graph lives in the StateStore too, so scoring pods
hold no per-identity state.

SECURITY: KS2 — the behavioural feature is fed a server-resolved similarity
(from a signed, attested assertion) or ``None``. ``None`` means MISSING and maps
to a neutral cold-start value — never an assumed owner-like 0.99.
"""
from .schemas import Channel, EventType, IdentityEvent
from .state_store import IdentityState

EVENT_CRITICALITY = {
    EventType.LOGIN: 0.2,
    EventType.TRANSACTION: 0.4,
    EventType.PROFILE_CHANGE: 0.5,
    EventType.ONBOARDING: 0.6,
    EventType.ACCOUNT_RECOVERY: 0.8,
    EventType.PRIVILEGED_ACCESS: 0.9,
}

CHANNEL_RISK = {
    Channel.BRANCH: 0.1,
    Channel.MOBILE_APP: 0.3,
    Channel.INTERNET_BANKING: 0.4,
    Channel.API: 0.5,
    Channel.ADMIN_CONSOLE: 0.7,
}

FEATURE_NAMES = [
    "new_device",          # 1 = device never seen for this identity (0.4 on cold start)
    "new_geo",             # 1 = geo bucket never seen (0.5 on cold start)
    "hour_deviation",      # 0..1 distance from identity's usual active hours
    "amount_zscore",       # txn amount vs identity's own mean/std (capped)
    "behavior_anomaly",    # 1 - attested behavioural similarity (0.5 = MISSING/cold start)
    "event_criticality",   # inherent risk of the action
    "channel_risk",        # inherent risk of the channel
    "new_beneficiary",     # 1 = paying someone new
    "recovery_change",     # 1 = recovery contact being changed
    "velocity",            # decaying burst counter (rapid-fire activity)
    "device_sharing",      # distinct identities seen on this device (mule farms)
]

NEUTRAL_BEHAVIOR_ANOMALY = 0.5  # MISSING behaviour → cold-start neutral


def _hour_deviation(state: IdentityState, hour: int) -> float:
    if not state.hours:
        return 0.5  # unknown identity → neutral
    diffs = [min(abs(hour - h), 24 - abs(hour - h)) for h in state.hours[-200:]]
    return min(min(diffs) / 12.0, 1.0)


def _amount_zscore(state: IdentityState, amount: float | None) -> float:
    if amount is None:
        return 0.0
    hist = state.amounts[-200:]
    if len(hist) < 5:
        return min(amount / 100_000.0, 1.0)  # cold start: scale by absolute size
    mean = sum(hist) / len(hist)
    var = sum((a - mean) ** 2 for a in hist) / len(hist)
    std = max(var ** 0.5, 1.0)
    return min(abs(amount - mean) / (3 * std), 1.0)


def compute_features(e: IdentityEvent, state: IdentityState,
                     device_share_count: int,
                     behavior_anomaly: float | None = None) -> list[float]:
    """Build the 11-dim serving vector. Mutates ``state.burst`` (velocity is a
    property of every attempt, allowed or not). Profile membership
    (devices/geos/hours/amounts) is folded in only by ``commit_features``."""
    cold = state.event_count == 0
    state.burst = state.burst * 0.5 + 1.0  # decaying burst counter
    sharing = min((device_share_count - 1) / 3.0, 1.0)
    beh = NEUTRAL_BEHAVIOR_ANOMALY if behavior_anomaly is None else behavior_anomaly

    return [
        (0.4 if cold else (0.0 if e.device_id in state.devices else 1.0)),
        (0.5 if cold else (0.0 if e.geo in state.geos else 1.0)),
        _hour_deviation(state, e.hour_of_day),
        _amount_zscore(state, e.amount),
        beh,
        EVENT_CRITICALITY[e.event_type],
        CHANNEL_RISK[e.channel],
        1.0 if e.is_new_beneficiary else 0.0,
        1.0 if e.recovery_contact_changed else 0.0,
        min(state.burst / 15.0, 1.0),
        sharing,
    ]


def commit_features(state: IdentityState, e: IdentityEvent) -> None:
    """Fold the event into the per-identity baseline. Called ONLY after the
    event was allowed (or a step-up verified) — otherwise an attacker could
    poison the profile (anti-poisoning)."""
    state.devices.add(e.device_id)
    state.geos.add(e.geo)
    state.hours.append(e.hour_of_day)
    if e.amount is not None:
        state.amounts.append(e.amount)
    state.event_count += 1
