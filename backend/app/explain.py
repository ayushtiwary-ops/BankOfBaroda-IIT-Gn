"""Explainability — SHAP values + a counterfactual, for the SOC/audit plane.

DPDP §13 (grievance redress) + RBI explainability: every decision must be
explainable to an analyst and, ultimately, to the customer.

The deterministic risk is a LINEAR additive scorer (Σ wᵢfᵢ), so the exact
Shapley value of feature i is wᵢ·(fᵢ − baselineᵢ) relative to a benign-login
baseline — this is true SHAP for an additive model, computed exactly and
cheaply per decision. (The IsolationForest's SHAP is heavier; it is produced
offline by ``src/explain_shap.py`` into results/explainability/.)

A single counterfactual ("would have been ALLOWED if amount < ₹X") names the
one change that would most reduce risk — the most actionable thing for the SOC.
"""
from __future__ import annotations

from .features import FEATURE_NAMES

# Benign-login reference point (the SHAP baseline): no anomalies, owner-like
# behaviour, a low-criticality login on a low-risk channel.
BASELINE = [0.0, 0.0, 0.0, 0.0, 0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.0]

_CF_TEMPLATE = {
    "new_device": "the device were already recognised for this identity",
    "new_geo": "the login came from a known location",
    "hour_deviation": "the activity were within the usual hours",
    "amount_zscore": "the amount were closer to this identity's normal range",
    "behavior_anomaly": "the behavioural-biometric match were stronger",
    "event_criticality": "the action were lower-risk",
    "channel_risk": "the channel were lower-risk",
    "new_beneficiary": "the payee were already known",
    "recovery_change": "the recovery contact were not being changed",
    "velocity": "the activity were less rapid",
    "device_sharing": "the device were not shared across identities",
}


def shap_values(vec, weights, baseline=BASELINE):
    """Exact additive Shapley contributions, descending by magnitude."""
    phis = [(round(w * (f - b), 4), name)
            for w, f, b, name in zip(weights, vec, baseline, FEATURE_NAMES)]
    return sorted(phis, key=lambda t: abs(t[0]), reverse=True)


def counterfactual(vec, weights, event, baseline=BASELINE, amount_ref=None):
    """The single highest-leverage change that would most reduce risk.

    ``amount_ref`` is the identity's own typical amount (mean of history); when
    available it anchors a concrete, MEANINGFUL numeric target instead of the
    earlier vacuous ₹0 (R3 fix). Returns a sentence for the SOC/audit plane."""
    phis = shap_values(vec, weights, baseline)
    top_phi, top_name = phis[0]
    if top_phi <= 0:
        return "No single factor dominated; behaviour was broadly consistent."
    if top_name == "amount_zscore" and getattr(event, "amount", None):
        if amount_ref and amount_ref > 0:
            return (f"Would have scored materially lower if the amount were closer to "
                    f"≈ ₹{int(amount_ref):,} (this identity's typical range).")
        return ("Would have scored lower if the amount were closer to this "
                "identity's normal range.")
    return f"Would have scored lower if {_CF_TEMPLATE.get(top_name, 'this factor were normal')}."
