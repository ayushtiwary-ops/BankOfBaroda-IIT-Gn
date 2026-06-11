#!/usr/bin/env python3
"""Low-and-slow poisoning attack — OLD engine poisoned, NEW engine catches it.

    python simulator/attack_low_and_slow.py

Demonstrates KS9. A hijacked session on the victim's OWN device sends
transactions with slowly-rising amounts. Each step is small enough that, on the
OLD engine, it is ALLOWED and committed — nudging the amount baseline up — so the
NEXT (bigger) amount still looks normal. The baseline creeps until a large
cash-out reads as routine (takeover succeeds).

The NEW engine adds drift detection on the per-identity risk window: the
sustained upward trend trips a secondary review (STEP_UP) BEFORE the baseline is
poisoned — and because a step-up is not an ALLOW, the profile stops creeping.

Outputs:
  results/adversarial/low_and_slow.png   — trust + decision trajectory, OLD vs NEW
  results/adversarial/low_and_slow.json  — the raw trajectories (reproducible)
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.attestation import (  # noqa: E402
    BehaviorProvider,
    BehaviorResolver,
    DeviceAttestationProvider,
)
from app.config import Settings  # noqa: E402
from app.risk_engine import TrustEngine  # noqa: E402
from app.schemas import Channel, Decision, EventType, IdentityEvent  # noqa: E402
from app.verifier import Ed25519Signer, Ed25519Verifier  # noqa: E402

OUT = ROOT / "results" / "adversarial"
N_CREEP = 20
OWNER_DEVICE = "dev_owner_phone"


def _providers():
    a, b = Ed25519Signer.generate(), Ed25519Signer.generate()
    resolver = BehaviorResolver(
        attest_verifier=Ed25519Verifier.from_b64(a.public_key_b64),
        behavior_verifier=Ed25519Verifier.from_b64(b.public_key_b64))
    return DeviceAttestationProvider(a), BehaviorProvider(b), resolver


def _attested(ap, bp, device, identity, sim):
    return dict(device_attestation=ap.issue(device_id=device),
                behavior_assertion=bp.issue(device_id=device, identity_id=identity,
                                            similarity=sim))


def run(engine, ap, bp, identity):
    # warm the victim baseline: ~10 attested owner transactions around ₹2,000
    for i in range(10):
        engine.assess(IdentityEvent(
            identity_id=identity, event_type=EventType.TRANSACTION,
            channel=Channel.MOBILE_APP, device_id=OWNER_DEVICE, geo="IN-GJ",
            hour_of_day=12, amount=2000.0 + (i % 3) * 100,
            **_attested(ap, bp, OWNER_DEVICE, identity, 0.95)))
    # hijacked session: same device, NO behaviour assertion (MISSING), creeping amount
    traj = []
    amount = 2500.0
    for i in range(N_CREEP):
        soc = engine.assess(IdentityEvent(
            identity_id=identity, event_type=EventType.TRANSACTION,
            channel=Channel.INTERNET_BANKING, device_id=OWNER_DEVICE, geo="IN-GJ",
            hour_of_day=12, amount=amount, is_new_beneficiary=(i >= N_CREEP - 1)))
        traj.append({"session": i, "amount": round(amount), "trust": soc.trust_score,
                     "event_risk": round(soc.event_risk, 3), "decision": soc.decision.value,
                     "caught": soc.decision != Decision.ALLOW})
        amount *= 1.25  # +25% each session → ~₹230k by the end
    return traj


def main():
    s = Settings.from_env({"PRAMAAN_MODE": "demo_synthetic"})
    ap, bp, resolver = _providers()
    old = TrustEngine.legacy(settings=s, behavior_resolver=resolver)
    new = TrustEngine(settings=s, behavior_resolver=resolver)
    old_traj = run(old, ap, bp, "victim_old")
    new_traj = run(new, ap, bp, "victim_new")

    old_caught = next((t["session"] for t in old_traj if t["caught"]), None)
    new_caught = next((t["session"] for t in new_traj if t["caught"]), None)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "low_and_slow.json").write_text(json.dumps({
        "old_engine": old_traj, "new_engine": new_traj,
        "old_first_caught_session": old_caught,
        "new_first_caught_session": new_caught,
    }, indent=2))

    fig, ax = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    xs = [t["session"] for t in old_traj]
    ax[0].plot(xs, [t["event_risk"] for t in old_traj], "o-", color="#f4636e",
               label="OLD engine (no drift)")
    ax[0].plot(xs, [t["event_risk"] for t in new_traj], "s-", color="#2dd4a7",
               label="NEW engine (drift + capped recovery)")
    ax[0].axhline(0.45, ls="--", color="#888", label="step-up threshold")
    if new_caught is not None:
        ax[0].axvline(new_caught, ls=":", color="#2dd4a7")
        ax[0].annotate("NEW: drift → secondary review",
                       (new_caught, 0.5), color="#2dd4a7")
    ax[0].set_ylabel("event risk")
    ax[0].legend()
    ax[0].set_title("Low-and-slow amount creep — OLD allows through, NEW catches the drift")
    for t in old_traj:
        ax[1].scatter(t["session"], t["amount"],
                      color="#f4636e" if not t["caught"] else "#333", s=18)
    for t in new_traj:
        ax[1].scatter(t["session"], t["amount"],
                      color="#2dd4a7" if t["caught"] else "#bbb", marker="s", s=18)
    ax[1].set_yscale("log")
    ax[1].set_ylabel("txn amount (₹, log)")
    ax[1].set_xlabel("session")
    fig.tight_layout()
    fig.savefig(OUT / "low_and_slow.png", dpi=110)

    print(f"OLD engine first caught at session: {old_caught} "
          f"(None = never — takeover succeeded)")
    print(f"NEW engine first caught at session: {new_caught}")
    print(f"artifacts -> {OUT}/low_and_slow.png + .json")


if __name__ == "__main__":
    main()
