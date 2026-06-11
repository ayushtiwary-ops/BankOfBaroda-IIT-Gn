#!/usr/bin/env python3
"""Cold-start friction metric — population prior reduces new-user step-ups (KS10).

    python src/coldstart_metric.py [--limit N]

Replays each RBA user's FIRST genuine login through the engine WITH the
cold-start population prior vs WITHOUT it, and reports the new-user step-up rate
before/after. Genuine new users (new device + new geo at onboarding) are exactly
the population the old engine friction-bombs; the prior dampens that while
keeping attacker detection (drift/impossible-travel/attestation) intact.

Output: results/coldstart/report.json
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "src"))

from app.config import Settings  # noqa: E402
from app.features import FEATURE_NAMES  # noqa: E402
from app.model_loader import load_serving_model  # noqa: E402
from app.risk_engine import TrustEngine  # noqa: E402
from app.schemas import Decision  # noqa: E402

from export_models import SAMPLE, _row_to_event  # noqa: E402

OUT = ROOT / "results" / "coldstart"


def _real_model():
    prod = Settings(mode="prod", edge_secret=b"x", audit_signing_key=b"x",
                    stepup_pubkey="", attest_pubkey="", behavior_pubkey="",
                    api_keys={}, cors_origins=["x"],
                    model_dir=ROOT / "results" / "models", redis_url=None)
    return load_serving_model(prod, FEATURE_NAMES)


def _first_logins(limit: int | None):
    df = pd.read_csv(SAMPLE)
    df.columns = [c.strip() for c in df.columns]
    ts = pd.to_datetime(df["Login Timestamp"], errors="coerce")
    df = df.assign(_ts=ts).dropna(subset=["_ts"])
    df["_succ"] = df["Login Successful"].astype(str).str.lower().eq("true")
    df["_ato"] = df["Is Account Takeover"].astype(str).str.lower().eq("true")
    df["_hour"] = df["_ts"].dt.hour.fillna(0).astype(int)
    g = df[df["_succ"] & ~df["_ato"]].sort_values(["User ID", "_ts"])
    first = g.groupby("User ID", sort=False).head(1)   # each user's first genuine login
    if limit:
        first = first.head(limit)
    return first.rename(columns={
        "User ID": "user", "Device Type": "device_type",
        "Browser Name and Version": "browser", "OS Name and Version": "os",
        "Country": "country", "_hour": "hod"})


def _stepup_rate(engine, rows) -> float:
    n = stepped = 0
    for r in rows.itertuples(index=False):
        e = _row_to_event(r.user, r.device_type, r.browser, r.os, r.country, r.hod)
        if engine.assess(e).decision != Decision.ALLOW:
            stepped += 1
        n += 1
    return stepped / max(n, 1)


def main(limit: int | None = None):
    s = Settings.from_env({"PRAMAAN_MODE": "demo_synthetic"})
    real = _real_model()
    rows = _first_logins(limit)

    # naive baseline: a synthetic/np.random-style model with no cold-start prior
    # (the original engine) friction-bombs genuine new users.
    naive = _stepup_rate(TrustEngine(settings=s, behavior_resolver=None,
                                     cold_start_prior=False), rows)
    # real model (KS3), no prior — training on real first-logins already helps.
    real_noprior = _stepup_rate(TrustEngine(settings=s, serving_model=real,
                                            behavior_resolver=None,
                                            cold_start_prior=False), rows)
    # shipped config: real model + cold-start population prior.
    shipped = _stepup_rate(TrustEngine(settings=s, serving_model=real,
                                       behavior_resolver=None,
                                       cold_start_prior=True), rows)

    OUT.mkdir(parents=True, exist_ok=True)
    report = {
        "metric": "new-user (first genuine login) step-up rate — lower is less friction",
        "n_new_users": int(len(rows)),
        "dataset": "RBA / Wiefling (genuine successful first logins)",
        "naive_baseline_synthetic_no_prior": round(naive, 4),
        "real_model_no_prior": round(real_noprior, 4),
        "shipped_real_model_with_prior": round(shipped, 4),
        "before_after": {
            "before": round(naive, 4), "after": round(shipped, 4),
            "absolute_reduction": round(naive - shipped, 4),
            "relative_reduction_pct": round(100 * (naive - shipped) / naive, 1)
            if naive else 0.0,
        },
        "finding": ("Training the live model on real first-logins is what "
                    "actually fixes cold-start: genuine new users drop from "
                    f"{naive:.0%} step-up (synthetic baseline) to {shipped:.0%} "
                    "(real model + population prior)."),
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2))
    print(f"new-user step-up rate: {naive:.0%} (naive) -> {real_noprior:.0%} "
          f"(real model) -> {shipped:.0%} (real+prior)  -> {OUT}/report.json")
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    main(a.limit)
