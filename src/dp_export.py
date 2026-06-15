#!/usr/bin/env python3
"""Differentially-private feature-aggregation export (closes).

    python src/dp_export.py            # writes results/{dp_feature_aggregates,privacy_budget}.json

SECURITY: ``privacy.dp_noise`` was never called. This is the REAL
export path used to publish per-cohort feature statistics for model retraining /
monitoring. It:

  * replays real RBA logins into the 11-dim serving feature schema;
  * reduces each IDENTITY to ONE clipped mean vector in [0,1]^11 (bounded
    contribution → bounded sensitivity);
  * groups identities into disjoint COHORTS by geo (country);
  * releases each cohort's mean via the GAUSSIAN MECHANISM (noise σ = m·Δ₂,
    Δ₂ = √F from the [0,1] clip);
  * accounts the spent ε with a real RDP accountant (``dp_accounting``).

Because cohorts are a disjoint partition of users, PARALLEL composition applies:
total ε = the single-cohort ε (≈1.0 at noise_multiplier 4.0, δ=1e-5), NOT the
sum across cohorts. There is NO un-noised export path - ``noise_multiplier<=0``
raises.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.features import FEATURE_NAMES, commit_features, compute_features  # noqa: E402
from app.state_store import InMemoryStateStore  # noqa: E402

sys.path.insert(0, str(ROOT / "src"))
from export_models import SAMPLE, _row_to_event  # noqa: E402

F = len(FEATURE_NAMES)
DELTA = 1e-5
DEFAULT_NOISE_MULTIPLIER = 4.0       # → ε ≈ 1.013 at δ=1e-5 (verified by accountant)
MIN_COHORT = 25                      # suppress tiny cohorts (re-id risk)
COUNT_EPSILON = 0.1                  # DP budget for the cohort-SIZE query
COUNT_BUFFER = 10                    # stability margin above MIN_COHORT


def _epsilon(noise_multiplier: float, delta: float = DELTA) -> float:
    from dp_accounting import dp_event, rdp

    orders = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))
    acc = rdp.RdpAccountant(orders)
    acc.compose(dp_event.GaussianDpEvent(noise_multiplier))
    return float(acc.get_epsilon(delta))


def _identity_means(sample_path: Path, limit: int | None):
    """Replay genuine RBA logins → one clipped mean feature vector per identity,
    plus that identity's cohort (country)."""
    df = pd.read_csv(sample_path)
    df.columns = [c.strip() for c in df.columns]
    ts = pd.to_datetime(df["Login Timestamp"], errors="coerce")
    df = df.assign(_ts=ts).dropna(subset=["_ts"])
    df["_succ"] = df["Login Successful"].astype(str).str.lower().eq("true")
    df["_ato"] = df["Is Account Takeover"].astype(str).str.lower().eq("true")
    df["_hour"] = df["_ts"].dt.hour.fillna(0).astype(int)
    g = df[df["_succ"] & ~df["_ato"]].sort_values(["User ID", "_ts"])
    if limit:
        g = g.head(limit)
    g = g.rename(columns={
        "User ID": "user", "Device Type": "device_type",
        "Browser Name and Version": "browser", "OS Name and Version": "os",
        "Country": "country", "_hour": "hod"})

    store = InMemoryStateStore()
    sums: dict[str, np.ndarray] = {}
    counts: dict[str, int] = {}
    cohort: dict[str, str] = {}
    for r in g[["user", "device_type", "browser", "os", "country", "hod"]].itertuples(index=False):
        e = _row_to_event(r.user, r.device_type, r.browser, r.os, r.country, r.hod)
        with store.lock(e.identity_id):
            state = store.load(e.identity_id)
            dev = store.device_add(e.device_id, e.identity_id)
            vec = np.clip(compute_features(e, state, dev, behavior_anomaly=None), 0.0, 1.0)
            commit_features(state, e)
            store.commit(e.identity_id, state, state.version)
        iid = e.identity_id
        sums[iid] = sums.get(iid, np.zeros(F)) + vec
        counts[iid] = counts.get(iid, 0) + 1
        cohort[iid] = str(r.country)

    means = {iid: sums[iid] / counts[iid] for iid in sums}  # clipped → [0,1]^F
    return means, cohort


def compute_cohort_means(sample_path: Path, limit: int | None = None,
                         noise_multiplier: float = DEFAULT_NOISE_MULTIPLIER,
                         seed: int = 42):
    if noise_multiplier <= 0:
        raise RuntimeError(
            "SECURITY: feature-aggregation export blocked - the DP mechanism is "
            "mandatory (noise_multiplier must be > 0)."
       )
    means, cohort = _identity_means(sample_path, limit)
    # group identity means by cohort
    by_cohort: dict[str, list[np.ndarray]] = {}
    for iid, m in means.items():
        by_cohort.setdefault(cohort[iid], []).append(m)

    sensitivity_l2 = math.sqrt(F)                 # one clipped [0,1]^F vector / identity
    sigma = noise_multiplier * sensitivity_l2     # Gaussian noise on the SUM
    rng = np.random.default_rng(seed)
    raw, noised = {}, {}
    for c, vecs in by_cohort.items():
        n = len(vecs)
        # SECURITY: the release/suppress DECISION must be DP too, else
        # the published cohort SET leaks boundary-user membership. Suppress on a
        # NOISY count (Laplace, sensitivity 1) against MIN_COHORT + buffer.
        noisy_n = n + rng.laplace(0.0, 1.0 / COUNT_EPSILON)
        if noisy_n < MIN_COHORT + COUNT_BUFFER:
            continue
        s = np.sum(vecs, axis=0)
        raw[c] = (s / n).tolist()
        noisy_sum = s + rng.normal(0.0, sigma, size=F)
        noised[c] = np.clip(noisy_sum / n, 0.0, 1.0).tolist()
    return raw, noised, sensitivity_l2


def export_dp_aggregates(out_dir: Path | None = None, limit: int | None = None,
                         noise_multiplier: float = DEFAULT_NOISE_MULTIPLIER,
                         seed: int = 42) -> Path:
    out_dir = Path(out_dir) if out_dir else (ROOT / "results")
    out_dir.mkdir(parents=True, exist_ok=True)
    raw, noised, sens = compute_cohort_means(SAMPLE, limit=limit,
                                             noise_multiplier=noise_multiplier, seed=seed)
    mean_epsilon = _epsilon(noise_multiplier)
    total_epsilon = mean_epsilon + COUNT_EPSILON  # sequential: mean release + count query
    (out_dir / "dp_feature_aggregates.json").write_text(json.dumps({
        "feature_names": FEATURE_NAMES,
        "cohorts": sorted(noised.keys()),
        "noised_cohort_means": noised,
    }, indent=2))
    (out_dir / "privacy_budget.json").write_text(json.dumps({
        "mechanism": "Gaussian (per-cohort vector mean) + Laplace noisy-count suppression",
        "epsilon": round(total_epsilon, 4),
        "epsilon_breakdown": {
            "mean_release_gaussian": round(mean_epsilon, 4),
            "cohort_count_laplace": COUNT_EPSILON,
            "composition": "sequential (mean ⊕ count); each parallel over disjoint cohorts",
        },
        "delta": DELTA,
        "noise_multiplier": noise_multiplier,
        "sensitivity_l2": round(sens, 4),
        "clipping": "per-feature [0,1]; one mean vector per identity",
        "accountant": "dp_accounting.rdp.RdpAccountant (Gaussian) + Laplace mechanism (count)",
        "cohort_suppression": f"DP noisy count (Laplace) >= {MIN_COHORT + COUNT_BUFFER}",
        "cohorts_released": len(noised),
    }, indent=2))
    print(f"DP export: ε_total={total_epsilon:.4f} "
          f"(mean {mean_epsilon:.4f} + count {COUNT_EPSILON}, δ={DELTA}) over "
          f"{len(noised)} cohorts -> {out_dir}/privacy_budget.json")
    return out_dir


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--noise-multiplier", type=float, default=DEFAULT_NOISE_MULTIPLIER)
    ap.add_argument("--out", default=str(ROOT / "results"))
    a = ap.parse_args()
    export_dp_aggregates(out_dir=Path(a.out), limit=a.limit,
                         noise_multiplier=a.noise_multiplier)
