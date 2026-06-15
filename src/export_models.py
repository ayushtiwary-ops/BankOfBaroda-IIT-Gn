#!/usr/bin/env python3
"""Export the live serving model from REAL data (closes reproducibility).

    python src/export_models.py            # full RBA sample → results/models/
    python src/export_models.py --limit 5000

The live engine scores an 11-dim *behavioural* serving vector
(``app.features.FEATURE_NAMES``), which is a DIFFERENT feature space from the
offline dataset-native models in ``train.py``. To wire real data into serving
without feature drift, we replay real RBA (Wiefling) logins through the EXACT
serving feature code (``compute_features`` / ``commit_features``) and fit the
serving anomaly detector (StandardScaler + IsolationForest) on the resulting
distribution of *genuine* logins. The baseline of "normal" the live engine
trusts is therefore learned from real logins, not ``np.random``.

We do NOT touch the offline metrics pipeline (train.py / evaluate.py); we only
read its outputs (the RBA metric for the model card) and the committed RBA
sample.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))  # share the serving feature code

from app.features import FEATURE_NAMES, commit_features, compute_features  # noqa: E402
from app.model_loader import save_artifact  # noqa: E402
from app.schemas import Channel, EventType, IdentityEvent  # noqa: E402
from app.state_store import InMemoryStateStore  # noqa: E402

SAMPLE = ROOT / "data" / "samples" / "rba_sample.csv"
MODELS = ROOT / "results" / "models"
CONTAMINATION = 0.03


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "no-git"


def _rba_metric() -> dict:
    try:
        m = json.loads((ROOT / "results" / "rba_noip" / "metrics.json").read_text())
        return {k: m.get(k) for k in ("roc_auc", "pr_auc", "operating_point")}
    except Exception:
        return {}


def _fam(value) -> str:
    parts = str(value).split()
    return parts[0] if parts else "unknown"


def _row_to_event(user, device_type, browser, os_name, country, hour) -> IdentityEvent:
    # Coarse device fingerprint proxy from the RBA device/browser/os triple.
    device_id = f"{device_type}|{_fam(browser)}|{_fam(os_name)}"
    return IdentityEvent(
        identity_id=str(user),
        event_type=EventType.LOGIN,
        channel=Channel.INTERNET_BANKING,
        device_id=device_id,
        geo=str(country),
        hour_of_day=int(hour),
   )


def build_serving_vectors(sample_path: Path, limit: int | None = None):
    """Replay genuine RBA logins through the serving feature schema."""
    df = pd.read_csv(sample_path)
    df.columns = [c.strip() for c in df.columns]
    ts = pd.to_datetime(df["Login Timestamp"], errors="coerce")
    df = df.assign(_ts=ts).dropna(subset=["_ts"])
    df["_succ"] = df["Login Successful"].astype(str).str.lower().eq("true")
    df["_ato"] = df["Is Account Takeover"].astype(str).str.lower().eq("true")
    df["_hour"] = df["_ts"].dt.hour.fillna(0).astype(int)
    # the baseline of "normal" = genuine successful logins (no ATO)
    g = df[df["_succ"] & ~df["_ato"]].sort_values(["User ID", "_ts"])
    if limit:
        g = g.head(limit)
    g = g.rename(columns={
        "User ID": "user", "Device Type": "device_type",
        "Browser Name and Version": "browser", "OS Name and Version": "os",
        "Country": "country", "_hour": "hod"})

    store = InMemoryStateStore()
    vectors: list[list[float]] = []
    for r in g[["user", "device_type", "browser", "os", "country", "hod"]].itertuples(index=False):
        e = _row_to_event(r.user, r.device_type, r.browser, r.os, r.country, r.hod)
        with store.lock(e.identity_id):
            state = store.load(e.identity_id)
            dev = store.device_add(e.device_id, e.identity_id)
            # behaviour is MISSING in RBA → cold-start neutral, like serving
            vectors.append(compute_features(e, state, dev, behavior_anomaly=None))
            commit_features(state, e)  # genuine login → fold into baseline
            store.commit(e.identity_id, state, state.version)
    return np.asarray(vectors, dtype=float)


def export(out_dir: Path | None = None, limit: int | None = None) -> Path:
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler

    out_dir = Path(out_dir) if out_dir else MODELS
    t0 = time.time()
    x = build_serving_vectors(SAMPLE, limit=limit)
    scaler = StandardScaler().fit(x)
    model = IsolationForest(n_estimators=200, contamination=CONTAMINATION,
                            random_state=42).fit(scaler.transform(x))
    card = {
        "name": "serving_anomaly",
        "dataset": "RBA / Wiefling (Zenodo 6782156) - real logins, behavioural serving schema",
        "dataset_url": "https://zenodo.org/records/6782156",
        "detector": "IsolationForest(n=200) over StandardScaler(11 serving features)",
        "metric": _rba_metric(),
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_sha": _git_sha(),
        "n_train": int(x.shape[0]),
        "contamination": CONTAMINATION,
        "provenance": "rba_wiefling",
        "note": "Replayed through app.features.compute_features so training and "
                "serving share one feature definition (no drift).",
    }
    save_artifact(out_dir, model=model, scaler=scaler,
                  feature_names=FEATURE_NAMES, card=card)
    print(f"exported serving_anomaly artifact: n_train={x.shape[0]:,} "
          f"-> {out_dir} ({time.time()-t0:.1f}s)")
    return out_dir


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="cap genuine logins replayed (default: all)")
    ap.add_argument("--out", default=str(MODELS))
    a = ap.parse_args()
    export(out_dir=Path(a.out), limit=a.limit)
