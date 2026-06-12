#!/usr/bin/env python3
"""Closed-loop load test against a running PRAMAAN service — measured p50/p95/p99.

    # terminal 1:
    PRAMAAN_MODE=demo_synthetic uvicorn app.main:app --app-dir backend --port 8099
    # terminal 2:
    python scripts/loadtest.py --url http://127.0.0.1:8099 --concurrency 32 --requests 4000

Produces honest, measured numbers (no asserting the p99 claim):
  results/load/latency_histogram.png
  results/load/load_report.json   {p50,p95,p99,throughput,hardware,mode}
"""
import argparse
import asyncio
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import httpx
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "load"
HEADERS = {"X-API-Key": "demo-key", "Content-Type": "application/json"}


def _event(i: int) -> dict:
    return {"identity_id": f"load_{i % 500}", "event_type": "login",
            "channel": "mobile_app", "device_id": f"dev_{i % 500}",
            "geo": "IN-GJ", "hour_of_day": 12}


async def _run(url: str, concurrency: int, requests: int):
    sem = asyncio.Semaphore(concurrency)
    latencies: list[float] = []
    errors = 0
    async with httpx.AsyncClient(timeout=10.0) as client:
        # warm up (model first-touch, JIT caches)
        for i in range(20):
            await client.post(f"{url}/v1/events", headers=HEADERS, json=_event(i))

        async def one(i: int):
            nonlocal errors
            async with sem:
                t = time.perf_counter()
                try:
                    r = await client.post(f"{url}/v1/events", headers=HEADERS, json=_event(i))
                    if r.status_code != 200:
                        errors += 1
                except Exception:
                    errors += 1
                    return
                latencies.append((time.perf_counter() - t) * 1000.0)

        t0 = time.perf_counter()
        await asyncio.gather(*(one(i) for i in range(requests)))
        wall = time.perf_counter() - t0
    return latencies, errors, wall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8099")
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--requests", type=int, default=4000)
    a = ap.parse_args()

    latencies, errors, wall = asyncio.run(_run(a.url, a.concurrency, a.requests))
    if not latencies:
        print("no successful requests — is the server up?", file=sys.stderr)
        sys.exit(1)
    latencies.sort()

    def pct(p):
        return round(latencies[min(len(latencies) - 1, int(p / 100 * len(latencies)))], 2)

    report = {
        "samples": len(latencies),
        "errors": errors,
        "concurrency": a.concurrency,
        "throughput_rps": round(len(latencies) / wall, 1),
        "latency_ms": {"p50": pct(50), "p95": pct(95), "p99": pct(99),
                       "max": round(latencies[-1], 2),
                       "mean": round(statistics.mean(latencies), 2)},
        "claim_p99_under_50ms": pct(99) < 50.0,
        "hardware": f"{platform.machine()} / {platform.system()} {platform.release()}",
        "mode": "single-node, InMemory StateStore (no Redis server in this env)",
        "note": ("Measured over a real socket against a live uvicorn worker. "
                 "Redis/Kafka multi-pod scale-out is the production topology; this "
                 "is the single-node scoring latency, reported as-measured."),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "load_report.json").write_text(json.dumps(report, indent=2))

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(latencies, bins=60, color="#2dd4a7", edgecolor="#0e3a2d")
    for p, c in [(50, "#888"), (95, "#f5b14d"), (99, "#f4636e")]:
        ax.axvline(pct(p), ls="--", color=c, label=f"p{p} = {pct(p)} ms")
    ax.axvline(50, ls=":", color="#333", label="50 ms target")
    ax.set_xlabel("request latency (ms)")
    ax.set_ylabel("count")
    ax.set_title(f"PRAMAAN /v1/events latency — {len(latencies)} reqs @ "
                 f"concurrency {a.concurrency} ({report['throughput_rps']} rps)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "latency_histogram.png", dpi=110)

    lm = report["latency_ms"]
    print(f"p50={lm['p50']}ms p95={lm['p95']}ms p99={lm['p99']}ms "
          f"({report['throughput_rps']} rps, {errors} errors) -> {OUT}/")


if __name__ == "__main__":
    main()
