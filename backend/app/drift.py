"""Drift detection on per-identity risk — catches low-and-slow poisoning (KS9).

SECURITY: KS9 — the original anti-poisoning ("commit only on ALLOW") still
let a patient attacker drift a profile a little each allowed session.

Two complementary detectors (the engine fires on EITHER):

1. ``detect(window)`` — a fast sliding-window mean-shift (early vs late half),
   corroborated by a KS test. Catches a *visible transition* quickly.

2. ``step(ewma, cusum, risk)`` — a CUSUM over the excess of each event's risk
   above a SLOW per-identity EWMA baseline. This is what closes the two
   evasions a sliding window misses (SECURITY: R3):
     - an arbitrarily-slow monotone ramp (slope below the window's shift
       threshold) — CUSUM accumulates the persistent small excess and fires;
     - a high plateau reached after a long benign history — the EWMA remembers
       the long-run norm, so the elevation is detected even after the climb
       ages out of any fixed window.
   A verified step-up resets the CUSUM (see risk_engine.apply_verified_step_up).
"""
from __future__ import annotations

from statistics import mean


class DriftDetector:
    def __init__(self, window: int = 12, min_samples: int = 8,
                 min_shift: float = 0.08, alpha: float = 0.05,
                 ewma_alpha: float = 0.03, cusum_slack: float = 0.015,
                 cusum_threshold: float = 0.4):
        self.window = window
        self.min_samples = min_samples
        self.min_shift = min_shift
        self.alpha = alpha
        self.ewma_alpha = ewma_alpha
        self.cusum_slack = cusum_slack
        self.cusum_threshold = cusum_threshold

    # --- fast sliding-window mean-shift -------------------------------- #
    def detect(self, risk_window: list[float]) -> bool:
        if len(risk_window) < self.min_samples:
            return False
        recent = risk_window[-self.window:]
        half = len(recent) // 2
        early, late = recent[:half], recent[half:]
        if not early or not late:
            return False
        if mean(late) - mean(early) < self.min_shift:
            return False
        try:
            from scipy import stats

            _, p = stats.ks_2samp(early, late)
            return bool(p < self.alpha)
        except Exception:  # pragma: no cover - scipy absent → mean-shift only
            return True

    # --- persistent-baseline CUSUM (catches arbitrarily-slow drift) ----- #
    def step(self, ewma: float, cusum: float, risk: float):
        """Advance the per-identity EWMA + CUSUM. Returns (ewma, cusum, fired)."""
        baseline = ewma if ewma > 0 else risk
        new_cusum = max(0.0, cusum + (risk - baseline - self.cusum_slack))
        new_ewma = risk if ewma == 0.0 else ewma + self.ewma_alpha * (risk - ewma)
        return new_ewma, new_cusum, new_cusum > self.cusum_threshold
