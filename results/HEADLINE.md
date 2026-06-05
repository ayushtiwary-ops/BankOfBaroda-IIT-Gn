# PRAMAAN — The Numbers (real data, reproducible)

> Every figure below is produced by `python src/download_data.py → train.py → evaluate.py`
> on a **public, cited dataset** (see `DATA_SOURCES.md`). Seeds fixed, temporal splits,
> leakage guarded. Curves: `results/<run>/detection_vs_stepup.png`.

## The sentence

> **On 31.3 million real logins, PRAMAAN catches 93% of account-takeover attempts while
> stepping up only 2% of genuine users — and on mobile-money fraud it catches 100% of mule
> cash-outs at a 1% step-up rate, versus the bank's own rule at 0.5%. Here are the curves;
> regenerate every number with three commands.**

## Per-detection scorecard

| # | Detection | Dataset (real) | Headline result | Honest caveat |
|---|---|---|---|---|
| 4 | Account-takeover / risk-based auth | RBA / Wiefling — 31.3M logins, 56 test ATO | **93% of ATO caught @ 2% step-up** (ROC 0.965); 70% @ 1% | small positive class (141 total ATO) → recall-at-budget, not precision |
| 3 | Mule / money-flow fraud | PaySim — 6.36M txns, 1,600 test fraud | **100% caught @ 1% step-up** (ROC 1.00, PR-AUC 1.00) vs built-in rule **0.5%** | PaySim is cleanly separable; the realistic challenge sets are RBA/IEEE/CERT |
| 1+2 | New-device / device trust | IEEE-CIS — 590K txns, 4,064 test fraud | on device-bearing flows **PR-AUC 0.686, 47% @ 1% step-up**; device columns **+2.4 pts** recall@1% | population lift modest — only 24% of txns carry device data |
| 1 | Behavioral biometrics | CMU keystroke — 51 users | **mean EER 0.0955** — matches Killourhy & Maxion (0.096) | per-user enrollment; password-specific benchmark |
| 5 | Privileged / insider misuse | CERT r4.2 — 1,000 users, 70 insiders | **100% of IP-theft insiders @ 10 alerts/day; +80% of sabotage @ 25/day** | data-theft-on-departure (scen 1) needs the web-proxy feed we excluded (14.5 GB) |

## What makes each number trustworthy (not theater)

- **Temporal splits** everywhere time exists (PaySim `step`, IEEE `TransactionDT`, RBA timestamp,
  CERT day) — the model never trains on the future.
- **Leakage guarded & ablated:** PaySim reported full vs balance-free (behavioral-only ROC drops
  1.00→0.967, shown honestly); IEEE reported with/without device columns; RBA reported with/without
  the IP-reputation feed; we explicitly rejected the HF mirror's target-leakage features.
- **The blunt baselines lose:** PaySim's shipped `isFlaggedFraud` rule = 0.5% recall; a raw
  IP-reputation blocklist on RBA = 0% recall at a 2% budget. The model is the lift.
- **Sampling-corrected metrics:** RBA negatives were sub-sampled; FPR/recall/ROC are unbiased and
  precision/PR-AUC are corrected by the sampling rate (documented in `evaluate.py`).
- **Unsupervised where it must be:** CERT insider detection uses per-user robust-deviation features
  + IsolationForest (no labels at train time) — labels touch only the scoring.

## Two findings worth saying out loud (mastery signals)

1. **Behavioral risk beats the blocklist.** On RBA, our behavioral model (no IP feed) catches 89%
   of ATO @ 2% step-up; the IP-reputation flag *alone* catches 0% at that budget. Risk scoring,
   not IP blocklists, is what protects account recovery.
2. **Device data is where fraud hides.** IEEE-CIS fraud is 7.85% on device/identity-bearing flows
   vs 2.09% without (3.7×); that is exactly where our model concentrates its detection.

## Regenerate everything

```bash
python src/download_data.py --all          # fetch + SHA-256 verify (DATA_SOURCES.md)
python src/train.py    paysim|ieee_cis|rba|cmu_keystroke|cert_insider
python src/evaluate.py all                  # metrics.json + curves under results/
```
Outputs: `results/<run>/metrics.json`, `detection_vs_stepup.png`, `pr_curve.png`, `roc_curve.png`,
`results/*/ablation_stepup.png`, `results/cmu_keystroke/eer_distribution.png`,
`results/cert_budget/detection_vs_budget.png`.
