# PRAMAAN - The Numbers (real data, reproducible)

> Every figure below is produced by `python src/download_data.py → train.py → evaluate.py`
> on a **public, cited dataset** (see `DATA_SOURCES.md`). Seeds fixed, temporal splits,
> leakage guarded. Curves: `results/<run>/detection_vs_stepup.png`.

## The sentence

> **On 31.3 million real logins, PRAMAAN catches 93% of account-takeover attempts while
> stepping up only 2% of genuine users - and on mobile-money fraud it catches 100% of mule
> cash-outs at a 1% step-up rate, versus the bank's own rule at 0.5%. Here are the curves;
> regenerate every number with three commands.**

## Per-detection scorecard

| # | Detection | Dataset (real) | Headline result | Honest caveat |
|---|---|---|---|---|
| 4 | Account-takeover / risk-based auth | RBA / Wiefling - 31.3M logins, full 6.79M-login held-out test, 56 test ATO | **93% of ATO caught @ 2% step-up, 87.5% @ 1%** (ROC 0.993, behavioural, no IP feed) | small positive class (140 total ATO) → recall-at-budget with bootstrap CI, not precision |
| 3 | Mule / money-flow fraud | PaySim - 6.36M txns, 1,600 test fraud | **100% caught @ 1% step-up** (ROC 1.00, PR-AUC 1.00) vs built-in rule **0.5%** | PaySim is cleanly separable; the realistic challenge sets are RBA/IEEE/CERT |
| 1+2 | New-device / device trust | IEEE-CIS - 590K txns, 4,064 test fraud | on device-bearing flows **PR-AUC 0.686, 47% @ 1% step-up**; device columns **+2.4 pts** recall@1% | population lift modest - only 24% of txns carry device data |
| 1 | Behavioral biometrics | CMU keystroke - 51 users | **mean EER 0.0955** - matches Killourhy & Maxion (0.096) | per-user enrollment; password-specific benchmark |
| 5 | Privileged / insider misuse | CERT r4.2 - 1,000 users, 70 insiders | **100% of IP-theft insiders @ 10 alerts/day; +80% of sabotage @ 25/day** | data-theft-on-departure (scen 1) needs the web-proxy feed we excluded (14.5 GB) |

## What makes each number trustworthy (not theater)

- **Temporal splits** everywhere time exists (PaySim `step`, IEEE `TransactionDT`, RBA timestamp,
  CERT day) - the model never trains on the future.
- **Leakage guarded & ablated:** PaySim reported full vs balance-free (behavioral-only ROC drops
  1.00→0.967, shown honestly); IEEE reported with/without device columns; RBA reported with/without
  the IP-reputation feed; we explicitly rejected the HF mirror's target-leakage features.
- **The blunt baselines lose:** PaySim's shipped `isFlaggedFraud` rule = 0.5% recall; a raw
  IP-reputation blocklist on RBA = 0% recall at a 2% budget. The model is the lift.
- **Full-population RBA eval:** the headline RBA numbers come from `train_rba_full.py`, which streams
  the full 31.3M-login dataset and scores the entire 6.79M-login held-out test (all negatives, no
  subsampling, so no correction is needed); bootstrap 95% CIs are reported on every figure. The
  inverse-rate sampling correction is still implemented for any subsampled run (see `eval_full.py`).
- **Unsupervised where it must be:** CERT insider detection uses per-user robust-deviation features
  + IsolationForest (no labels at train time) - labels touch only the scoring.

## Two findings worth saying out loud (mastery signals)

1. **Behavioral risk beats the blocklist.** On the full RBA test, our behavioral model (no IP feed)
   catches 93% of ATO @ 2% step-up (ROC 0.993); the IP-reputation flag *alone* catches 0% at that
   budget (ROC 0.734), and adding the IP feed to the model slightly lowers ROC (0.962). Risk scoring,
   not IP blocklists, is what protects account recovery.
2. **Device data is where fraud hides.** IEEE-CIS fraud is 7.85% on device/identity-bearing flows
   vs 2.09% without (3.7×); that is exactly where our model concentrates its detection.

## Regenerate everything

```bash
python src/download_data.py --all # fetch + SHA-256 verify (DATA_SOURCES.md)
python src/train.py paysim|ieee_cis|rba|cmu_keystroke|cert_insider
python src/train_rba_full.py # RBA on the full 31.3M-login dataset (headline RBA numbers)
python src/evaluate.py all # metrics.json + curves under results/
python src/eval_full.py all # full metric suite (PR-AUC, pAUC, CIs) under results/evaluation/
python src/eval_full.py rba_full # full-data RBA metrics (results/evaluation/rba_full_*)
```
See `docs/EVALUATION.md` for the complete metric suite, the model bake-off, calibration, and the
old-versus-new comparison with confidence intervals.
Outputs: `results/<run>/metrics.json`, `detection_vs_stepup.png`, `pr_curve.png`, `roc_curve.png`,
`results/*/ablation_stepup.png`, `results/cmu_keystroke/eer_distribution.png`,
`results/cert_budget/detection_vs_budget.png`.
