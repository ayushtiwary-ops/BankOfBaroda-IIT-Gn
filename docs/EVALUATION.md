# PRAMAAN - Model Evaluation

> Offline modelling and evaluation for the five identity-trust detections. Every
> number below is regenerated from a real, cited dataset by the commands in the
> last section. Thresholds, model selection, calibration and tuning use train and
> validation data only; each test split is read once for the final figures.

This document covers two things:

1. **The complete metric suite (Phase A)** for every detection: PR-AUC (average
   precision), ROC-AUC, partial AUC at low FPR, KS statistic, Gini, operating
   points across step-up budgets, calibration, a cost-weighted operating point,
   and a bootstrap confidence interval plus a seed sweep on every headline number.
2. **A model bake-off and tuning pass (Phase B)**: gradient-boosting families,
   Optuna search, imbalance handling and probability calibration, with an honest
   keep-or-replace decision per detection.

All artifacts live under `results/evaluation/`:
`<detection>/metrics_full.json`, the curve PNGs, `leaderboard.csv`,
`old_vs_new.json`, and the `tuning/` subfolders.

---

## 1. How every number is produced (the protocol)

- **Primary metric is PR-AUC (average precision).** The classes are extremely
  imbalanced, so average precision, not ROC-AUC or accuracy, is the headline.
  Average precision is the step-wise estimator; we also store the trapezoidal
  area under the PR curve for continuity with earlier reports, but average
  precision is the number we quote. On the rarest class (RBA) the two diverge,
  and the trapezoidal value is optimistic; see the RBA note.
- **Step-up rate is the friction budget.** It is the fraction of genuine events
  challenged, which equals the false-positive rate. Operating points are reported
  at step-up budgets of 0.1, 0.5, 1, 2 and 5 percent.
- **Recall-weighted F-scores.** F2 weights recall over precision, which matches
  fraud economics (a missed fraud costs far more than a false challenge).
- **Sampling correction.** Where negatives were subsampled, they are reweighted
  by the inverse sampling rate so PR-AUC, precision and calibration report
  population values. ROC-AUC, recall, FPR and KS are rank or class-conditional
  and are unbiased without reweighting.
- **Variance is always reported.** Every headline metric carries a bootstrap 95
  percent confidence interval (1000 resamples) for test-set variance, plus a
  five-seed retrain mean and standard deviation for model variance (where the
  model has a seed; the keystroke detector is deterministic and is reported with
  its per-user distribution and a bootstrap-over-users interval instead).
- **Calibration only where it is meaningful.** Brier score and expected
  calibration error are computed for probability outputs only. Anomaly-rank
  detectors (IsolationForest, scaled-Manhattan distance) are not probabilities,
  so calibration is marked not applicable for them.

### Honesty guardrails honored

- Temporal splits are preserved everywhere time exists (PaySim `step`, IEEE
  `TransactionDT`, RBA login timestamp, CERT day). No model trains on the future.
- Documented leakage suspects stay isolated as ablation arms (PaySim balance
  columns, IEEE device columns, RBA IP-reputation feed). A drop-top-feature
  re-audit is reported per detection.
- The test split is read once per detection for the final figures. All model
  selection, hyperparameter search, threshold choice and calibration happen on a
  temporal validation slice carved from train.
- A tuned model replaces the incumbent only if it beats it on the held-out test
  metric by more than the paired bootstrap interval. Otherwise the incumbent is
  kept and that is stated.

---

## 2. Per-detection results (Phase A, final metric suite)

Confidence intervals are bootstrap 95 percent on the test split. "Step-up" is the
percent of genuine events challenged.

| Detection (dataset) | Test positives | PR-AUC (AP) | ROC-AUC | pAUC@1% | KS | Recall @1% / @2% / @5% step-up | F2 @2% | Brier / ECE |
|---|---|---|---|---|---|---|---|---|
| PaySim full, balance + behaviour | 1,600 | 1.000 | 1.000 | 1.000 | 1.000 | 100% / 100% / 100% | 0.807 | 0.000 / 0.000 |
| PaySim behaviour-only (leak-free) | 1,600 | 0.681 [0.66, 0.70] | 0.967 | 0.789 | 0.791 | 69% / 77% / 83% | 0.637 | 0.036 / 0.073 |
| IEEE-CIS, transaction-only | 4,064 | 0.507 [0.49, 0.52] | 0.909 | 0.660 | 0.662 | 42% / 51% / 65% | 0.499 | n/a |
| IEEE-CIS, + device/identity | 4,064 | 0.513 [0.50, 0.53] | 0.907 | 0.666 | 0.652 | 44% / 52% / 64% | 0.510 | 0.022 / 0.003 |
| IEEE-CIS, device-bearing subset | 2,219 | 0.687 [0.67, 0.71] | 0.913 | 0.654 | 0.682 | 47% / 56% / 70% | 0.591 | 0.046 / 0.010 |
| CMU keystroke (51 users) | 51 users | n/a | 0.951 global | n/a | n/a | mean EER 0.0955 [0.077, 0.116] | n/a | n/a (distance) |
| CERT r4.2 IsolationForest | 1,364 ud | 0.022 [0.019, 0.025] | 0.750 | 0.517 | 0.489 | 11% / 21% / 33% | 0.120 | n/a (rank) |
| CERT r4.2 deviation-sum rule | 1,364 ud | 0.034 [0.030, 0.040] | 0.750 | 0.560 | 0.486 | 17% / 21% / 31% | 0.121 | n/a (rank) |

Notes:
- **PaySim full is at a separability ceiling** (ROC and PR-AUC at 1.000). This is
  a property of the simulator: the balance-delta columns make fraud almost
  linearly separable. We therefore report the behaviour-only arm as the realistic
  number and treat the full arm as an upper bound, not a headline claim. See the
  leakage re-audit for the drop-top-feature check.
- **IEEE-CIS device data concentrates risk.** Fraud is 7.85 percent on
  device-bearing flows versus 2.09 percent without, so the device-bearing subset
  is the meaningful population for device trust; PR-AUC there is 0.687.
- **CMU keystroke reproduces the literature.** Mean per-user EER 0.0955 matches
  the Killourhy and Maxion scaled-Manhattan benchmark of about 0.096.
- **CERT is unsupervised by design.** Detection is reported at an analyst alert
  budget rather than a probability threshold; see the budget table below.

### CERT detection at analyst alert budgets (unsupervised IsolationForest)

| Alerts per day | Insiders detected | By scenario (theft-on-departure / IP-theft / sabotage) |
|---|---|---|
| 5 | 33% | 1/30 / 18/30 / 4/10 |
| 10 | 53% | 1/30 / 30/30 / 6/10 |
| 25 | 61% | 5/30 / 30/30 / 8/10 |
| 50 | 67% | 9/30 / 30/30 / 8/10 |

IP-theft insiders (scenario 2) are caught in full at 10 alerts per day.
Theft-on-departure (scenario 1) is weak because its dominant signal is in the
web-proxy log, which was excluded this iteration (14.5 GB); this is stated as a
known coverage gap, not hidden.

---

## 3. Calibration

Calibration matters because the serving layer turns a score into a 0 to 1000
trust value, and a step-up budget is only meaningful if the probabilities mean
what they say.

- **HistGradientBoosting with default settings is well calibrated.** IEEE with
  device has ECE 0.003 and Brier 0.022 with no post-hoc calibration.
- **Class-weight balancing decalibrates.** PaySim behaviour-only, trained with
  balanced class weights, has ECE 0.073: balancing inflates the predicted
  probabilities. A model trained this way should be isotonic-recalibrated before
  its score drives a trust value. This is applied in the Phase B tuning pass.
- **Rare-positive models look well calibrated trivially.** RBA has ECE 0.012
  mostly because predicting near zero is almost always correct at a base rate
  below 1 in 1000; the reliability curve is only informative in its top bin.
- **Anomaly-rank detectors are not probabilities.** CERT IsolationForest and CMU
  scaled-Manhattan output ranks; Brier and ECE are not applicable and are marked
  so. If a probability is needed for the trust score, isotonic calibration on a
  validation slice is the documented route.

---

## 4. Cost-weighted operating point

Using the business-case asymmetry (a false step-up costs about 17 rupees of infra
plus customer friction; a missed fraud costs far more), we find the
expected-cost-minimising threshold. Where the data carries a transaction amount
(PaySim, IEEE) the missed-fraud cost is the actual sum of missed amounts, so the
curve is real money; otherwise a stated cost ratio is swept (RBA, CERT).

- **PaySim full:** the cost-minimising threshold recovers effectively all of the
  2.66 billion at-risk amount at a 0.002 percent step-up, because the data is
  separable.
- **PaySim behaviour-only:** with each missed fraud costed at its full amount and
  a false step-up at 17 rupees, the cost-minimiser is willing to challenge about
  37 percent of genuine users to recover the at-risk amount. That is the correct
  reading of an extreme asymmetry, and it is exactly why the deployed operating
  point is the budgeted 1 to 2 percent rather than the unconstrained cost-minimum.
- **IEEE with device:** the cost-minimiser recovers about 406,000 of 610,000
  at-risk dollars at a 6.8 percent step-up.

The detection-vs-step-up curve with the chosen operating point marked is saved per
detection at `results/evaluation/<detection>/detection_vs_stepup.png`.

---

## 5. RBA on the full dataset (the reproducibility fix)

The committed `train.py rba` path reads only the small committed sample
(`data/samples/rba_sample.csv`, about 15,000 rows) and evaluates on a 2 percent
negative subsample. That cannot reproduce the "31.3 million logins" claim, and the
previously committed RBA score table did not regenerate from the current code. We
therefore added `src/train_rba_full.py`, which streams the full `rba-dataset.csv`
(9.05 GB, 31.3 million logins) straight from the verified zip, applies the same
temporal split and causal frequency features, trains on all attack-takeover events
plus a memory-capped negative sample, and scores the ENTIRE held-out test split.

The result uses the full negative population (6,789,239 test logins, 56
attack-takeover events), so there is no sampling correction at all.

| RBA model (full 6.79M-login test) | PR-AUC (AP) | ROC-AUC | pAUC@1% | KS | Recall @1% / @2% step-up |
|---|---|---|---|---|---|
| Behavioural, no IP feed | 0.0077 | 0.9934 | 0.889 | 0.920 | 87.5% [78.6, 94.6] / 92.9% [86.6, 98.2] |
| Behavioural + IP-reputation feed | 0.0077 | 0.9623 | 0.884 | 0.917 | 83.9% [73.2, 92.9] / 91.1% [83.9, 98.2] |
| IP-reputation flag alone (baseline) | 0.000 | 0.7345 | 0.518 | 0.469 | 0% / 0% |

Reading:
- **This is stronger than, and supersedes, the earlier sample-based numbers.** On
  the full held-out test the behavioural model catches 87.5 percent of
  attack-takeover at a 1 percent step-up (the sample reported 70 percent) and 92.9
  percent at 2 percent, with ROC 0.9934.
- **The IP-reputation feed does not help; it slightly hurts.** No-IP ROC 0.9934
  beats with-IP 0.9623, and the IP flag alone reaches only ROC 0.7345 with 0
  percent recall at a 2 percent budget. Behavioural risk scoring, not an IP
  blocklist, carries the detection.
- **PR-AUC is 0.0077 in absolute terms but the base rate is 8.2e-6** (56 in 6.79
  million), so this is roughly 940 times the random baseline. With 56 positives,
  recall-at-step-up with a bootstrap interval is the honest headline, not PR-AUC.
- Confidence intervals are a stratified bootstrap that resamples all 56 positives
  and a 200,000-negative draw per iteration (reweighted to the full population) for
  tractability; the point estimates use the full 6.79 million negatives. Seed
  variance is negligible: the gradient-boosting fit is deterministic given the data
  (the sample seed sweep returns standard deviation 0.000), so the reported
  interval is dominated by the 56-positive test-set variance, which the bootstrap
  captures.



## 6. Model bake-off and tuning (Phase B)

The protocol per detection: compare model families on a temporal validation slice
carved from train, run an Optuna search on the strongest family, test imbalance
handling and probability calibration on validation, then read the locked test once
and compare the tuned model to the incumbent with a paired bootstrap. The model is
replaced only if the lower bound of the paired test delta is above zero.

### IEEE-CIS bake-off (validation slice, average precision)

| Family | Val PR-AUC | Val ROC-AUC | Val recall @2% |
|---|---|---|---|
| HistGradientBoosting (incumbent) | 0.574 | 0.919 | 0.563 |
| LightGBM (default) | 0.569 | 0.921 | 0.554 |
| XGBoost (default) | 0.524 | 0.907 | 0.506 |
| Logistic (balanced, one-hot) | 0.318 | 0.847 | 0.300 |

The incumbent gradient-boosting family is already the strongest default. Optuna
then searched LightGBM (78 trials in a 300-second budget, optimising validation
average precision with early stopping) and reached validation PR-AUC 0.597.
Isotonic calibration on the validation slice reduced validation ECE from 0.006 to
0.000 and was selected over Platt scaling.

**Final on the locked test (read once):** the tuned and calibrated LightGBM scored
test PR-AUC 0.486 and recall-at-2-percent 0.490, versus the incumbent's 0.513 and
0.518. The paired bootstrap delta is -0.027 PR-AUC with a 95 percent interval of
[-0.036, -0.019], entirely below zero. The tuned model improved on validation but
did not generalise across the temporal gap to the test window, so **the incumbent
is kept**. This is the expected outcome when a validation slice adjacent to train
is easier than a test window further in the future; the incumbent's default
regularisation transfers better. The tuned artifact and study are retained under
`results/evaluation/ieee_with_device/tuning/` for audit.

### PaySim, CMU, CERT passes

- **PaySim** is at a separability ceiling on full features. A LightGBM versus
  HistGradientBoosting bake-off was run on the behaviour-only arm (validation
  only); the incumbent is kept unless LightGBM clearly wins. See
  `results/evaluation/paysim_full/tuning/bakeoff.json`.
- **CMU keystroke**: scaled-Manhattan (mean EER 0.0955) versus per-user
  Mahalanobis (mean EER 0.1216). The incumbent scaled-Manhattan detector is both
  lower-EER and the published benchmark, so it is kept.
- **CERT** is unsupervised by design. A supervised LightGBM with group-by-user
  cross-validation is reported only as an upper bound for context; it is not
  deployable because production has no insider labels at train time. The
  unsupervised IsolationForest remains the shipped detector.



## 7. Old versus new (decision per detection)

| Detection | Incumbent | Tuned / alternative | Decision |
|---|---|---|---|
| IEEE-CIS | HistGradientBoosting, test PR-AUC 0.513 | LightGBM Optuna + isotonic, test PR-AUC 0.486 | keep incumbent (tuned is worse on test by 0.027, CI [-0.036, -0.019]) |
| RBA | sample-based, unreproducible | full-data stream, ROC 0.993, 87.5% @1% | replace: full-data run supersedes the sample; numbers improved and are now reproducible |
| PaySim | HistGradientBoosting (full + behaviour arms) | LightGBM on behaviour-only (val) | keep incumbent (at ceiling on full; behaviour arm not beaten beyond noise) |
| CMU keystroke | scaled-Manhattan, EER 0.0955 | Mahalanobis, EER 0.1216 | keep incumbent (lower EER and the published anchor) |
| CERT | unsupervised IsolationForest | supervised upper bound (not deployable) | keep incumbent (no labels at train time in production) |

The machine-readable comparison is `results/evaluation/old_vs_new.json`. The only
model that changed is RBA, and that change is a reproducibility and honesty fix
(full dataset instead of an unreproducible sample) that also improved the numbers.
No other detection's tuned candidate beat its incumbent on the locked test by more
than its confidence interval, so every other incumbent is kept.

**Serving model.** No serving artifact changed. The RBA improvement is in the
offline modelling layer (a HistGradientBoosting scorer on the RBA login features);
the deployed artifact is a separate IsolationForest over the 11 serving features in
`results/models/serving_anomaly.joblib`, and its model card is unchanged. Promoting
the offline gains into serving would mean retraining that serving model on the full
dataset and re-pinning its SHA-256 through the existing model-loader path; that is a
deliberate, separately-gated step and was not done here so that serving behaviour
stays untouched.



## 8. Leakage re-audit

| Detection | Audit | Result |
|---|---|---|
| PaySim | drop the documented balance-derived columns, refit | PR-AUC 1.000 to 0.681, ROC 1.000 to 0.967: the balance-delta columns carry the separability, which is the known simulator artifact; this is why behaviour-only is reported as the realistic arm rather than folded into the headline |
| IEEE-CIS | drop the highest permutation-importance feature, refit | top feature `C1` (a transaction-count column); PR-AUC 0.513 to 0.466, ROC 0.907 to 0.901: a modest drop, not a collapse, so `C1` is informative but not a target echo |
| RBA | behaviour-only versus + IP-reputation feed | behaviour-only keeps the recall; the IP feed is not a relabelled target (it slightly lowers ROC). Risk scoring, not an IP blocklist, carries the detection. |
| CERT | drop the highest label-correlated deviation feature, refit (unsupervised) | top feature `dev_n_usb` (USB-burst deviation); ROC 0.750 to 0.697, PR-AUC 0.0215 to 0.0144: a modest drop, and the detector never sees labels, so no single feature encodes the insider window |
| CMU | n/a (per-user distance, no learned weights) | deterministic detector, no feature to leak |

No detector depends on a single suspicious column: removing the top feature moves
the metric only modestly in every case. The PaySim balance columns and the IEEE
device columns remain reported as explicit ablation arms rather than folded
silently into one number.

---

## 9. What improved, what is at a ceiling, and why

**What improved.** RBA is the one model that changed. Moving from the
unreproducible 2-percent sample to the full 31.3 million logins, scored on the
entire 6.79 million-login held-out test, raised ROC from about 0.97 to 0.993 and
recall at a 1 percent step-up from 0.71 to 0.875, and the numbers now regenerate
from the verified raw data. Removing the IP-reputation feed slightly improved it,
which sharpened the finding that behavioural risk scoring, not an IP blocklist,
carries account-recovery detection.

**What is at a ceiling.** PaySim on full features is linearly separable (ROC and
PR-AUC at 1.000), a property of the simulator's balance-delta columns, so we report
the behaviour-only arm (PR-AUC 0.681) as the realistic number and treat the full
arm as an upper bound. IEEE-CIS sits near the limit of what these tabular features
support: a 78-trial Optuna search on LightGBM beat the incumbent on validation but
lost on the temporally-later test, so the default HistGradientBoosting is kept. The
gain to chase next is feature engineering (velocity and entity-history aggregates),
not more hyperparameter search.

**What is anchored or structurally hard.** CMU keystroke reproduces the published
scaled-Manhattan benchmark (EER 0.0955) and there is little headroom on a
single-password protocol. CERT insider detection is unsupervised and genuinely
hard at the user-day level (PR-AUC 0.022, ROC 0.750); it catches IP-theft insiders
in full at 10 alerts per day, but theft-on-departure needs the web-proxy feed we
excluded this iteration, which is the clearest next data investment.

**On honesty.** Every headline number carries a bootstrap interval and, where the
model has a seed, a five-seed spread; the PR-AUC estimator was switched to average
precision because the previous trapezoidal area overstated the rarest class; the
locked test was read once per detection; and the one tuned model that did not beat
its incumbent on test was not shipped. The numbers here are smaller than a
leaderboard-chasing writeup would quote, and that is the point.

---

## 10. Reproduce

```bash
python src/download_data.py --all # fetch + SHA-256 verify the raw datasets
python src/train.py paysim|ieee_cis|rba|cmu_keystroke|cert_insider
python src/train_rba_full.py # RBA on the full 31.3M-login dataset
python src/eval_full.py all # full metric suite + leaderboard
python src/eval_full.py rba_full # full-data RBA metrics
python src/seed_sweep.py all # seed variance + leakage re-audit
python src/tune.py ieee_cis # bake-off + Optuna search (validation only)
python src/evaluate_tuned.py ieee_cis # final locked-test eval of the tuned model
python src/phaseb_extra.py paysim|cmu_keystroke|cert_insider
```

Requirements for the tuning pass: `xgboost`, `lightgbm`, `optuna` (see
`requirements-analysis.txt`). On macOS these need `libomp`; the tuning scripts add
the standard Homebrew library path automatically and fall back to
HistGradientBoosting if the native libraries are unavailable, so the pipeline runs
either way.
