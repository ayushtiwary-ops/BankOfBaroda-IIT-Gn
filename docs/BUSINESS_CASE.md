# PRAMAAN - Business Case (₹ ROI, arithmetic shown)

> Bankers buy money saved, not architecture. This model has **two kinds of
> numbers**: (1) PRAMAAN's **measured efficacy** (from `results/`, reproducible)
> and (2) **scale/loss ASSUMPTIONS** - every assumption is labelled `[A]` and
> must be replaced with Bank of Baroda's own actuals. The arithmetic is shown so
> any number can be checked or re-run with different inputs.

## 1. Measured efficacy (NOT assumptions - see `results/`)

| Lever | Measured result | Source |
|---|---|---|
| Account-takeover caught | **93% @ 2% step-up, 87.5% @ 1%** (ROC 0.993, full 6.79M-login test) | `results/evaluation/rba_full_noip/metrics_full.json` |
| Mule / money-flow caught | **100% @ 1% step-up** (PR-AUC 1.00) | `results/paysim_full/metrics.json` |
| Genuine new-user friction | **0% step-up** (real model + cold-start prior) | `results/coldstart/report.json` |
| Per-request scoring latency | **p99 ≈ 5 ms** | `results/load/load_report.json` |

## 2. Assumptions (replace with BoB actuals)

| # | Assumption | Value `[A]` | Basis |
|---|---|---|---|
| A1 | Annual digital sessions (login/txn/recovery) | **2,000,000,000** | large-PSB order of magnitude; use BoB's actual volume |
| A2 | Residual ATO/fraud **loss** after current controls (per year) | **₹250 crore** | placeholder; use BoB's reported digital-fraud residual |
| A3 | Fraction of that loss in ATO + mule patterns PRAMAAN addresses | **70%** | ATO + money-mule share of digital fraud |
| A4 | Cost of one step-up challenge (OTP/biometric infra + ops) | **₹2** | per-challenge marginal cost |
| A5 | Cost of a false step-up to CX (churn/abandonment proxy) | **₹15** | per genuine-user friction event |
| A6 | Annual platform run cost (compute + ops, BoB scale) | **₹6 crore** | stateless pods + Redis/Kafka/Postgres + SRE |

## 3. The arithmetic

**Fraud avoided.** Addressable residual loss = A2 × A3 = ₹250 cr × 0.70 = **₹175 cr/yr**.
PRAMAAN catches 93% of ATO and ~100% of mule flow; using a blended **90%**
catch on the addressable loss (conservative vs the measured 93–100%):

```
fraud avoided = ₹175 cr × 0.90 = ₹157.5 cr / yr
```

**Friction cost.** PRAMAAN only steps up elevated-risk events. At a **2%**
step-up rate on A1 sessions, of which the vast majority are true-risk (genuine
new-user step-up is measured at 0%):

```
step-ups / yr = 2,000,000,000 × 0.02 = 40,000,000
challenge infra cost = 40,000,000 × ₹2 = ₹8.0 cr / yr (A4)
false-step-up CX cost: assume only 10% of step-ups hit genuine users
                     = 40,000,000 × 0.10 × ₹15 = ₹6.0 cr / yr (A5)
friction cost total = ₹14.0 cr / yr
```

**Net benefit & ROI.**

```
net benefit = fraud avoided − friction cost − run cost
            = ₹157.5 cr − ₹14.0 cr − ₹6.0 cr (A6)
            = ₹137.5 cr / yr

ROI = net benefit / run cost = ₹137.5 cr / ₹6.0 cr ≈ 23×
```

## 4. Why the friction number is the differentiator

Every "we detect fraud" vendor can quote a catch rate. PRAMAAN's edge is the
**denominator**: it challenges only ~2% of sessions and **0% of genuine new
users** (measured), so the CX cost stays a rounding error against the fraud
avoided. A naïve "step up everyone risky-looking" system inverts this - that is
exactly the cold-start friction-bomb we measured at 95% on the synthetic
baseline and drove to 0% with the real-data model.

## 5. Sensitivity (so the case survives scrutiny)

| If A2 (residual loss) is… | fraud avoided | net benefit | ROI |
|---|---|---|---|
| ₹100 cr | ₹63 cr | ₹43 cr | ~7× |
| ₹250 cr (base) | ₹157.5 cr | ₹137.5 cr | ~23× |
| ₹500 cr | ₹315 cr | ₹295 cr | ~49× |

Even at the pessimistic end (₹100 cr residual, all costs as stated), the
platform pays for itself several times over. **Plug BoB's real A1–A6 to get the
bank-specific figure; the arithmetic above is the model, not a claim.**
