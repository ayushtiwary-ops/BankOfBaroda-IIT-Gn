# PRAMAAN — Data Sources & Detection/Metric Mapping

> The evidence base. Every quantitative claim PRAMAAN makes is traceable to a row
> in this file: a real, public, licensed dataset with a recorded source URL,
> license, size, schema, checksum, and the exact detection + metric it proves.
> No synthetic-only models, no magic numbers. (Charter §"Definition of Done".)

**Reproduce everything:** `python src/download_data.py --all` → verifies SHA-256 →
`python src/train.py` → `python src/evaluate.py` regenerate every reported number.

---

## 1. Master mapping — 5 detections × real data × real metric

| # | Problem-statement detection | Dataset (proof) | Ground-truth label | Headline metric | Status |
|---|---|---|---|---|---|
| 1 | Anomalous behavior (UEBA / behavioral biometrics) | **CMU Keystroke Dynamics** + IEEE-CIS | impostor vs genuine typing; `isFraud` | EER, ROC-AUC | ✅ keystroke acquired |
| 2 | New-device usage / device & access trust | **IEEE-CIS** (`id_30/31`, `DeviceType`, `DeviceInfo`) | `isFraud` | PR-AUC, recall @1% FPR | ✅ acquired+verified |
| 3 | Suspicious onboarding (mule / synthetic identity) | **PaySim** (money-flow graph) | `isFraud` (TRANSFER→CASH_OUT chains) | PR-AUC, recall @1% FPR | ✅ acquired+verified |
| 4 | Suspicious account recovery (ATO precursor) | **RBA login dataset** (Wiefling) | `Is Account Takeover` | PR-AUC, recall @ fixed step-up rate | ✅ acquired+verified |
| 5 | Privileged-access misuse (insider threat) | **CERT Insider Threat r4.2** | insider `scenario` labels | PR-AUC, detection @ alert budget | ✅ acquired+verified |

Cross-cutting constraints (each must independently score 97+): **Privacy · Compliance ·
Scalability · Friction-optimized UX** — see §8 for how the data handling demonstrates each.

---

## 2. PaySim — mule & money-flow fraud  ·  Detection #3  ✅

| Field | Value |
|---|---|
| Source | HF mirror `theman10/paysim` → `paysim.csv` (original: PaySim, Lopez-Rojas, Elmir & Axelsson, 2016) |
| URL | https://huggingface.co/datasets/theman10/paysim · original sim: https://github.com/EdgarLopezPhD/PaySim |
| License | MIT (mirror); PaySim simulator is academic/open |
| Size | 493,534,783 bytes (471 MB CSV) |
| SHA-256 | `16910f90577b0d981bf8ff289714510bb89bc71bff7d3f220f024e287e4eea6b` |
| Rows × cols | 6,362,620 × 11 |

**Schema:** `step` (hour, 1–743 ≈ 31 days), `type` (PAYMENT/TRANSFER/CASH_OUT/CASH_IN/DEBIT),
`amount`, `nameOrig`, `oldbalanceOrg`, `newbalanceOrig`, `nameDest`, `oldbalanceDest`,
`newbalanceDest`, `isFraud`, `isFlaggedFraud`.

**Profiled facts (real, from our copy):**
- Fraud = **8,213 / 6,362,620 = 0.1291%** — realistic extreme imbalance → report **PR-AUC**, not accuracy.
- Fraud occurs **only in CASH_OUT (4,116) and TRANSFER (4,097)** — the textbook mule
  cash-out signature; PAYMENT/CASH_IN/DEBIT have **zero** fraud.
- The dataset's own rule `isFlaggedFraud` catches **16 of 8,213 (0.19%)** → a weak baseline
  we beat by a wide margin = clean "ML lift over rules" story.
- 6,353,307 unique originators; 2,722,362 unique destinations; 2,151,495 merchant (`M…`) dests.

**What it proves:** mule-account onboarding & money mule chains (TRANSFER→CASH_OUT to the same
dest), velocity/balance-drain anomalies, synthetic-identity money movement.
**Leakage caution:** `oldbalance/newbalance` deltas trivially encode the transfer; we engineer
*behaviour-relative* features (per-account velocity, dest in-degree, balance-zeroing flag) and
report metrics with and without raw balance columns to avoid an inflated, gameable score.
**Reproduce:** `python src/download_data.py paysim`

---

## 3. RBA Login Dataset (Wiefling et al.) — account recovery / login risk  ·  Detection #4  ✅

| Field | Value |
|---|---|
| Source | Zenodo record 6782156 — "Login Data Set for Risk-Based Authentication" (das-group) |
| URL | https://zenodo.org/records/6782156 · repo: https://github.com/das-group/rba-dataset |
| License | **CC BY 4.0** (cite Wiefling et al., ACM TOPS 2022 "Pump Up Password Security!") |
| Size | 1,093,700,330 bytes zip → 9,052,907,531 bytes (8.4 GB) `rba-dataset.csv` |
| SHA-256 (zip) | `ca1d974e97aebfb30878a613f4ca5c793860a98ba2acb5185c9bc610d7432a33` |
| Rows × cols | 31,269,264 × 16 |

**Schema (confirmed):** `index`, `Login Timestamp`, `User ID`, `Round-Trip Time [ms]`, `IP Address`,
`Country`, `Region`, `City`, `ASN`, `User Agent String`, `Browser Name and Version`,
`OS Name and Version`, `Device Type`, `Login Successful`, `Is Attack IP`, `Is Account Takeover`.
> Synthesized from the real login behaviour of 3.3M users at a Norwegian SSO (Feb 2020–Feb 2021);
> IPs/UA/timestamps/RTTs randomized so it carries **no sensitive values** — privacy-clean by construction.

**Profiled facts (real, from our copy):**
- **31,269,264 logins**; **141 account-takeover events = 0.00045%** — the realistic extreme-rarity ATO
  scenario → PR-AUC + recall-at-fixed-step-up-rate is the only honest way to score it.
- On the first 10M logins: **9.55% from attack IPs**, **45.9% successful logins** — rich attack signal
  for new-device/new-geo/new-ASN risk even where ATO itself is sparse.
- All **141 ATO positives** are retained in `data/samples/rba_sample.csv` (+ a seeded, order-preserved
  ~15k negative sample, kept small for GitHub); full-set exact tallies are regenerated by `train.py`
  streaming the zip in chunks.

**What it proves:** this is the *crown jewel* for the problem statement — it is literally an
RBA dataset with **account-takeover labels**. Proves new-device/new-geo/new-ASN login risk,
impossible-travel, and suspicious account-recovery precursors with a real ATO ground truth.
**Metric:** PR-AUC + recall at a fixed **step-up rate** (the friction budget) — our differentiator.
**Reproduce:** `python src/download_data.py rba`

---

## 4. IEEE-CIS Fraud — anomalous behavior + new-device / device trust  ·  Detections #1 & #2  ✅

| Field | Value |
|---|---|
| Source | Kaggle competition `ieee-fraud-detection` (IEEE-CIS / Vesta Corporation) |
| URL | https://www.kaggle.com/c/ieee-fraud-detection |
| License | Competition use (cite IEEE-CIS & Vesta) — raw data **not redistributed**; no committed sample |
| Size | zip 123,856,947 bytes → 1,354,953,156 bytes across 5 CSVs |
| SHA-256 (zip) | `4cc646da09d0a9b265983ffed775b1f9ee15af5266586df610e04d6adae0b829` |
| Rows | 590,540 train transactions × 394 cols; identity 144,233 × 41 |

**Profiled facts (real, from our copy):**
- **20,663 fraud = 3.499%** over a 182-day span.
- Identity/device table joins to **24.42%** of transactions; within identity rows the device columns
  are rich: `DeviceType` 97.6%, `id_31` (browser) 97.3%, `DeviceInfo` 82.3%, `id_30` (OS) 53.8% non-null.
- **Fraud rate with identity present: 7.85% vs 2.09% without (≈3.7×)** — device-signal-bearing flows
  are the risk concentration, which is precisely the device-trust thesis.

**Schema highlights:** `TransactionDT/Amt`, `card1–6`, `addr1/2`, `P/R_emaildomain`, `C1–14`,
`D1–15`, `M1–9`, `V1–339`; identity: `id_01–38`, **`id_30` (OS)**, **`id_31` (browser)**,
**`DeviceType`**, **`DeviceInfo`**. Label `isFraud`.

**What it proves:** real device/identity signals for **new-device detection** and **device trust**
(OS+browser+device fingerprint mismatch vs a customer's history) and high-dimensional behavioral
anomaly. Chosen raw (not pre-engineered) so the device columns survive.
**Metric:** PR-AUC, ROC-AUC, recall @1% FPR; device-trust ablation (with/without `id_30/31/DeviceInfo`).
**Reproduce:** `python src/download_data.py ieee_cis` (needs Kaggle CLI + one-time rules accept).

---

## 5. CMU Keystroke Dynamics — behavioral biometrics  ·  Detection #1  ✅

| Field | Value |
|---|---|
| Source | Killourhy & Maxion, CMU — "Comparing Anomaly-Detection Algorithms for Keystroke Dynamics" (DSN 2009) |
| URL | https://www.cs.cmu.edu/~keystroke/ (mirror used: github.com/njanakiev/keystroke-biometrics) |
| License | Free for research use (CMU benchmark) |
| Size | 4,629,134 bytes (4.5 MB) |
| SHA-256 | `4a7086f601052e307eff24a4bc525c8d104662f8ba06da1ac8080e70a6d55789` |
| Rows × cols | 20,400 × 34 |

**Profiled facts:** **51 subjects**, **400 reps each** (8 sessions × 50), typing the password
`.tie5Roanl`; **31 timing features** = 11 Hold (`H.*`) + 10 Down-Down (`DD.*`) + 10 Up-Down (`UD.*`).

**What it proves:** behavioral-biometric authentication done **end-to-end on real human timing
data** (not an assumed client float). Genuine-vs-impostor per user → the behavior signal is
*demonstrated* end-to-end, never assumed from a client-supplied score.
**Metric:** Equal Error Rate (EER) per user + mean EER, ROC-AUC (Manhattan/Mahalanobis detector
benchmark ≈ 0.084–0.10 EER in the literature → our reproduction target).
**Privacy note:** in production the similarity score is computed **on-device**; raw keystroke
timings never leave the client. This dataset is used only to train/validate that detector offline.
**Reproduce:** `python src/download_data.py cmu_keystroke`

---

## 6. CERT Insider Threat r4.2 — privileged-access misuse  ·  Detection #5  ✅

| Field | Value |
|---|---|
| Source | CMU CERT / ExactData — Insider Threat Test Dataset (Kilthub); fetched per-file from HF mirror `jinmang2/cert_insider_threat` |
| URL | https://kilthub.cmu.edu/articles/dataset/Insider_Threat_Test_Dataset/12841247 |
| License | **ExactData EUA — no redistribution.** We hold raw locally and publish only aggregate metrics/derived features ("minimal extent necessary to describe performance"), with the required © notice. **No committed sample.** |
| Files held | `logon.csv` 58,514,706 B · `device.csv` 28,982,749 B · `file.csv` 193,055,265 B · `psychometric.csv` · `answers/insiders.csv` · readme/license (SHA-256 in `data/raw/cert_insider/checksums.sha256`) |
| Scope decision | `http.csv` (14.5 GB) excluded this sprint; `email.csv` (1.36 GB) deferred — logon/device/file is the standard feature set for scenarios 1–2 in the CERT literature. |

**Profiled facts (real, from our copy):**
- `logon.csv`: **854,859 events, 1,000 users**, 2010-01-02 → 2011-05-17 (~16.5 months); **5.31% of logons are off-hours** (before 06:00 / after 20:00).
- `device.csv`: **405,380 USB connect/disconnect events** across 265 users; `file.csv`: **445,581
  file-copies to removable media** across 264 users.
- Ground truth: **70 r4.2 insiders** — scenario 1 (data theft on departure) 30, scenario 2 (IP
  theft/exfil) 30, scenario 3 (IT-admin sabotage) 10. Label join verified: **70/70 in logon,
  70/70 in device, 69/70 in file**.
- Privileged users are identifiable via job role `ITAdmin` (LDAP) — the privileged-misuse frame.

**What it proves:** privileged/insider misuse via UEBA — off-hours logon, logins to other users'
machines, abnormal USB/file-exfil bursts — against **real insider-threat labels**.
**Metric:** PR-AUC + detection rate at a fixed analyst **alert budget** (top-N user-days/day).
**Reproduce:** `python src/download_data.py cert_insider`

---

## 7. Why these five (and not others)

Each dataset is the closest *public, labelled* proxy to one problem-statement detection, and
together they span the full identity-trust surface (customer txn fraud, login/ATO, device trust,
behavioral biometrics, insider/privileged misuse). The charter's crown jewels are all present:
**RBA (Wiefling)** for login risk, **IEEE-CIS** for device+identity, **PaySim** for money mules,
**CMU keystroke** for behavioral biometrics, **CERT** for insider threat.

## 8. Data handling ↔ the four cross-cutting constraints

- **Privacy:** RBA is privacy-clean by construction (randomized IP/UA/RTT); keystroke similarity
  is computed on-device; identifiers (`nameOrig`, `User ID`, IPs) are HMAC-tokenized at the edge
  before the engine sees them; DP noise applied to feature-aggregation exports (stated ε).
- **Compliance (DPDP/RBI):** raw datasets stay local (`data/raw/**` is git-ignored); only
  samples + processed feature tables + checksums are committed; erasure modeled via
  crypto-shredding of per-identity keys, not deletion from the immutable audit chain.
- **Scalability:** feature tables are columnar/streamable; class imbalance handled with
  PR-AUC + threshold-at-FPR rather than resampling tricks that don't survive production volume.
- **Friction-optimized UX:** every detector is reported as a **detection-vs-step-up-rate curve**
  so we can state "catch X% of fraud while challenging only Y% of genuine users".

## 9. Provenance & licensing compliance
- All datasets are public and used within their licenses; attributions above are reproduced in
  the README and submission. CC BY 4.0 (RBA) and MIT (PaySim mirror) permit redistribution of
  samples; IEEE-CIS raw is **not** redistributed (download via Kaggle); CERT subset retained per
  its public-research terms.
- Multi-GB raw files are **never committed** (`data/.gitignore`); the repo carries
  `*.sha256` checksums + `download_data.py` so any reviewer regenerates byte-identical inputs.

_Last updated by the Tier-1 data sprint. Status flags above flip to ✅ as each acquisition is
verified; see `results/profile_*.json` for machine-readable profiles._
