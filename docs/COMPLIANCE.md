# PRAMAAN — Compliance Map (DPDP Act 2023 · RBI)

> Every obligation below maps to a **specific code path** (file:symbol) and, where
> applicable, a test that proves it. Nothing here is narrated — it is demonstrated.

## DPDP Act 2023 (Digital Personal Data Protection)

| Obligation | How PRAMAAN satisfies it | Code path | Proof |
|---|---|---|---|
| **Right to erasure** (§12) | Crypto-shredding: per-identity material is encrypted under a per-identity key; "erase" destroys the key → material irrecoverable while the immutable audit chain stays verifiable | `app/keystore.py::KeyedPiiStore.erase`, `app/main.py::erase_identity` (`DELETE /v1/identity/{id}/erase`, scope `identity:erase`) | `test_ks6_crypto_shred_makes_material_irrecoverable`, `test_ks6_audit_chain_still_verifies_after_erasure`, `test_ks6_erase_endpoint_is_soc_scoped_and_keeps_chain_intact` |
| **Data minimization** (§8) | No raw PII reaches the engine: identifiers are HMAC-tokenized at the edge; geo coarsened to a state/country bucket; behaviour reduced to a similarity score on-device | `app/privacy.py::pseudonymize` / `coarsen_geo`; `app/schemas.py::IdentityEvent` (tokens only) | `test_pseudonymization_is_deterministic_and_opaque` |
| **Purpose limitation / no client trust** | The engine believes only signed/attested inputs; reason codes & trust score stay on the SOC plane (no over-collection-by-oracle) | `app/verifier.py`, `app/attestation.py`, `app/schemas.py::RiskAssessment` vs `SocAssessment` | `test_ks1_*`, `test_ks2_*`, `test_ks8_*` |
| **Grievance redress / explainability** (§13) | Every decision carries human-readable reason codes + feature contributions on the audit plane for redressal; SHAP + a counterfactual are attached | `app/risk_engine.py::_explain`, `app/explain.py`, `SocAssessment.to_audit_payload` | `test_ks8_audit_payload_carries_full_reasons` |
| **Security safeguards** (§8(5)) | Secrets from env/KMS (no defaults), scoped auth on every endpoint, keyed tamper-evident audit, model-artifact integrity pin | `app/config.py`, `app/auth.py`, `app/audit.py`, `app/model_loader.py` | `test_ks4_*`, `test_ks4d_*`, `test_x_keyed_audit_*`, `test_r2_pinned_model_*` |
| **Retention limitation** | Per-identity keys held only for `RETENTION_DAYS` (default 365); an erase request short-circuits the window immediately | `app/keystore.py::RETENTION_DAYS` | documented constant |
| **DP on analytics exports** | Aggregate feature statistics exported for retraining pass through the Gaussian mechanism with a stated, accounted ε≈1.0 (δ=1e-5) | `src/dp_export.py`, `results/privacy_budget.json` | `test_dp_export_*` |

## RBI (Master Directions — Digital Payment Security Controls; data localization)

| Obligation | How PRAMAAN satisfies it | Code path |
|---|---|---|
| **Data localization** (payment data resides in India) | Multi-GB raw datasets are git-ignored and stay local; only samples + processed feature tables + checksums are committed; the keyed PII vault is a single-region store | `data/.gitignore`, `DATA_SOURCES.md §9`, `app/keystore.py` |
| **Risk-based authentication / continuous validation** | Continuous trust score with risk-banded step-up; verification triggered only on elevated risk | `app/risk_engine.py`, `app/policy.py` |
| **Strong customer authentication for step-up** | Step-up outcomes are signed verifier assertions (OTP/WebAuthn/video-KYC), never client-asserted | `app/verifier.py`, `app/main.py::step_up_outcome` |
| **Audit trail / non-repudiation** | Keyed (HMAC) hash chain + out-of-band head checkpoint detect any edit, recompute, or truncation | `app/audit.py::verify_chain` / `head_checkpoint` |
| **Fraud monitoring & explainable alerts** | Reason codes + feature contributions to the SOC plane; fairness audit across geo cohorts | `app/audit.py`, `results/fairness/report.json` |

## The erasure ↔ immutable-audit reconciliation (the sophistication signal)

A naïve design cannot offer BOTH an immutable audit chain AND right-to-erasure.
PRAMAAN resolves it with **crypto-shredding**:

1. The audit chain stores only the pseudonymous token + a ciphertext `pii_ref` —
   never plaintext (`main.py::ingest_event` writes `payload["pii_ref"]`).
2. PII / behavioural-template material lives in `KeyedPiiStore`, each record
   encrypted under a per-identity key.
3. Erasure destroys the key (`KeyedPiiStore.erase`). The ciphertext and the chain
   are byte-for-byte unchanged — `verify_chain()` still returns `True` — but the
   subject's data is cryptographically unrecoverable.

Result: the regulator gets a provably-intact audit trail, and the data subject
gets a real, irreversible erasure. Both, with no contradiction.
