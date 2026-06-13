# Security Hardening Changelog

This document records the security weaknesses we found in PRAMAAN during
development and how we closed each one. Every item was implemented with a
failing test first and a passing test after the fix (red → green), and every
fix carries a `# SECURITY:` / `# HARDENING:` tag in the code. After the first
pass we ran an internal adversarial review that re-attacked each surface; it
found a few regressions and gaps, which are listed here too and were fixed the
same way.

Run the security suite:

```bash
python -m pytest -q tests/test_security.py     # the hardening tests
python -m pytest -q                            # full suite (84 tests)
```

The offline metrics pipeline (`src/train.py`, `src/evaluate.py`) was not changed
during hardening; the serving code only reads its outputs.

---

## 1. The trust boundary (the core idea)

The single principle behind everything below: **the server never believes a
security claim the client makes about itself.** Signals are either recomputed
server-side, or accepted only as a signature from a key the server cannot mint.

| Area | What was weak | What we changed | Proof |
|---|---|---|---|
| Step-up outcome | `POST /v1/stepup/{id}?verified=true` trusted a client boolean | Step-up is accepted only as an **Ed25519-signed assertion** from a trusted verifier, with signature + freshness + single-use nonce + identity binding checked server-side. The engine holds only the **public** verify key, so it can check a verifier's blessing but never forge one. | `tests/test_security.py::test_ks1_*` (`verifier.py`, `main.py`) |
| Behaviour signal | Client could send a trusted `behavior_score` (e.g. a fixed `0.99`) | The field is removed (`extra="forbid"`). Behaviour is trusted only via a **device-attested, signed similarity assertion** bound to the attested device; otherwise it is treated as MISSING (cold-start), never as a perfect score. Non-finite (NaN) similarity is rejected. | `test_ks2_*`, `test_r2_nan_similarity_is_not_trusted` (`attestation.py`, `schemas.py`, `features.py`) |

---

## 2. Real model and supply-chain integrity

| What was weak | What we changed | Proof |
|---|---|---|
| The live engine could fall back to a model fit on random noise | The serving anomaly model is exported from **real RBA login data** through the exact serving feature code (`src/export_models.py`), with a model card. In `prod` the loader **fails loud** if the artifact is absent. | `test_ks3_*`, `test_export_builds_a_loadable_real_artifact` (`model_loader.py`) |
| Loading a `joblib` artifact is a pickle (RCE) surface, and a card that signs its own hash is not a trust anchor | The loader verifies the artifact against an **out-of-band pinned digest** (`PRAMAAN_MODEL_SHA256`) with a constant-time compare **before** `joblib.load`; the card hash is demoted to a corruption check. | `test_r2_pinned_model_digest_mismatch_is_rejected`, `..._match_loads` |

---

## 3. Authentication, authorization, and secrets

| What was weak | What we changed | Proof |
|---|---|---|
| Service was effectively open; CORS wildcard; an identity-lookup IDOR; default secrets in source | Every endpoint requires a **scoped API key** (constant-time compare). CORS is an explicit allowlist (wildcard banned at config load). `GET /v1/identity/{id}` is gated behind an `identity:read` scope and returns minimal fields. All secrets come from env/KMS with **no shipped defaults**; config fails loud if they are missing. | `test_ks4_*`, `test_ks4d_*` (`config.py`, `auth.py`, `main.py`) |
| Silent crypto downgrade: a malformed Ed25519 public key fell back to HMAC | Config **fails loud** on a non-Ed25519 public key when `cryptography` is present. | `test_r2_malformed_ed25519_pubkey_fails_config` |

---

## 4. Two planes: never leak the detector

| What was weak | What we changed | Proof |
|---|---|---|
| Rich reason codes (and the trust score) were returned to the client, turning the detector into an oracle | The **client plane** receives a generic decision and a challenge id only. The **SOC/audit plane** carries the full reasons, feature contributions, SHAP values, and counterfactuals. The split is enforced in the schemas and verified end-to-end (including `/openapi.json` and error bodies). | `test_ks8_*`, `test_p5_shap_and_counterfactual_on_audit_plane_not_client` (`schemas.py`, `explain.py`) |

---

## 5. State and concurrency

| What was weak | What we changed | Proof |
|---|---|---|
| In-process mutable state could race and would not scale | State sits behind a `StateStore` interface (`InMemoryStateStore` + `RedisStateStore`) with **per-key locking and version CAS**, so scoring pods are stateless and concurrent updates cannot be lost. | `test_ks7_*` (`state_store.py`, `risk_engine.py`) |
| Deserializing state from a writable store via pickle is a CWE-502 RCE | State round-trips through a schema-explicit **JSON serializer**, never pickle. | covered by `test_ks7_*` round-trips |

---

## 6. Tamper-evident audit

| What was weak | What we changed | Proof |
|---|---|---|
| A plain hash chain can be recomputed by an attacker who can write the store | Each record is **SHA-256 + HMAC** under the SOC key, so rewriting the chain is useless without the key. | `test_x_keyed_audit_chain_resists_recompute_attack` (`audit.py`) |
| A prefix of a valid chain still verifies, so tail truncation goes unnoticed | A **signed head checkpoint** (length + head, MAC'd) is persisted out of band and verified, and `seq == index` is asserted; exposed on `/v1/audit/verify`. | `test_r2_audit_truncation_detected_by_head_checkpoint` |

---

## 7. Resilience and rate limiting

| What was weak | What we changed | Proof |
|---|---|---|
| No policy for a degraded engine | Sensitive events **fail closed** (at least step-up); routine events **fail open**; both are stamped `degraded` and audited. | `test_x_degraded_engine_fails_closed_on_privileged`, `..._fails_open_on_routine_login` (`resilience.py`) |
| No defense against step-up bombing / MFA fatigue | Per-identity step-up attempts are capped per window, with **failures (cap 20) and accepted (cap 5) counted separately** and validation done first, so garbage assertions cannot lock a victim out of their real one. | `test_x_stepup_bombing_is_rate_limited`, `test_r2_failed_stepups_do_not_block_a_valid_one` |
| Replayed events could manipulate trust | Idempotency is keyed on `(client_id, key)` plus a SHA-256 **payload fingerprint**; the same key with a different payload is rejected (409); a true replay returns the prior decision, never a fresh score. | `test_x_idempotency_dedupes_replayed_event`, `test_r2_idempotency_key_reuse_with_different_payload_is_rejected` |

---

## 8. Privacy in code

| What was weak | What we changed | Proof |
|---|---|---|
| Differential privacy was decorative | The feature-aggregation export passes through a real **Gaussian mean-release** mechanism with **Laplace noisy-count** cohort suppression, accounted by an RDP accountant: **ε ≈ 1.1126, δ = 1e-5** (`results/privacy_budget.json`). An un-noised export is blocked. | `test_dp_export_*` (`src/dp_export.py`, `privacy.py`) |
| Right-to-erasure conflicts with an immutable audit chain | **Crypto-shredding**: the chain holds only tokenized references; destroying a per-identity key makes the material irrecoverable while the chain still verifies. Erased ids are tombstoned so a later event cannot resurrect the identity. | `test_ks6_*`, `test_r3_erased_identity_is_not_resurrected` (`keystore.py`) |

See [COMPLIANCE.md](COMPLIANCE.md) for the DPDP Act 2023 and RBI mapping.

---

## 9. Robustness and fairness

| What was weak | What we changed | Proof |
|---|---|---|
| Low-and-slow baseline poisoning crept in under the threshold | Drift detection combines a windowed mean-shift with a **persistent-baseline CUSUM** over an EWMA, and passive trust recovery is **rate-limited**; only ALLOW events update the baseline. | `test_ks9_*`, `test_x_passive_trust_recovery_is_rate_limited` (`drift.py`) |
| Impossible-travel reference was poisoned by updating it on non-allowed events | `last_geo` is updated **only on ALLOW**, so repeated impossible-travel attempts keep flagging. | `test_r3_impossible_travel_keeps_flagging_repeated_attempts` |
| A "look new" first contact could be waved through by the cold-start prior | The prior is gated to benign-shaped first contacts and is one-shot. | `test_r3_cold_start_prior_does_not_help_malicious_first_contact` |
| Naive selection-rate "fairness" hides harm | The fairness audit **surfaces** the geo disparate-impact and **rejects** selection-rate parity, because on a fraud detector it is a quantile tautology that collapses high-attack ATO recall (measured). It recommends equalized-odds plus feature-level treatment. | `test_fairness_audit_surfaces_disparity_and_rejects_parity` (`src/fairness_audit.py`) |

---

## 10. Known limitations and future work

We state these openly rather than paper over them; none are fakeable.

- **Live deploy** is prepared (`deploy/`) but is an outbound publish under the
  team's own account, so it is not auto-run here.
- **Distributed load** is measured single-node today (per-request scoring
  p99 ≈ 5 ms; a single worker saturates around 210 rps). A multi-pod k6 run
  against the Redis/Kafka stack is the production scale proof.
- **Impossible-travel offline lift** needs the full ordered RBA stream; the
  small committed sample breaks consecutive-login adjacency. The live feature is
  implemented and tested.
- **DP-SGD** (e.g. Opacus) on the keystroke model would extend differential
  privacy from the aggregation export to the behavioural model itself.
- **Chaos / fault-injection** tests and equalized-odds fairness monitoring with
  dense per-cohort labels are planned.
