# PRAMAAN — Threat Model (STRIDE + attack tree)

> Every control maps to the threat it closes and the test/artifact that proves
> it. This is the "elite security team" signal: we name what the server is
> allowed to believe, and what happens when each assumption is attacked.

## Trust boundary (the spine)

```
UNTRUSTED (client / device)                 │  TRUSTED (server / engine)
  raw biometrics, device, client claims     │  verifies signatures, recomputes
  step-up provider (holds PRIVATE keys)     │  holds only PUBLIC verify keys
──────────────────────────────────────────────────────────────────────────
The server NEVER believes a security claim the client makes about itself.
```

## STRIDE

| Threat | Attack | Control | Where | Proof |
|---|---|---|---|---|
| **Spoofing** | Forge a step-up success (`?verified=true`) | Ed25519-signed verifier assertion; engine holds only the public key | `verifier.py`, `main.py::step_up_outcome` | `test_ks1_*` |
| **Spoofing** | Client sends `behavior_score=0.99` | Field removed; behaviour trusted only via signed + device-attested assertion | `attestation.py`, `schemas.py` | `test_ks2_*` |
| **Spoofing** | Spoofed / replayed events | Scoped API keys + per-caller idempotency bound to payload fingerprint | `auth.py`, `main.py::ingest_event` | `test_ks4_*`, `test_r2_idempotency_*` |
| **Tampering** | Edit/recompute the audit chain | SHA-256 + HMAC keyed chain; out-of-band head checkpoint catches truncation | `audit.py` | `test_x_keyed_audit_*`, `test_r2_audit_truncation_*` |
| **Tampering** | Swap the model artifact (→ RCE / silent miss) | Out-of-band pinned SHA-256 (`PRAMAAN_MODEL_SHA256`) verified before `joblib.load` | `model_loader.py` | `test_r2_pinned_model_*`, `test_ks3_tampered_*` |
| **Tampering** | Poison the per-identity profile (low-and-slow) | Commit only on ALLOW + capped recovery + drift detection (sticky review) | `risk_engine.py`, `drift.py` | `test_ks9_*`, `results/adversarial/low_and_slow.png` |
| **Repudiation** | Deny an action happened | Tamper-evident, keyed, checkpointed audit chain (token refs only) | `audit.py`, `main.py` | `test_ks6_audit_chain_still_verifies_after_erasure` |
| **Information disclosure** | Enumerate identities (IDOR) | `GET /v1/identity/{id}` gated behind `identity:read` SOC scope, minimal fields | `main.py::identity_state` | `test_ks4_idor_*` |
| **Information disclosure** | Use reason codes / trust score as an oracle | Client gets a generic decision only; reasons + SHAP + trust on the SOC plane | `schemas.py`, `risk_engine.py` | `test_ks8_*`, `test_p5_*` |
| **Information disclosure** | Reconstruct an individual from training exports | Gaussian DP on the aggregation export, ε≈1.0 (δ=1e-5), accounted | `dp_export.py` | `results/privacy_budget.json`, `test_dp_export_*` |
| **Information disclosure** | Recover erased PII from the immutable chain | Crypto-shredding: destroy the per-identity key → ciphertext unreadable | `keystore.py`, `main.py::erase_identity` | `test_ks6_*` |
| **Denial of service** | Step-up bombing / MFA fatigue | Per-identity accepted-step-up cap; failures tracked separately (no lockout) | `main.py::step_up_outcome` | `test_x_stepup_bombing_*`, `test_r2_failed_stepups_*` |
| **Denial of service** | Memory exhaustion via unique keys | Bounded LRU idempotency cache; empty rate buckets reclaimed | `main.py` | (R2 fix) |
| **Denial of service** | Engine/model down → fraud window or outage | Explicit fail-closed (sensitive) / fail-open (routine) resilience policy | `resilience.py` | `test_x_degraded_*` |
| **Elevation of privilege** | Unauthenticated / wrong-scope access to privileged ops | AuthN + per-scope authZ on every endpoint; constant-time key compare | `auth.py`, `config.py` | `test_ks4_*` |
| **Elevation of privilege** | Default/shared secret in prod | No defaults; fail-loud on missing secret; CORS wildcard banned | `config.py`, `privacy.py` | `test_ks4d_*` |

## Attack tree — "attacker performs a fraudulent high-value transfer"

```
GOAL: move ₹95k from a victim account
├── A. Defeat the step-up                          → BLOCKED: signed Ed25519 assertion only
│     └── A1. Replay a captured assertion          → BLOCKED: single-use nonce + exp
│     └── A2. Forge with the engine's key          → BLOCKED: engine holds only the PUBLIC key
├── B. Look like the owner                          → BLOCKED: behaviour needs signed+attested
│     └── B1. Send behavior_score=0.99             → BLOCKED: field removed (extra="forbid")
│     └── B2. Reuse a device's attestation         → BLOCKED: device+identity binding
├── C. Poison the profile slowly (low-and-slow)     → BLOCKED: drift review + capped recovery
│     └── C1. Creep the amount baseline up         → BLOCKED: drift trips sticky secondary review
├── D. Impersonate from a new location              → RAISED: impossible-travel + new_geo
├── E. Attack the service directly                  → BLOCKED: scoped auth, no IDOR, no defaults
│     └── E1. Enumerate identities                 → BLOCKED: identity:read SOC scope
│     └── E2. Swap the model to score 0 risk       → BLOCKED: out-of-band pinned digest
│     └── E3. Launder via idempotency replay       → BLOCKED: payload-bound idempotency
└── F. Read the detector to iterate to ALLOW        → BLOCKED: generic client decision only
```

Residual / accepted (see `SECURITY_HARDENING.md`):
multi-pod nonce/idempotency need Redis backing for horizontal scale; impossible-travel
offline lift needs the full ordered RBA stream; live-deploy hardening is
