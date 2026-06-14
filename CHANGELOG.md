# Changelog

All notable changes to PRAMAAN are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project aims to
follow semantic versioning.

## [1.0.0] - 2026-06-14

First public release prepared for the PSB Hackathon Series 2026 (Bank of Baroda,
Cybersecurity & Fraud track).

### Added
- Data pipeline: dataset inventory, download script with SHA-256 verification,
  feature engineering with temporal splits and leakage guards.
- Per-detection models and evaluation across five public datasets (RBA, PaySim,
  IEEE-CIS, CMU keystroke, CERT), reported as PR-AUC, ROC-AUC, and
  detection-vs-step-up curves.
- Hybrid risk engine with a continuous Trust Score (0–1000) and an
  ALLOW / STEP-UP / BLOCK policy orchestrator with reason codes.
- SHA-256 + HMAC hash-chained, tamper-evident audit ledger with a signed head
  checkpoint.
- FastAPI service and a live dashboard.

### Security
- Step-up outcomes accepted only as Ed25519-signed verifier assertions; the
  engine holds public verify keys only and can never mint one.
- Behavioural signals trusted only via device-attested, signed assertions;
  removed the client-supplied behaviour score.
- Scoped API-key auth on every endpoint; removed the identity-lookup IDOR;
  CORS allowlist and secrets sourced from env/KMS with no shipped defaults.
- Serving model loaded from a real, SHA-256-pinned artifact trained on real
  data; synthetic fallback is env-gated and stamped.

### Privacy
- Differential-privacy export path (RDP accountant, ε ≈ 1.1, δ = 1e-5).
- Right-to-erasure via crypto-shredding that keeps the audit chain verifiable.

### Robustness
- Drift detection (windowed shift + CUSUM) and capped trust recovery to defeat
  low-and-slow baseline poisoning.

### Infrastructure
- docker-compose topology (Kafka + Redis + Postgres + scoring pods + verifier +
  ingress) with an end-to-end smoke test.
- CI: 84 tests, ruff lint, model-integrity check, and a headline drift-gate.

[1.0.0]: https://github.com/ayushtiwary-ops/BankOfBaroda-IIT-Gn/releases/tag/v1.0.0
