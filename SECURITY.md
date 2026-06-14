# Security Policy

PRAMAAN is a security project, so we take vulnerabilities in it seriously.

## Reporting a vulnerability

Please **do not** open a public issue for a security vulnerability. Instead,
report it privately:

- Use GitHub's **"Report a vulnerability"** (Security → Advisories) on this
  repository, or
- email the maintainers (see the team section of the README).

Include, where possible: a description of the issue, the affected file or
endpoint, reproduction steps or a proof of concept, and the impact you expect.

We aim to acknowledge a report within **72 hours** and to provide a remediation
plan or fix timeline within **7 days**. Please give us a reasonable window to
address the issue before any public disclosure.

## Scope

This repository is a hackathon prototype. The trust boundary, signed step-up
assertions, scoped auth, audit chain, differential-privacy export, and
crypto-shredding erasure are implemented and tested (see
[docs/SECURITY_HARDENING.md](docs/SECURITY_HARDENING.md) and
[docs/THREAT_MODEL.md](docs/THREAT_MODEL.md)). The demo secrets in
`infra/docker-compose.yml` are clearly labelled non-production values; production
deployments must inject real secrets from a KMS/HSM.

## Out of scope

- Findings that require physical access to a developer machine.
- The demo/placeholder secrets committed for local `docker compose` runs (these
  are intentionally non-secret and documented as such).
