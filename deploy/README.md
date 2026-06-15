# PRAMAAN - Deploy & Attack Surface

> Two things run here: (A) the **full executable topology** locally
> (`docker compose`), and (B) a **single deployable demo** (the synchronous
> API + dashboard) for a public, attackable URL. The deploy step is
> **founder-signed** - it publishes to the internet under the founder's
> Fly/Render account, so the commands are provided but NOT auto-run.

---

## A. Run the full topology locally (executable architecture)

```bash
cd pramaan
docker compose -f infra/docker-compose.yml up --build -d   # Kafka+Redis+Postgres+pods+verifier+ingress
python scripts/e2e_smoke.py                                 # events flow Kafka→pods→decision→audit
docker compose -f infra/docker-compose.yml down -v
```
Ingress: `http://localhost:8090` · Verifier: `http://localhost:8081` ·
Postgres: `localhost:5433`. The e2e smoke asserts the decision lands in Postgres,
the keyed audit chain verifies, and the verifier mints assertions the pods can't.

## B. Deploy the public demo (single app + dashboard)

The deployable demo is `app.main:app` - the synchronous API the dashboard drives
(immediate client + SOC view). It runs in **prod mode with the REAL model**;
the secrets below are **clearly-labelled demo values** (rotate for anything real).

### Fly.io (simplest)
```bash
cd pramaan
fly launch --no-deploy --copy-config --dockerfile backend/Dockerfile   # uses deploy/fly.toml
fly secrets set \
  PRAMAAN_MODE=prod \
  PRAMAAN_EDGE_SECRET="$(openssl rand -hex 24)" \
  PRAMAAN_AUDIT_KEY="$(openssl rand -hex 24)" \
  PRAMAAN_STEPUP_PUBKEY=1zFJQhbWxZTgAfj2O7qbNPcCQn3LXMVdY_4s1W7Nh9A \
  PRAMAAN_ATTEST_PUBKEY=1zFJQhbWxZTgAfj2O7qbNPcCQn3LXMVdY_4s1W7Nh9A \
  PRAMAAN_BEHAVIOR_PUBKEY=1zFJQhbWxZTgAfj2O7qbNPcCQn3LXMVdY_4s1W7Nh9A \
  PRAMAAN_API_KEYS='{"demo-key":["events:write","audit:read","identity:read","stepup:write","identity:erase"]}' \
  PRAMAAN_CORS_ORIGINS=https://pramaan.fly.dev
fly deploy
# → https://pramaan.fly.dev   (rate-limit at the edge: see fly.toml [http_service] concurrency)
```

### Render (alternative)
`deploy/render.yaml` is a one-file blueprint - `render blueprint launch` (or
connect the repo in the Render dashboard) builds `backend/Dockerfile` and runs
the same command. Set the same env in the Render dashboard.

## C. Scan-and-attack QR (for the slide)

```bash
python deploy/make_qr.py https://pramaan.fly.dev   # → deploy/demo_qr.png
```
On the slide: "Scan this and attack our demo right now." A judge can try
`POST /v1/stepup/<id>?verified=true` (→ rejected), a spoofed `behavior_score`
(→ 422), an event replay (→ idempotency-bound), or enumerate `/v1/identity/{id}`
(→ 401 without the SOC scope). Each rejection is explained on the SOC plane.

## D. Demo-safe scoping

- Demo runs `PRAMAAN_MODE=prod` with the **real RBA model** but **demo secrets**.
- Only `demo-key` is issued (all scopes) so judges can exercise every endpoint.
- Rate-limit at the platform edge (Fly `concurrency`, Render plan) + the built-in
  step-up-bombing limiter and idempotency cache.
- No real customer data is ever loaded (Hard Rule #1); replays use the public
  sample under `data/samples/`.

> **Founder sign-off required before deploy** - `fly deploy` / `render` publishes
> publicly under your account. Nothing here was auto-deployed.
