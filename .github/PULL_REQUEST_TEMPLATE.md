<!-- Keep PRs focused and reviewable. Delete sections that do not apply. -->

## What and why

Briefly describe the change and the problem it solves.

## How it was tested

- [ ] `python -m pytest -q` passes
- [ ] `python scripts/check_headline.py` gate is green (no metric drift)
- [ ] `ruff check` is clean
- [ ] If infra changed: `docker compose -f infra/docker-compose.yml config -q`

## Checklist

- [ ] No secrets, tokens, or private keys added
- [ ] No raw datasets or files > 25 MB committed (samples stay < 5 MB)
- [ ] Any new metric claim is backed by a committed artifact under `results/`
- [ ] Docs updated if behaviour or interfaces changed
