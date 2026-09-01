# Security Policy

## Reporting a Vulnerability

If you discover a security issue in this repository (including accidentally exposed API tokens), please open a private advisory or contact the maintainers. Do **not** open a public issue that includes live credentials.

## Secrets

This project talks to third-party APIs (MinerU, Paddle AI Studio). Treat tokens as secrets:

1. Copy `lightweight_pipeline/config.example.yaml` → `config.yaml` and keep `config.yaml` **out of git**.
2. Prefer environment variables: `MINERU_TOKEN`, `AISTUDIO_TOKEN` (optional: `AISTUDIO_LAYOUT_TOKEN`, `AISTUDIO_SPOTTING_TOKEN`).
3. Never commit `work/`, `output/`, or real document samples that contain private data.
4. If a token was ever committed or shared, **revoke and rotate it** immediately on the provider console.

## Scope

Local GPU / high-precision deployment guides under `docs/deploy/` are documentation only and do not ship model weights or private datasets.
