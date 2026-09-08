# timesfm-service

A standalone FastAPI microservice wrapping Google TimesFM 3 (330M): POST a
numeric daily series, get back a mean forecast plus P10–P90 quantiles. It
knows nothing about any caller's domain — callers supply the series data,
this service never fetches market data or anything else itself.

This is Component 1 of a larger design (see the architecture doc referenced
in `docs/plans/2026-09-08-timesfm-service.md`); the FinSage integration that
consumes this service is a separate, later repo/plan. **This repo never
imports or references that other codebase.**

## Status

As of this task, `model.py` ships the `ForecastEngine` interface plus a stub
that reports itself unloaded and raises `NotLoaded` — there is no real
model wired up yet (that's Task 3, transcribed from `docs/feasibility.md`).
The HTTP contract (auth, validation, response shape) implemented in
`service.py` is real and fully tested against that stub via monkeypatching.

## API

```
POST /v1/forecast   (Authorization: Bearer $FORECAST_API_KEY)
GET  /v1/health     (open, no auth)
```

See `docs/superpowers/specs/2026-09-08-timesfm-forecast-service-design.md`
(Component 1) in the finsage repo for the full request/response contract —
that document is authoritative; this repo just implements it.

## Run

```
python -m venv .venv
.venv/bin/pip install -r requirements.txt
# real inference needs the extra deps documented in deploy.md /
# docs/feasibility.md — NOT installed by requirements.txt alone.

export FORECAST_API_KEY=some-secret
export FORECAST_PORT=8090          # optional, defaults to 8090
export FORECAST_WARM_AT_BOOT=1     # optional, defaults to 0

.venv/bin/uvicorn service:app --host 127.0.0.1 --port 8090
```

Without `FORECAST_API_KEY` set (or set blank), every `/v1/*` route except
`/v1/health` fails closed with `503 {"error": "service not configured"}`.

## Test

```
python -m pytest tests/ -q
```

`tests/test_contract.py` covers the full HTTP contract — auth, every
validation rule, and the response shape — with `ENGINE` monkeypatched to a
fake. It passes with no `timesfm`/`torch` installed. A later, separately
marked slow test suite (Task 3) exercises the real model and is skipped
when that package isn't present.

## LICENSE NOTE

(Verbatim from `docs/feasibility.md`, the license section on record from
the feasibility gate — read that file for the full context.)

> TimesFM 3.0 **weights** are under the TimesFM **Non-Commercial License v1.0**
> (no commercial/production use); the code is Apache-2.0. Owner decision
> 2026-09-08: use 3.0 now (personal, pre-revenue research posture) and purchase
> a commercial license — or swap to Apache-licensed TimesFM 2.5 — **before any
> commercialization**. The service contract is model-agnostic to keep that swap
> a checkpoint change. This item belongs on any due-diligence checklist.
