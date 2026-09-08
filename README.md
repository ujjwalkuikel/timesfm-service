# timesfm-service

A standalone FastAPI microservice wrapping Google TimesFM 3 (330M): POST a
numeric daily series, get back a mean forecast plus P10–P90 quantiles. It
knows nothing about any caller's domain — callers supply the series data,
this service never fetches market data or anything else itself.

It is designed to be consumed by any caller over HTTP — see
`docs/plans/2026-09-08-timesfm-service.md` for the design this repo
implements. **This repo never imports or references any consumer's
codebase**; any integration that calls this service lives in its own,
separate repo/plan.

## Status

`model.py`'s `ForecastEngine` is a real implementation: it lazily loads the
TimesFM 3 checkpoint (`google/timesfm-3.0-pytorch`, transcribed from
`docs/feasibility.md`'s reference snippet) in `ensure_loaded()`, and
`forecast()` runs single-flight inference (an `asyncio.Lock`, with
`queue_depth` reporting waiters) and maps `predict_batch`'s output to the
mean + p10..p90 contract shape. `timesfm3`/`torch`/`numpy` are imported
lazily inside the methods that need them, never at module import time, so
the HTTP contract tests (`tests/test_contract.py`) run — with `ENGINE`
monkeypatched to a fake — in an environment with none of those packages
installed. A separate, slow test suite
(`tests/test_model_real.py`, `pytest.importorskip`-guarded and marked
`@pytest.mark.slow`) exercises the real checkpoint; see "Test" below.

**Covariates (v1 policy):** `docs/feasibility.md` records that
`predict_batch` accepts covariate kwargs
(`past_only_covariates`/`past_future_covariates`), but this service does
not wire them into the model in v1 — they exist in the upstream API but
were found not worth plumbing through yet. Multi-series requests with any
`covariates` block are rejected at request validation (422, `detail`
shape). Single-series requests with a `covariates` block that actually
carries data pass request validation (array lengths are still checked)
but are rejected by the engine itself with a 422
`{"error": "covariates unsupported in v1"}` — see
`model.CovariatesUnsupported` / `service.py`'s handler for it. This is a
deliberate "discovered but not wired" gap, not faked support.

## API

```
POST /v1/forecast   (Authorization: Bearer $FORECAST_API_KEY)
GET  /v1/health     (open, no auth)
```

The full request/response contract (Component 1 of the design this repo
implements) is authoritative elsewhere per
`docs/plans/2026-09-08-timesfm-service.md`; this repo just implements it —
see `service.py` for the concrete request/response shapes and validation
rules.

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
fake. It passes with no `timesfm`/`torch` installed.

`tests/test_model_real.py` exercises the real checkpoint: a real load, a
real forecast on a 512-point sine series (`p10 <= mean <= p90` elementwise,
exact `horizon` lengths), and a warm second call under 30s. Every test in
it is marked `@pytest.mark.slow`; `pytest.ini` deselects `slow` by default
(`addopts = -m "not slow"`), and the module also self-skips via
`pytest.importorskip("timesfm3")` when that package isn't installed — so
plain `python -m pytest tests/ -q` never needs it. Run the real-model
suite explicitly once `timesfm[torch]` (see `docs/feasibility.md`) is
installed:

```
python -m pytest tests/test_model_real.py -m slow -q
```

## LICENSE NOTE

(Verbatim from `docs/feasibility.md`, the license section on record from
the feasibility gate — read that file for the full context.)

> TimesFM 3.0 **weights** are under the TimesFM **Non-Commercial License v1.0**
> (no commercial/production use); the code is Apache-2.0. Owner decision
> 2026-09-08: use 3.0 now (personal, pre-revenue research posture) and purchase
> a commercial license — or swap to Apache-licensed TimesFM 2.5 — **before any
> commercialization**. The service contract is model-agnostic to keep that swap
> a checkpoint change. This item belongs on any due-diligence checklist.
