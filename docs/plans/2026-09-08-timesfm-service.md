# TimesFM Forecast Service Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A standalone FastAPI microservice wrapping Google TimesFM 3 (330M): POST a numeric series, get mean + P10–P90 quantile forecasts. Deployed on the Oracle box (127.0.0.1:8090, own venv, systemd). No FinSage code in this repo — FinSage integration is a separate later plan.

**Architecture:** Spec (authoritative): `../finsage/docs/superpowers/specs/2026-09-08-timesfm-forecast-service-design.md` — Component 1 only. Thin service: `settings.py` (env) → `model.py` (load/predict, single-flight) → `service.py` (FastAPI, auth, validation). Contract tests run with the model mocked; one `@pytest.mark.slow` test uses the real model.

**Tech Stack:** Python 3.11+/3.12, FastAPI, uvicorn, httpx (tests), torch CPU (aarch64 on the server), `timesfm` package, pytest.

## Global Constraints

- This repo NEVER imports or references FinSage. Callers supply series data.
- Auth: every `/v1/*` route requires `Authorization: Bearer $FORECAST_API_KEY`; `/v1/health` is open. Missing/blank key in env → all `/v1/*` return 503 "service not configured" (fail closed), health still works.
- Input validation is strict: reject NaN/inf/empty series, >4096 points, horizon outside 1..64, freq != "D", >8 series — all 422 with a specific message. No silent imputation.
- Single-flight inference: an asyncio lock so only one model call runs at a time; report `queue_depth` in health.
- Tests: `python -m pytest tests/ -q` from repo root; contract tests must pass WITHOUT the model installed (model.py mocked). The slow real-model test is opt-in (`-m slow`) and skipped when the model/package is absent.
- Conventional commits; never push (coordinator pushes).
- Server work (venv, systemd) happens over SSH by the coordinator, not by task implementers — implementer tasks produce code + docs only, EXCEPT Task 1 which runs ON the server via the coordinator's SSH (see Task 1).

---

### Task 1: Feasibility gate + API discovery (ON THE SERVER, coordinator-assisted)

**This task runs first and can ABORT the plan.** Deliverable: `docs/feasibility.md` in this repo recording hard numbers, and a working reference snippet.

**Steps:**
- [ ] **Step 1: Discover the current TimesFM 3 install + API.** Check the official sources: https://github.com/google-research/timesfm (README install instructions for the PyTorch TimesFM 3 checkpoint) and the HuggingFace model card (search "timesfm-3" on huggingface.co/google). Record: exact pip install spec, exact checkpoint id, the license (expect Apache-2.0 — record what the card actually says), and the minimal Python snippet the README gives for loading + forecasting with quantiles (v3 API — do NOT assume the v1/v2 `TimesFm(hparams…)` API survives).
- [ ] **Step 2: Server venv + install.** On the Oracle box (3 OCPU ARM / 18GB): `python3.12 -m venv ~/timesfm-service/.venv && ~/timesfm-service/.venv/bin/pip install <discovered spec> fastapi uvicorn pytest httpx`. Record install size (`du -sh ~/timesfm-service/.venv`) and any aarch64 wheel issues.
- [ ] **Step 3: One real forecast, timed.** Adapt the discovered snippet: 512 synthetic points (sine + noise), horizon 20, quantiles on. Run twice in one process; record cold wall time, warm wall time, peak RSS (`/usr/bin/time -v` or resource module), checkpoint disk size.
- [ ] **Step 4: Verdict against abort criteria.** ABORT (stop the plan, report to owner) if: warm inference > 30s, or RSS > 6GB, or no working aarch64 torch/timesfm, or the license forbids our use. Otherwise write `docs/feasibility.md` with all numbers + the working snippet + install spec, and commit: `docs: feasibility gate results and TimesFM 3 reference snippet`

### Task 2: Service scaffold — settings, app, auth, validation, contract tests

**Files:**
- Create: `settings.py`, `service.py`, `model.py` (stub interface only this task), `requirements.txt`, `tests/test_contract.py`, `.gitignore`, `README.md`

**Interfaces (produced):**
```python
# settings.py
API_KEY: str | None       # os.getenv("FORECAST_API_KEY") or None
PORT: int                 # os.getenv("FORECAST_PORT", "8090")
WARM_AT_BOOT: bool        # os.getenv("FORECAST_WARM_AT_BOOT", "0") == "1"

# model.py (this task: interface + NotLoaded stub; real impl Task 3)
class ForecastEngine:
    model_version: str
    def loaded(self) -> bool
    async def ensure_loaded(self) -> None          # heavy load in asyncio.to_thread
    async def forecast(self, series: list[SeriesIn], horizon: int,
                       covariates: dict | None) -> list[SeriesOut]
    @property
    def queue_depth(self) -> int
ENGINE = ForecastEngine()  # module singleton

# service.py — FastAPI app named `app`
POST /v1/forecast   (auth)  -> spec contract exactly (see spec §Component 1)
GET  /v1/health     (open)  -> {"ok": true, "model_loaded": bool, "queue_depth": int}
```
Pydantic request models enforce: 1..8 series, each 1..4096 finite floats, freq=="D", horizon 1..64; covariate arrays length-checked (known_future = len+horizon; past_only = len) — violations 422 with field-specific messages. Auth via dependency: no/blank env key → 503 {"error": "service not configured"}; wrong key → 401.

- [ ] **Step 1: Write failing contract tests** (`tests/test_contract.py`, TestClient, ENGINE mocked via monkeypatch): health open + shape; 401 wrong key; 503 when env key blank; 422 for each validation rule (NaN, too long, bad freq, bad horizon, covariate length mismatch); happy path returns the mocked engine's quantiles in the contract shape; 503 when engine not loaded.
- [ ] **Step 2:** Run: `python -m pytest tests/ -q` — expect FAIL (nothing exists).
- [ ] **Step 3:** Implement settings.py, model.py stub, service.py.
- [ ] **Step 4:** `python -m pytest tests/ -q` — PASS.
- [ ] **Step 5:** Commit: `feat: service scaffold — contract, auth, validation (engine stubbed)`

### Task 3: Real model integration

**Files:**
- Modify: `model.py`
- Create: `tests/test_model_real.py` (all `@pytest.mark.slow`, auto-skip if timesfm not importable), `pytest.ini` (markers, slow deselected by default)

**Interfaces (consumes):** the WORKING SNIPPET and install spec from `docs/feasibility.md` — that file is the source of truth for the TimesFM 3 API; transcribe it, do not improvise.

- [ ] **Step 1:** Implement `ForecastEngine` for real: lazy `ensure_loaded()` running the checkpoint load in `asyncio.to_thread`; `forecast()` acquires the single-flight `asyncio.Lock`, increments/decrements `_queue_depth`, runs predict in `to_thread`, maps output to the contract (mean + p10..p90 lists of exactly `horizon` floats each). Covariates: pass through to the model per the feasibility snippet IF v3's API exposes them there; if the discovered API takes covariates differently (or Task 1 found them awkward), raise 422 "covariates unsupported in v1" and note it in README — DO NOT fake covariate support.
- [ ] **Step 2:** Slow tests: real load; real forecast on 512 sine points → assert p10 <= mean <= p90 elementwise, correct lengths, warm second call < 30s.
- [ ] **Step 3:** Contract tests still green with engine mocked: `python -m pytest tests/ -q`.
- [ ] **Step 4:** Commit: `feat: TimesFM 3 engine — lazy load, single-flight, quantile mapping`

### Task 4: Deployment artifacts + runbook

**Files:**
- Create: `deploy/timesfm-forecast.service` (systemd unit: user ubuntu, WorkingDirectory ~/timesfm-service, ExecStart .venv/bin/uvicorn service:app --host 127.0.0.1 --port 8090, Restart=on-failure, MemoryMax=6G, EnvironmentFile=~/timesfm-service/.env), `deploy.md` (venv create, pip install -r requirements.txt + the torch/timesfm spec from feasibility.md, .env keys incl. generated FORECAST_API_KEY, systemctl enable+start, curl smoke commands), `.env.example`
- [ ] **Step 1:** Write the unit + deploy.md + .env.example (FORECAST_API_KEY=, FORECAST_PORT=8090, FORECAST_WARM_AT_BOOT=1).
- [ ] **Step 2:** `python -m pytest tests/ -q` — still green.
- [ ] **Step 3:** Commit: `feat: systemd unit and deployment runbook`

### Coordinator post-plan (not tasks): push repo; on the server follow deploy.md; live verification = health shows model_loaded after warm-up, one real authorized /v1/forecast on NVDA daily closes (coordinator supplies bars from FinSage data purely as test input) returns sane corridors in seconds; hand the owner the API key + example curl; owner plays with it → then the FinSage-integration plan (spec Components 2–4) gets written.
