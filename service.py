"""FastAPI app for the TimesFM forecast service.

Auth, request validation, and the response contract live here. Inference
itself is entirely delegated to `ENGINE` (model.ForecastEngine) — this
module never imports timesfm/torch, so contract tests run without them.

Routes:

    POST /v1/forecast   (Bearer auth)
    GET  /v1/health     (open)

The request/response contract implemented below is standalone and
consumer-agnostic — any caller over HTTP can use it; this repo never
imports or references a consumer's codebase.
"""
from __future__ import annotations

import asyncio
import math
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import Depends, FastAPI, Header, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator, model_validator

import settings
from model import ENGINE, CovariatesUnsupported, SeriesIn

# ---------------------------------------------------------------------------
# Request schema
# ---------------------------------------------------------------------------


class SeriesRequest(BaseModel):
    id: str
    values: list[float]
    freq: str = "D"

    @field_validator("values")
    @classmethod
    def _values_are_well_formed(cls, v: list[float]) -> list[float]:
        if len(v) < 1 or len(v) > 4096:
            raise ValueError("values must contain between 1 and 4096 points")
        if not all(math.isfinite(x) for x in v):
            raise ValueError("values must contain only finite numbers (no NaN/inf)")
        return v

    @field_validator("freq")
    @classmethod
    def _freq_is_supported(cls, v: str) -> str:
        if v != "D":
            raise ValueError('freq must be "D" (no other frequency is supported yet)')
        return v


class Covariates(BaseModel):
    known_future: dict[str, list[float]] | None = None
    past_only: dict[str, list[float]] | None = None


class ForecastRequest(BaseModel):
    series: list[SeriesRequest]
    horizon: int
    quantiles: bool = True
    covariates: Covariates | None = None

    @field_validator("series")
    @classmethod
    def _series_count_bounds(cls, v: list[SeriesRequest]) -> list[SeriesRequest]:
        if len(v) < 1 or len(v) > 8:
            raise ValueError("series must contain between 1 and 8 entries")
        return v

    @field_validator("horizon")
    @classmethod
    def _horizon_bounds(cls, v: int) -> int:
        if v < 1 or v > 64:
            raise ValueError("horizon must be between 1 and 64")
        return v

    @model_validator(mode="after")
    def _covariate_rules(self) -> "ForecastRequest":
        if self.covariates is None:
            return self
        if len(self.series) > 1:
            raise ValueError("covariates with multiple series unsupported in v1")

        # Single series: the one covariates block unambiguously refers to it.
        s = self.series[0]
        n = len(s.values)
        if self.covariates.known_future:
            expected = n + self.horizon
            for name, arr in self.covariates.known_future.items():
                if len(arr) != expected:
                    raise ValueError(
                        f"covariates.known_future.{name} must have length "
                        f"{expected} (len(values)+horizon) for series "
                        f"{s.id!r}, got {len(arr)}"
                    )
        if self.covariates.past_only:
            for name, arr in self.covariates.past_only.items():
                if len(arr) != n:
                    raise ValueError(
                        f"covariates.past_only.{name} must have length "
                        f"{n} (len(values)) for series {s.id!r}, got "
                        f"{len(arr)}"
                    )
        return self


# ---------------------------------------------------------------------------
# Response schema
# ---------------------------------------------------------------------------


class SeriesResult(BaseModel):
    id: str
    mean: list[float]
    # None when the request set quantiles=false; response_model_exclude_none
    # on the route drops the key entirely in that case (mean-only response).
    quantiles: dict[str, list[float]] | None = None


class ForecastResponse(BaseModel):
    model_version: str
    generated_at: str
    results: list[SeriesResult]


# ---------------------------------------------------------------------------
# Auth — errors below need a bare {"error": ...} body (no "detail"
# wrapper), per the API contract, so they get their own exception classes
# and handlers rather than raising HTTPException.
# ---------------------------------------------------------------------------


class ServiceNotConfigured(Exception):
    """No FORECAST_API_KEY is set — fail closed on every /v1/* route."""


class Unauthorized(Exception):
    """Missing or wrong bearer token."""


class ModelLoading(Exception):
    """Engine not loaded yet."""


async def require_api_key(authorization: str | None = Header(default=None)) -> None:
    if not settings.API_KEY:
        raise ServiceNotConfigured()
    if not authorization or not authorization.startswith("Bearer "):
        raise Unauthorized()
    token = authorization[len("Bearer "):].strip()
    if token != settings.API_KEY:
        raise Unauthorized()


@asynccontextmanager
async def lifespan(app: FastAPI):
    if settings.WARM_AT_BOOT:
        await ENGINE.ensure_loaded()
    yield


app = FastAPI(title="timesfm-forecast-service", lifespan=lifespan)

# Self-healing load: with FORECAST_WARM_AT_BOOT unset, nothing but the
# lifespan hook above ever calls ENGINE.ensure_loaded(), so a service that
# started cold would 503 "model loading" forever. When a forecast request
# finds the engine unloaded, schedule a background load and still return the
# 503 immediately below (self-healing without blocking the request).
# ENGINE.ensure_loaded()'s own double-checked lock already makes duplicate
# calls harmless, but a bare `asyncio.create_task(...)` with no reference
# held is eligible for GC before it runs — this module-level ref plus the
# `.done()` check avoids both the GC risk and spawning a pile of redundant
# tasks while one load is already in flight.
_warmup_task: asyncio.Task | None = None


def _schedule_warmup() -> None:
    global _warmup_task
    if _warmup_task is None or _warmup_task.done():
        _warmup_task = asyncio.create_task(ENGINE.ensure_loaded())


@app.exception_handler(ServiceNotConfigured)
async def _service_not_configured_handler(
    request: Request, exc: ServiceNotConfigured
) -> JSONResponse:
    return JSONResponse(status_code=503, content={"error": "service not configured"})


@app.exception_handler(Unauthorized)
async def _unauthorized_handler(request: Request, exc: Unauthorized) -> JSONResponse:
    return JSONResponse(status_code=401, content={"error": "unauthorized"})


@app.exception_handler(ModelLoading)
async def _model_loading_handler(request: Request, exc: ModelLoading) -> JSONResponse:
    return JSONResponse(status_code=503, content={"error": "model loading"})


@app.exception_handler(CovariatesUnsupported)
async def _covariates_unsupported_handler(
    request: Request, exc: CovariatesUnsupported
) -> JSONResponse:
    # Single-series covariates pass request validation (only multi-series +
    # covariates is rejected there — see ForecastRequest._covariate_rules)
    # but the real engine does not wire covariates into the model in v1
    # (docs/feasibility.md). Raised by model.ForecastEngine.forecast();
    # surfaced here as a 422 with the same bare {"error": ...} shape as the
    # other engine/service-level errors above.
    return JSONResponse(
        status_code=422, content={"error": "covariates unsupported in v1"}
    )


def _sanitize_non_finite(obj):
    """Replace non-finite floats with their repr so json.dumps can encode them.

    A rejected NaN/Infinity value gets echoed back verbatim inside
    pydantic's error detail (the "input" that failed validation).
    Starlette's JSONResponse renders with allow_nan=False, so left alone
    that echo would blow up the very 422 response meant to report it.
    """
    if isinstance(obj, float) and not math.isfinite(obj):
        return str(obj)
    if isinstance(obj, dict):
        return {k: _sanitize_non_finite(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_non_finite(v) for v in obj]
    return obj


@app.exception_handler(RequestValidationError)
async def _validation_error_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    errors = _sanitize_non_finite(jsonable_encoder(exc.errors()))
    return JSONResponse(status_code=422, content={"detail": errors})


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/v1/health")
async def health() -> dict:
    return {
        "ok": True,
        "model_loaded": ENGINE.loaded(),
        "queue_depth": ENGINE.queue_depth,
    }


@app.post(
    "/v1/forecast",
    dependencies=[Depends(require_api_key)],
    response_model=ForecastResponse,
    response_model_exclude_none=True,
)
async def forecast(body: ForecastRequest) -> ForecastResponse:
    if not ENGINE.loaded():
        _schedule_warmup()
        raise ModelLoading()

    series_in = [SeriesIn(id=s.id, values=s.values, freq=s.freq) for s in body.series]
    covariates = body.covariates.model_dump() if body.covariates is not None else None
    results = await ENGINE.forecast(series_in, body.horizon, covariates)

    return ForecastResponse(
        model_version=ENGINE.model_version,
        generated_at=datetime.now(timezone.utc).isoformat(),
        results=[
            SeriesResult(
                id=r.id,
                mean=list(r.mean),
                quantiles=dict(r.quantiles) if body.quantiles else None,
            )
            for r in results
        ],
    )
