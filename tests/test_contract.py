"""Contract tests for the forecast service.

These tests exercise service.py's HTTP contract only: auth, validation,
and response shape. The real TimesFM engine is never touched — ENGINE is
monkeypatched with a fake that implements the same interface
(model.ForecastEngine). This file must pass with no `timesfm`/`torch`
installed.
"""
from __future__ import annotations

import json
import math
from datetime import datetime

import pytest
from fastapi.testclient import TestClient

import service
import settings
from model import SeriesOut


@pytest.fixture()
def client() -> TestClient:
    return TestClient(service.app)


@pytest.fixture()
def configured(monkeypatch: pytest.MonkeyPatch) -> str:
    """Configure a known API key and return it."""
    key = "test-secret-key"
    monkeypatch.setattr(settings, "API_KEY", key)
    return key


class FakeEngine:
    """A minimal stand-in for model.ForecastEngine, used via monkeypatch."""

    model_version = "timesfm-3-330m-fake"

    def __init__(self, is_loaded: bool = True, queue_depth: int = 0) -> None:
        self._loaded = is_loaded
        self._queue_depth = queue_depth
        self.last_call: dict | None = None

    def loaded(self) -> bool:
        return self._loaded

    async def ensure_loaded(self) -> None:
        self._loaded = True

    async def forecast(self, series, horizon, covariates=None):
        self.last_call = {
            "series": series,
            "horizon": horizon,
            "covariates": covariates,
        }
        results = []
        for s in series:
            mean = [float(i) for i in range(horizon)]
            quantiles = {
                f"p{p}": [float(i) + p / 100 for i in range(horizon)]
                for p in range(10, 91, 10)
            }
            results.append(SeriesOut(id=s.id, mean=mean, quantiles=quantiles))
        return results

    @property
    def queue_depth(self) -> int:
        return self._queue_depth


def valid_payload(**overrides) -> dict:
    payload = {
        "series": [{"id": "NVDA", "values": [1.0, 2.0, 3.0, 4.0, 5.0], "freq": "D"}],
        "horizon": 5,
    }
    payload.update(overrides)
    return payload


# ---------------------------------------------------------------------------
# Health — open, no auth required
# ---------------------------------------------------------------------------


def test_health_is_open_and_has_contract_shape(client: TestClient) -> None:
    resp = client.get("/v1/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body.keys() == {"ok", "model_loaded", "queue_depth"}
    assert body["ok"] is True
    assert isinstance(body["model_loaded"], bool)
    assert isinstance(body["queue_depth"], int)


def test_health_reflects_engine_state(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeEngine(is_loaded=True, queue_depth=3)
    monkeypatch.setattr(service, "ENGINE", fake)
    resp = client.get("/v1/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body == {"ok": True, "model_loaded": True, "queue_depth": 3}


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------


def test_forecast_503_when_service_not_configured(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "API_KEY", None)
    resp = client.post(
        "/v1/forecast",
        json=valid_payload(),
        headers={"Authorization": "Bearer whatever"},
    )
    assert resp.status_code == 503
    assert resp.json() == {"error": "service not configured"}


def test_forecast_503_when_service_not_configured_blank_key(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "API_KEY", "")
    resp = client.post(
        "/v1/forecast",
        json=valid_payload(),
        headers={"Authorization": "Bearer whatever"},
    )
    assert resp.status_code == 503
    assert resp.json() == {"error": "service not configured"}


def test_forecast_401_wrong_key(client: TestClient, configured: str) -> None:
    resp = client.post(
        "/v1/forecast",
        json=valid_payload(),
        headers={"Authorization": "Bearer not-the-right-key"},
    )
    assert resp.status_code == 401


def test_forecast_401_missing_header(client: TestClient, configured: str) -> None:
    resp = client.post("/v1/forecast", json=valid_payload())
    assert resp.status_code == 401


# ---------------------------------------------------------------------------
# Validation — 422s
# ---------------------------------------------------------------------------


def test_forecast_422_too_many_series(client: TestClient, configured: str) -> None:
    series = [
        {"id": f"S{i}", "values": [1.0, 2.0], "freq": "D"} for i in range(9)
    ]
    resp = client.post(
        "/v1/forecast",
        json=valid_payload(series=series),
        headers={"Authorization": f"Bearer {configured}"},
    )
    assert resp.status_code == 422


def test_forecast_422_zero_series(client: TestClient, configured: str) -> None:
    resp = client.post(
        "/v1/forecast",
        json=valid_payload(series=[]),
        headers={"Authorization": f"Bearer {configured}"},
    )
    assert resp.status_code == 422


def test_forecast_422_empty_values(client: TestClient, configured: str) -> None:
    payload = valid_payload(series=[{"id": "NVDA", "values": [], "freq": "D"}])
    resp = client.post(
        "/v1/forecast",
        json=payload,
        headers={"Authorization": f"Bearer {configured}"},
    )
    assert resp.status_code == 422


def test_forecast_422_too_many_values(client: TestClient, configured: str) -> None:
    payload = valid_payload(
        series=[{"id": "NVDA", "values": [1.0] * 4097, "freq": "D"}]
    )
    resp = client.post(
        "/v1/forecast",
        json=payload,
        headers={"Authorization": f"Bearer {configured}"},
    )
    assert resp.status_code == 422


def _post_raw_json(client: TestClient, payload: dict, token: str):
    """Post a payload that may contain NaN/Infinity.

    httpx's `json=` kwarg refuses to serialize non-finite floats
    (allow_nan=False), but the wire format (and pydantic-core's JSON
    parser) does accept the bare NaN/Infinity/-Infinity tokens Python's
    stdlib `json` module emits by default — that's exactly the shape a
    misbehaving client could send, and the one we need to reject with a
    422, not a transport-level error.
    """
    body = json.dumps(payload)  # allow_nan defaults to True here
    return client.post(
        "/v1/forecast",
        content=body,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
    )


def test_forecast_422_nan_value(client: TestClient, configured: str) -> None:
    payload = valid_payload(
        series=[{"id": "NVDA", "values": [1.0, float("nan"), 3.0], "freq": "D"}]
    )
    resp = _post_raw_json(client, payload, configured)
    assert resp.status_code == 422


def test_forecast_422_inf_value(client: TestClient, configured: str) -> None:
    payload = valid_payload(
        series=[{"id": "NVDA", "values": [1.0, float("inf"), 3.0], "freq": "D"}]
    )
    resp = _post_raw_json(client, payload, configured)
    assert resp.status_code == 422


def test_forecast_422_bad_freq(client: TestClient, configured: str) -> None:
    payload = valid_payload(
        series=[{"id": "NVDA", "values": [1.0, 2.0], "freq": "H"}]
    )
    resp = client.post(
        "/v1/forecast",
        json=payload,
        headers={"Authorization": f"Bearer {configured}"},
    )
    assert resp.status_code == 422


def test_forecast_422_horizon_too_low(client: TestClient, configured: str) -> None:
    resp = client.post(
        "/v1/forecast",
        json=valid_payload(horizon=0),
        headers={"Authorization": f"Bearer {configured}"},
    )
    assert resp.status_code == 422


def test_forecast_422_horizon_too_high(client: TestClient, configured: str) -> None:
    resp = client.post(
        "/v1/forecast",
        json=valid_payload(horizon=65),
        headers={"Authorization": f"Bearer {configured}"},
    )
    assert resp.status_code == 422


def test_forecast_422_known_future_covariate_wrong_length(
    client: TestClient, configured: str
) -> None:
    # values has 5 points, horizon 5 -> known_future must have length 10
    payload = valid_payload(
        covariates={"known_future": {"earnings_flag": [0, 0, 1, 0]}}
    )
    resp = client.post(
        "/v1/forecast",
        json=payload,
        headers={"Authorization": f"Bearer {configured}"},
    )
    assert resp.status_code == 422


def test_forecast_422_past_only_covariate_wrong_length(
    client: TestClient, configured: str
) -> None:
    # values has 5 points -> past_only must have length 5
    payload = valid_payload(covariates={"past_only": {"volume_z": [0.1, 0.2]}})
    resp = client.post(
        "/v1/forecast",
        json=payload,
        headers={"Authorization": f"Bearer {configured}"},
    )
    assert resp.status_code == 422


def test_forecast_covariates_correct_length_pass_validation(
    client: TestClient, configured: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeEngine(is_loaded=True)
    monkeypatch.setattr(service, "ENGINE", fake)
    payload = valid_payload(
        covariates={
            "known_future": {"earnings_flag": [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]},
            "past_only": {"volume_z": [0.1, 0.2, 0.3, 0.4, 0.5]},
        }
    )
    resp = client.post(
        "/v1/forecast",
        json=payload,
        headers={"Authorization": f"Bearer {configured}"},
    )
    assert resp.status_code == 200
    assert fake.last_call["covariates"] == payload["covariates"]


# ---------------------------------------------------------------------------
# Engine state
# ---------------------------------------------------------------------------


def test_forecast_503_when_engine_not_loaded(
    client: TestClient, configured: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(service, "ENGINE", FakeEngine(is_loaded=False))
    resp = client.post(
        "/v1/forecast",
        json=valid_payload(),
        headers={"Authorization": f"Bearer {configured}"},
    )
    assert resp.status_code == 503
    assert resp.json() == {"error": "model loading"}


# ---------------------------------------------------------------------------
# Happy path — exact response contract
# ---------------------------------------------------------------------------


def test_forecast_happy_path_response_shape(
    client: TestClient, configured: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeEngine(is_loaded=True, queue_depth=0)
    monkeypatch.setattr(service, "ENGINE", fake)

    payload = valid_payload(horizon=4)
    resp = client.post(
        "/v1/forecast",
        json=payload,
        headers={"Authorization": f"Bearer {configured}"},
    )
    assert resp.status_code == 200
    body = resp.json()

    assert body.keys() == {"model_version", "generated_at", "results"}
    assert body["model_version"] == fake.model_version
    # generated_at must be a parseable iso8601 timestamp
    datetime.fromisoformat(body["generated_at"])

    assert len(body["results"]) == 1
    result = body["results"][0]
    assert result.keys() == {"id", "mean", "quantiles"}
    assert result["id"] == "NVDA"
    assert len(result["mean"]) == 4
    expected_quantile_keys = {f"p{p}" for p in range(10, 91, 10)}
    assert result["quantiles"].keys() == expected_quantile_keys
    for q_values in result["quantiles"].values():
        assert len(q_values) == 4
        assert all(math.isfinite(v) for v in q_values)

    # the engine received the request in the documented shape
    assert fake.last_call["horizon"] == 4
    assert fake.last_call["series"][0].id == "NVDA"
    assert fake.last_call["series"][0].values == [1.0, 2.0, 3.0, 4.0, 5.0]
