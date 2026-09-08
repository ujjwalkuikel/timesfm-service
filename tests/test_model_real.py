"""Slow tests against the real TimesFM 3 checkpoint.

Skipped automatically (`pytest.importorskip`) when `timesfm3` is not
installed, and deselected by default via pytest.ini's `-m "not slow"`. Run
explicitly with:

    python -m pytest tests/test_model_real.py -m slow -q

These are the only tests in this repo that touch the real model; see
tests/test_contract.py for the mocked HTTP contract suite that runs
everywhere, including with no `timesfm3`/`torch` installed.
"""
from __future__ import annotations

import asyncio
import math
import time

import pytest

pytest.importorskip("timesfm3")

from model import ForecastEngine, SeriesIn  # noqa: E402 (import after skip guard)

pytestmark = pytest.mark.slow


def _sine_series(id_: str = "SINE", n: int = 512) -> SeriesIn:
    values = [10.0 + math.sin(i / 10.0) for i in range(n)]
    return SeriesIn(id=id_, values=values, freq="D")


@pytest.fixture(scope="module")
def loaded_engine() -> ForecastEngine:
    """One real engine, loaded once and shared by this module's tests.

    Sharing the instance (rather than loading fresh per test) is what lets
    `test_warm_second_call_is_fast` exercise a genuinely warm second call.
    """
    engine = ForecastEngine()
    assert engine.loaded() is False
    asyncio.run(engine.ensure_loaded())
    return engine


def test_real_load_succeeds(loaded_engine: ForecastEngine) -> None:
    assert loaded_engine.loaded() is True


def test_real_forecast_sine_wave_h20(loaded_engine: ForecastEngine) -> None:
    horizon = 20
    series = [_sine_series()]
    results = asyncio.run(loaded_engine.forecast(series, horizon))

    assert len(results) == 1
    result = results[0]
    assert result.id == "SINE"
    assert len(result.mean) == horizon
    assert result.quantiles.keys() == {f"p{p}" for p in range(10, 91, 10)}
    for q_values in result.quantiles.values():
        assert len(q_values) == horizon

    for i in range(horizon):
        p10 = result.quantiles["p10"][i]
        p90 = result.quantiles["p90"][i]
        mean = result.mean[i]
        assert p10 <= mean <= p90, (
            f"step {i}: expected p10 <= mean <= p90, got "
            f"{p10} <= {mean} <= {p90}"
        )


def test_warm_second_call_is_fast(loaded_engine: ForecastEngine) -> None:
    series = [_sine_series()]
    # Prime it, in case this is the first inference this module has run
    # against loaded_engine (e.g. invoked in isolation via -k).
    asyncio.run(loaded_engine.forecast(series, 20))

    start = time.monotonic()
    asyncio.run(loaded_engine.forecast(series, 20))
    elapsed = time.monotonic() - start

    assert elapsed < 30.0, f"warm forecast took {elapsed:.2f}s (limit 30s)"
