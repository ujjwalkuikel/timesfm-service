"""The forecast engine interface.

This module ships the interface plus a stub implementation only — the
real TimesFM 3 checkpoint load + predict lands in Task 3, transcribed from
docs/feasibility.md's reference snippet, without changing this interface.

The stub always reports itself unloaded (`loaded()` is False) and
`forecast()` always raises NotLoaded, since there is no real model to run.
service.py's contract tests never exercise this stub directly — they
monkeypatch `service.ENGINE` with a fake that implements the same
interface, so this file can stay honest about having nothing real to do
yet.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass


class NotLoaded(Exception):
    """Raised by forecast() when the engine has not finished loading."""


@dataclass
class SeriesIn:
    """One input series, already validated by service.py."""

    id: str
    values: list[float]
    freq: str = "D"


@dataclass
class SeriesOut:
    """One forecast result in the contract shape.

    `mean` and each list in `quantiles` (keys "p10".."p90") have exactly
    `horizon` floats, per docs/feasibility.md's output-shape note.
    """

    id: str
    mean: list[float]
    quantiles: dict[str, list[float]]


class ForecastEngine:
    """Loads a TimesFM checkpoint and runs single-flight forecasts.

    Task 2 (this file, as of now): interface + NotLoaded stub, no real
    model. Task 3 replaces `_load_sync`/`forecast`'s bodies with a real
    checkpoint load and `predict_batch` call, guarded by `_lock` for
    single-flight inference, without touching this public interface.
    """

    model_version: str = "timesfm-3-330m"

    def __init__(self) -> None:
        self._loaded = False
        self._queue_depth = 0
        self._lock = asyncio.Lock()

    def loaded(self) -> bool:
        return self._loaded

    async def ensure_loaded(self) -> None:
        """Load the model if it isn't already. Idempotent.

        Task 3: run the heavy checkpoint load in asyncio.to_thread and set
        self._loaded = True on success. The stub has no real model to
        load, so it intentionally leaves `_loaded` False.
        """
        return None

    async def forecast(
        self,
        series: list[SeriesIn],
        horizon: int,
        covariates: dict | None = None,
    ) -> list[SeriesOut]:
        """Run single-flight inference for `series` out to `horizon` steps.

        Raises NotLoaded if ensure_loaded() has not produced a usable
        model yet. Task 2's stub is never loaded, so it always raises.
        """
        if not self._loaded:
            raise NotLoaded("model not loaded")
        raise NotLoaded("model not loaded")  # unreachable until Task 3

    @property
    def queue_depth(self) -> int:
        """Number of forecast() calls currently queued on `_lock`."""
        return self._queue_depth


# Module singleton used by service.py. Tests monkeypatch
# `service.ENGINE` (the name service.py imported into its own namespace)
# rather than this one, so the fake takes effect regardless of import
# order.
ENGINE = ForecastEngine()
