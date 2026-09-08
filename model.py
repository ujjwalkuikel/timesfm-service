"""The forecast engine interface, backed by a real TimesFM 3 checkpoint.

This module ships the frozen public interface (`model_version`, `loaded()`,
`ensure_loaded()`, `forecast()`, `queue_depth`, module singleton `ENGINE`)
established in Task 2, now implemented for real per docs/feasibility.md's
reference snippet — the source of truth for the TimesFM 3 API, transcribed
here rather than improvised.

`timesfm3`/`torch`/`numpy` are imported lazily, inside the methods that need
them, never at module import time. This keeps service.py's contract tests
(which monkeypatch `service.ENGINE` with a fake and never touch this real
engine) importable and green in an environment with none of those packages
installed — see tests/test_contract.py's module docstring.

Covariates (v1 policy): docs/feasibility.md records that `predict_batch`
accepts `past_only_covariates`/`past_future_covariates` kwargs, but they are
not wired into this service in v1. Any request that reaches `forecast()`
with actual covariate data raises `CovariatesUnsupported` regardless of how
many series are in the request — service.py maps that to a 422
`{"error": "covariates unsupported in v1"}` response. This is a deliberate
"discovered but not wired" case per the plan's Task 3 step 1, not a bug: we
do not fake covariate support.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass

# The exact checkpoint + config transcribed from docs/feasibility.md's
# reference snippet. Do not change without re-reading that file.
_CHECKPOINT_PATH = "google/timesfm-3.0-pytorch"
_PER_CORE_BATCH_SIZE = 16
_DEVICE = "cpu"


class NotLoaded(Exception):
    """Raised by forecast() when the engine has not finished loading."""


class CovariatesUnsupported(Exception):
    """Raised by forecast() when the request carries covariate data.

    v1 does not wire covariates into the model (see module docstring).
    service.py maps this to a 422.
    """


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


def _covariates_present(covariates: dict | None) -> bool:
    """True iff `covariates` actually carries data (not None/empty/blank).

    service.py always passes a dict when the request included a
    `covariates` block at all, even one with both sub-fields left unset
    (e.g. `"covariates": {}`) — that shows up here as
    `{"known_future": None, "past_only": None}`, which is a dict but
    carries nothing. Only treat covariates as "present" when at least one
    of the two sub-fields has actual data.
    """
    if not covariates:
        return False
    return bool(covariates.get("known_future")) or bool(covariates.get("past_only"))


class ForecastEngine:
    """Loads a TimesFM checkpoint and runs single-flight forecasts.

    Public interface is frozen (see module docstring / docs/feasibility.md
    consumers in service.py + tests/test_contract.py): `model_version`,
    `loaded()`, `ensure_loaded()`, `forecast()`, `queue_depth`.
    """

    model_version: str = "timesfm-3-330m"

    def __init__(self) -> None:
        self._loaded = False
        self._queue_depth = 0
        self._lock = asyncio.Lock()  # single-flight inference
        self._load_lock = asyncio.Lock()  # guards concurrent ensure_loaded()
        self._model = None

    def loaded(self) -> bool:
        return self._loaded

    def _load_sync(self):
        """Blocking checkpoint load. Runs off the event loop via to_thread.

        Transcribed verbatim from docs/feasibility.md's reference snippet.
        """
        from timesfm3 import ModelConfig, TimesFM3Evaluator

        config = ModelConfig(
            checkpoint_path=_CHECKPOINT_PATH,
            per_core_batch_size=_PER_CORE_BATCH_SIZE,
            device=_DEVICE,
        )
        return TimesFM3Evaluator(config)

    async def ensure_loaded(self) -> None:
        """Load the model if it isn't already. Idempotent.

        The heavy checkpoint load runs in asyncio.to_thread. A dedicated
        lock (separate from the inference `_lock`) prevents two concurrent
        callers from each triggering their own checkpoint load; it is not
        reflected in `queue_depth`, which counts forecast() waiters only.
        """
        if self._loaded:
            return
        async with self._load_lock:
            if self._loaded:  # re-check: someone else may have loaded it
                return
            model = await asyncio.to_thread(self._load_sync)
            self._model = model
            self._loaded = True

    def _predict_sync(
        self, series: list[SeriesIn], horizon: int
    ) -> list[SeriesOut]:
        """Blocking predict_batch call + contract mapping.

        Each input series becomes its own single-series batch
        (`contexts` entry of shape (1, context_len)) so series of
        different lengths in one request can share a single
        `predict_batch` call, mirroring docs/feasibility.md's documented
        shape `(n_series, context_len)` per contexts entry without
        assuming all series in a request share a length.
        """
        import numpy as np

        contexts = [
            np.asarray(s.values, dtype=np.float32).reshape(1, -1) for s in series
        ]
        outputs = list(
            self._model.predict_batch(
                contexts=contexts,
                horizon=horizon,
                return_quantiles=True,
            )
        )

        results: list[SeriesOut] = []
        for s, out in zip(series, outputs):
            mean = [float(x) for x in out.forecast[0]]
            quantiles = {
                f"p{(i + 1) * 10}": [float(x) for x in out.quantiles[0, :, i]]
                for i in range(9)
            }
            results.append(SeriesOut(id=s.id, mean=mean, quantiles=quantiles))
        return results

    async def forecast(
        self,
        series: list[SeriesIn],
        horizon: int,
        covariates: dict | None = None,
    ) -> list[SeriesOut]:
        """Run single-flight inference for `series` out to `horizon` steps.

        Raises NotLoaded if ensure_loaded() has not produced a usable
        model yet, and CovariatesUnsupported if the request carries actual
        covariate data (v1 policy — see module docstring).
        """
        if not self._loaded:
            raise NotLoaded("model not loaded")
        if _covariates_present(covariates):
            raise CovariatesUnsupported("covariates unsupported in v1")

        self._queue_depth += 1
        decremented = False
        try:
            async with self._lock:
                self._queue_depth -= 1
                decremented = True
                return await asyncio.to_thread(self._predict_sync, series, horizon)
        finally:
            if not decremented:
                self._queue_depth -= 1

    @property
    def queue_depth(self) -> int:
        """Number of forecast() calls currently queued on `_lock`."""
        return self._queue_depth


# Module singleton used by service.py. Tests monkeypatch
# `service.ENGINE` (the name service.py imported into its own namespace)
# rather than this one, so the fake takes effect regardless of import
# order.
ENGINE = ForecastEngine()
