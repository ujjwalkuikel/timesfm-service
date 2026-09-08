"""Environment-derived settings for the forecast service.

Read once at import time. Tests monkeypatch these module attributes
directly (e.g. `monkeypatch.setattr(settings, "API_KEY", "...")`) rather
than mutating the environment, so keep any consumer reading them as
`settings.NAME` at call time (not via `from settings import NAME`).
"""
from __future__ import annotations

import os

# Bearer token required on every /v1/* route except /v1/health. A missing
# or blank value means the service fails closed (503 "service not
# configured") on all authenticated routes rather than accepting requests
# with no real key configured.
API_KEY: str | None = os.getenv("FORECAST_API_KEY") or None

# Port uvicorn binds to; see deploy.md for the systemd ExecStart line.
PORT: int = int(os.getenv("FORECAST_PORT", "8090"))

# When true, the service loads the model at startup instead of waiting for
# the first request. See model.ForecastEngine.ensure_loaded.
WARM_AT_BOOT: bool = os.getenv("FORECAST_WARM_AT_BOOT", "0") == "1"
