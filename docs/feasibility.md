# Feasibility gate results — 2026-09-08 (Task 1) ✅ PASS

Measured on the production target: Oracle A1.Flex, 3 OCPU aarch64, 18 GB RAM,
Ubuntu, Python 3.12.3, CPU-only.

## Install (the exact spec)

```
python3.12 -m venv .venv
.venv/bin/pip install 'timesfm[torch]' fastapi uvicorn pytest httpx
# resolved: timesfm 3.0.1, torch 2.14.0 (aarch64 CPU wheels — no issues)
# venv size: 5.3 GB; checkpoint download ~1.3 GB on first load
```

## Numbers (abort thresholds in parentheses)

| Metric | Result |
|---|---|
| Import | 3.0 s |
| Cold load incl. HF download | 43.8 s (one-time → use warm-at-boot) |
| Warm forecast, 512 pts → h=20 | **1.25–1.40 s** (limit 30 s) ✅ |
| Peak RSS | **3.19 GB** (limit 6 GB) ✅ |

Recommendation: set `HF_TOKEN` in the service env to avoid unauthenticated
HF rate limits on re-downloads.

## The reference snippet (source of truth for model.py — transcribe, don't improvise)

```python
import numpy as np
from timesfm3 import TimesFM3Evaluator, ModelConfig

config = ModelConfig(
    checkpoint_path="google/timesfm-3.0-pytorch",
    per_core_batch_size=16,
    device="cpu",
)
forecaster = TimesFM3Evaluator(config)   # heavy: do in a thread, once

# contexts: list of np.float32 arrays shaped (n_series, context_len)
out = list(forecaster.predict_batch(
    contexts=[series_2d], horizon=20, return_quantiles=True,
))[0]   # timesfm3.timesfm3_forecaster.ForecastOutput

out.forecast   # np (n_series, horizon)            -> contract "mean"
out.quantiles  # np (n_series, horizon, 9)         -> last axis ASCENDS p10..p90
out.ts_id      # series identifier
```

Contract mapping: `quantiles[..., i]` → `p{(i+1)*10}` for i in 0..8
(verified empirically: first-step values ascend 94.75 → 97.28 on a synthetic
series). Covariates (`past_only_covariates`, `past_future_covariates` kwargs
of `predict_batch`) exist in the API but are NOT wired in service v1.

## License (owner decision on record)

TimesFM 3.0 **weights** are under the TimesFM **Non-Commercial License v1.0**
(no commercial/production use); the code is Apache-2.0. Owner decision
2026-09-08: use 3.0 now (personal, pre-revenue research posture) and purchase
a commercial license — or swap to Apache-licensed TimesFM 2.5 — **before any
commercialization**. The service contract is model-agnostic to keep that swap
a checkpoint change. This item belongs on any due-diligence checklist.
