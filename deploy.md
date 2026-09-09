# TimesFM Forecast Service: Deployment Runbook

This document provides step-by-step instructions to deploy the TimesFM forecast service on Ubuntu using systemd.

## Prerequisites

- Ubuntu system with Python 3.12 available
- `sudo` access for systemd installation
- Internet connectivity for package downloads
- Recommended: 6 GB RAM, 6 GB disk space (venv + checkpoint)

## Installation Steps

### 1. Create Virtual Environment

```bash
cd /home/ubuntu
mkdir -p timesfm-service
cd timesfm-service
python3.12 -m venv .venv
```

### 2. Install Dependencies

Activate the virtual environment and install the exact spec from feasibility.md:

```bash
source .venv/bin/activate
.venv/bin/pip install 'timesfm[torch]' fastapi uvicorn pytest httpx
```

This resolves:
- timesfm 3.0.1
- torch 2.14.0 (aarch64 CPU wheels)
- venv size: 5.3 GB
- checkpoint download: ~1.3 GB on first load (cached thereafter)

### 3. Deploy Application Files

Copy the service code to the working directory:

```bash
# Assuming this repository is already cloned; copy to /home/ubuntu/timesfm-service
cp service.py settings.py model.py requirements.txt /home/ubuntu/timesfm-service/
```

### 4. Create Environment File

Generate a secure API key and create `.env`:

```bash
cd /home/ubuntu/timesfm-service
python -c "import secrets; print(secrets.token_hex(32))" > api_key.txt
cat << 'EOF' > .env
FORECAST_API_KEY=$(cat api_key.txt)
FORECAST_PORT=8090
FORECAST_WARM_AT_BOOT=1
# HF_TOKEN=  # Optional: set if you have a Hugging Face token to avoid rate limits
EOF
chmod 600 .env
rm api_key.txt
```

**Alternatively, manually create `.env`:**

1. Generate an API key:
   ```bash
   python3.12 -c "import secrets; print(secrets.token_hex(32))"
   ```
   Copy the output.

2. Edit `.env`:
   ```bash
   nano .env
   ```

3. Add these lines:
   ```
   FORECAST_API_KEY=<paste-the-generated-key-here>
   FORECAST_PORT=8090
   FORECAST_WARM_AT_BOOT=1
   # HF_TOKEN=  # Optional
   ```

4. Save and exit. Ensure `.env` is readable only by the ubuntu user:
   ```bash
   chmod 600 .env
   ```

### 5. Install and Enable Systemd Unit

Copy the systemd unit file and enable the service:

```bash
sudo cp deploy/timesfm-forecast.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now timesfm-forecast
```

Check the status:

```bash
sudo systemctl status timesfm-forecast
```

View recent logs:

```bash
sudo journalctl -u timesfm-forecast -n 50 -f
```

## Verification

### 1. Health Check (No Auth)

Poll the health endpoint until the model is loaded:

```bash
curl http://127.0.0.1:8090/v1/health
```

Example response while loading:
```json
{
  "ok": true,
  "model_loaded": false,
  "queue_depth": 0
}
```

Once the model is loaded (usually 45–90 seconds on cold start if WARM_AT_BOOT=1):
```json
{
  "ok": true,
  "model_loaded": true,
  "queue_depth": 0
}
```

### 2. Authorized Forecast Request

Once `model_loaded` is `true`, test with an authorized forecast:

```bash
API_KEY=$(grep FORECAST_API_KEY /home/ubuntu/timesfm-service/.env | cut -d= -f2)

curl -X POST http://127.0.0.1:8090/v1/forecast \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "series": [
      {
        "id": "NVDA",
        "values": [150.0, 151.5, 149.8, 152.2, 150.5, 153.1, 154.0, 152.8, 155.2, 151.9],
        "freq": "D"
      }
    ],
    "horizon": 5,
    "quantiles": true
  }'
```

Expected response (should complete in 1–2 seconds):
```json
{
  "model_version": "google/timesfm-3.0-pytorch",
  "generated_at": "2026-09-08T15:30:45.123456+00:00",
  "results": [
    {
      "id": "NVDA",
      "mean": [151.2, 151.5, 151.8, 152.1, 152.4],
      "quantiles": {
        "p10": [...],
        "p20": [...],
        "p30": [...],
        "p40": [...],
        "p50": [...],
        "p60": [...],
        "p70": [...],
        "p80": [...],
        "p90": [...]
      }
    }
  ]
}
```

## Maintenance

### Restart the Service

To restart the service after code changes:

```bash
sudo systemctl restart timesfm-forecast
```

### Update the Application

To pull new code and restart:

```bash
cd /home/ubuntu/timesfm-service
# Pull/sync new code
git pull  # or manually copy updated files
sudo systemctl restart timesfm-forecast
```

### Check Memory Usage

Verify that peak RSS stays below the 6 GB limit:

```bash
ps aux | grep uvicorn
```

## Troubleshooting

### Model fails to load

Check logs for download or permission errors:

```bash
sudo journalctl -u timesfm-forecast -n 100 | grep -i error
```

If Hugging Face is rate-limiting, set `HF_TOKEN` in `.env` (see Prerequisites).

### Service exits or restarts frequently

Check logs:

```bash
sudo journalctl -u timesfm-forecast --no-pager
```

Common causes:
- Out of memory (increase `MemoryMax` in the systemd unit if running on a larger machine)
- Missing `.env` file or invalid keys
- Port 8090 already in use

### Authorization fails

Verify the API key in `.env`:

```bash
grep FORECAST_API_KEY /home/ubuntu/timesfm-service/.env
```

Ensure it matches the key you use in the Bearer token header.

## License Note

TimesFM 3.0 weights are licensed under the **TimesFM Non-Commercial License v1.0** (no commercial/production use). The code is Apache-2.0.

**Owner decision (2026-09-08):** Use v3.0 now for personal, pre-revenue research. Before any commercialization, purchase a commercial license or swap to Apache-licensed TimesFM 2.5.

See `docs/feasibility.md` for full details.
