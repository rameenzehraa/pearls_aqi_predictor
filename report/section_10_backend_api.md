# Section 10 — Backend API

## 10.1 Role and hosting

The backend is a Flask application (`backend/app.py`) deployed on Render's free tier. It serves predictions and historical data as JSON over a small REST surface, consumed primarily by the Streamlit dashboard (Section 11) and demoable directly via `curl`/Postman for verification. All endpoints except `/health` require an API key, passed as a request header and checked by a `require_api_key` decorator; `/health` is left open so external uptime monitoring can reach it.

## 10.2 Endpoints

| Endpoint | Auth | Purpose |
|---|---|---|
| `/health` | none | Uptime check; returns status and loaded-model count |
| `/predictions` | API key | Current reading + 24/48/72h forecasts with conformal intervals |
| `/history` | API key | Last N hours of actual AQI (N clamped to 1–720) |
| `/metadata` | API key | Champion training info (metrics, features, training date) |
| `/current_live` | API key | Live Open-Meteo `current` snapshot for the dashboard tile |

## 10.3 Model loading at cold start

The three champion and three CQR models are loaded once when the Flask process starts, via `load_all_models()` reading the latest versions from the Hopsworks Model Registry. Artifacts are cached to local disk ("Approach B"), so a dyno that restarts reuses the local copy rather than re-downloading from the registry. On Render's free tier the dyno sleeps after inactivity, so the first request after sleep pays a one-time cold-start cost of roughly 2–3 seconds for model loading; subsequent requests are served from the warmed process.

## 10.4 Prediction logic

`/predictions` reads recent rows from the feature store via `fetch_features` (Arrow Flight with the JDBC fallback described in Section 7), then `compute_predictions` builds the forecast. The most recent row with complete champion features is used as the anchor; for each horizon it applies that horizon's scaler and Ridge model for the point estimate, and the CQR quantile regressors plus the calibrated `Q_widen` for the interval (Section 9). All outputs are clamped at zero, since negative AQI is meaningless.

The anchor selection is defensive against incomplete data. If no recent row has all five champion features present — which occurs when the longest lag (`aqi_lag_72h`) cannot be computed during a materialisation catch-up — the code falls back to median imputation, uses the current AQI as a persistence proxy for any missing lag feature, and applies a final guard so that no `NaN` ever reaches Ridge (which rejects it natively). This keeps the endpoint returning valid JSON through transient feature-store gaps. The honest caveat: while a lag feature is missing, the longest-horizon forecast runs on the proxy and is approximate rather than fully accurate; it self-heals as valid lags repopulate. The underlying gap and its cause are documented in Section 14.

## 10.5 Live tile and cold-start mitigation

`/current_live` queries Open-Meteo's `current` endpoint directly and caches the result server-side for five minutes; this is the display-only path whose rationale is in Section 4.3, deliberately separate from the validated data that anchors forecasts. Separately, an UptimeRobot monitor pings `/health` every five minutes to keep the Render dyno warm, suppressing the cold-start latency from Section 10.3 for normal dashboard traffic.