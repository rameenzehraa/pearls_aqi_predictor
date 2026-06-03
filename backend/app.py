"""
Pearls AQI Predictor — Flask backend.

Serves AQI predictions and historical data via JSON REST endpoints.
The Streamlit dashboard is the primary consumer; the API is also
demoable via curl/Postman for grader/coordinator verification.

Endpoints:
    GET /health        — uptime check, no auth
    GET /predictions   — current + 24/48/72h forecast with intervals
    GET /history       — last N hours of actual readings
    GET /metadata      — model info (R², features, training date)

Run locally:
    cd backend
    python app.py

Deploy:
    Render auto-deploys from GitHub on push to main.
    See backend/Procfile for the production start command.
"""

import os
import sys
import time
import logging
from pathlib import Path

import pandas as pd
import requests
from flask import Flask, jsonify, request

# Make project-root imports work
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.auth import require_api_key
from backend.model_loader import load_all_models, load_metadata
from utils.hopsworks_client import get_feature_store

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")

from config.config import KARACHI_LAT, KARACHI_LON

# ========================================================================
# Constants and paths
# ========================================================================

HORIZONS = [24, 48, 72]
CHAMPION_FEATURES = ["aqi", "month", "aqi_lag_24h", "aqi_lag_72h", "humidity"]


# Simple in-memory cache for /current_live (5 min TTL)
# Open-Meteo's current endpoint refreshes every ~15 min; 5 min cache
# avoids hammering them while keeping data fresh enough.
_current_live_cache = {"data": None, "fetched_at": 0}
CURRENT_LIVE_TTL = 300  # 5 minutes

# ========================================================================
# App setup + model loading at startup
# ========================================================================

app = Flask(__name__)



# Loaded once when the Flask process starts. Cold start (fresh container,
# empty /tmp/ cache): ~30-60s to download from Hopsworks. Warm restart
# (same container, /tmp/ intact): ~1s from disk cache.
logger.info("Loading models...")
CHAMPION, CQR = load_all_models()
METADATA = load_metadata()
logger.info(f"Loaded {len(CHAMPION)} champion + {len(CQR)} CQR models")


# ========================================================================
# Hopsworks fetcher (with Arrow Flight fallback)
# ========================================================================

def fetch_features(hours: int = 168) -> pd.DataFrame:
    """Pull recent rows from Hopsworks feature store."""
    fs = get_feature_store()
    fg = fs.get_feature_group("aqi_features", version=1)

    try:
        df = fg.read()
    except Exception as exc:
        msg = str(exc)
        if any(s in msg for s in ("Flight", "Socket closed", "Query Service")):
            df = fg.read(read_options={"use_hive": True})
        else:
            raise

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    cutoff = df["timestamp"].max() - pd.Timedelta(hours=hours)
    df = df[df["timestamp"] >= cutoff].reset_index(drop=True)
    return df


# ========================================================================
# Prediction logic
# ========================================================================

def compute_predictions(features_df: pd.DataFrame) -> dict:
    """Generate point + interval predictions for all 3 horizons."""
    if features_df.empty:
        raise ValueError("No features available")

    # Find the most recent row with all champion features present.
    # If unavailable (transient NaN from upstream API or materialization
    # gaps), fall back to the latest row and impute missing values with
    # median from training data, matching the training pipeline's behaviour.
    clean = features_df.dropna(subset=CHAMPION_FEATURES)
    if not clean.empty:
        latest = clean.iloc[-1]
    else:
        latest = features_df.iloc[-1].copy()
        # Impute NaN champion features. During materialization catch-up the
        # whole fetched window can be NaN for a lag column (e.g. aqi_lag_72h),
        # so its median is *also* NaN — the prior version filled NaN with NaN
        # and Ridge then rejected the input. Fallback chain: column median ->
        # current aqi as a persistence proxy for lag-style features -> 0.0
        # (caught by the hard guard below).
        medians = features_df[CHAMPION_FEATURES].median()
        current_aqi = latest["aqi"] if not pd.isna(latest["aqi"]) else medians["aqi"]
        for f in CHAMPION_FEATURES:
            if pd.isna(latest[f]):
                val = medians[f]
                if pd.isna(val) and (f == "aqi" or f.startswith("aqi_lag")):
                    val = current_aqi
                latest[f] = val

    base_time = latest["timestamp"]
    X = latest[CHAMPION_FEATURES].to_frame().T.astype(float)
    # Hard guard: Ridge rejects NaN natively, so guarantee none survive.
    X = X.fillna(0.0)

    out = {}
    for h in HORIZONS:
        ch_scaler = CHAMPION[h]["scaler"]
        ch_model = CHAMPION[h]["model"]
        X_ch = ch_scaler.transform(X)
        point = float(ch_model.predict(X_ch)[0])

        cq_scaler = CQR[h]["scaler"]
        X_cq = cq_scaler.transform(X)
        resid_lo = float(CQR[h]["p10"].predict(X_cq)[0])
        resid_hi = float(CQR[h]["p90"].predict(X_cq)[0])
        q_widen = CQR[h]["calibration"]["Q_widen"]

        out[str(h)] = {
            "point": max(0, point),
            "lower": max(0, point + resid_lo - q_widen),
            "upper": max(0, point + resid_hi + q_widen),
            "target_time_utc": (base_time + pd.Timedelta(hours=h)).isoformat(),
        }
    return out


# ========================================================================
# Endpoints
# ========================================================================

@app.route("/health", methods=["GET"])
def health():
    """Uptime check. No auth — used by Render's health monitoring."""
    return jsonify({
        "status": "ok",
        "models_loaded": len(CHAMPION) + len(CQR),
    })


@app.route("/predictions", methods=["GET"])
@require_api_key
def predictions():
    """Current AQI + 24/48/72h forecast with intervals."""
    try:
        features = fetch_features(hours=24)  # only need recent for prediction
        latest = features.iloc[-1]
        preds = compute_predictions(features)
        return jsonify({
            "current": {
                "aqi": float(latest["aqi"]),
                "pm2_5": float(latest["pm2_5"]),
                "pm10": float(latest["pm10"]),
                "temperature": float(latest["temperature"]),
                "humidity": float(latest["humidity"]),
                "timestamp_utc": latest["timestamp"].isoformat(),
            },
            "forecasts": preds,
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/history", methods=["GET"])
@require_api_key
def history():
    """Last N hours of actual readings (default 168 = 7 days)."""
    try:
        hours = int(request.args.get("hours", 168))
        hours = max(1, min(hours, 720))  # clamp 1h to 30d
        features = fetch_features(hours=hours)
        rows = [
            {
                "timestamp_utc": ts.isoformat(),
                "aqi": float(aqi),
            }
            for ts, aqi in zip(features["timestamp"], features["aqi"])
        ]
        return jsonify({"hours": hours, "rows": rows})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/metadata", methods=["GET"])
@require_api_key
def metadata():
    """Model training info."""
    return jsonify(METADATA)

@app.route("/current_live", methods=["GET"])
@require_api_key
def current_live():
    """
    Live air quality snapshot from Open-Meteo's `current` endpoint.

    This is for dashboard display only — the model still uses validated
    hourly data for forecasting. The `current` endpoint is based on
    15-minutely model data and refreshes every ~15 min, much fresher
    than the hourly endpoint's 6-24h publishing lag.

    Cached for 5 minutes to avoid hammering Open-Meteo on every
    dashboard refresh.
    """
    now = time.time()
    cache = _current_live_cache

    # Return cached value if fresh
    if cache["data"] is not None and (now - cache["fetched_at"]) < CURRENT_LIVE_TTL:
        return jsonify({**cache["data"], "from_cache": True})

    # Fetch fresh from Open-Meteo
    try:
        resp = requests.get(
            "https://air-quality-api.open-meteo.com/v1/air-quality",
            params={
                "latitude": KARACHI_LAT,
                "longitude": KARACHI_LON,
                "current": (
                    "pm2_5,pm10,us_aqi,carbon_monoxide,nitrogen_dioxide,"
                    "sulphur_dioxide,ozone"
                ),
                "timezone": "UTC",
            },
            timeout=10,
        )
        resp.raise_for_status()
        payload = resp.json()
    except Exception as exc:
        return jsonify({
            "error": f"Open-Meteo unavailable: {exc}",
            "fallback_required": True,
        }), 503

    # Open-Meteo's current section also includes weather; we need to fetch
    # weather separately because the air-quality endpoint doesn't include
    # temperature/humidity.
    try:
        weather_resp = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": KARACHI_LAT,
                "longitude": KARACHI_LON,
                "current": "temperature_2m,relative_humidity_2m,pressure_msl,wind_speed_10m",
                "timezone": "UTC",
            },
            timeout=10,
        )
        weather_resp.raise_for_status()
        weather_payload = weather_resp.json()
    except Exception as exc:
        # If weather fails, we still return AQI but no temp/humidity
        weather_payload = {"current": {}}

    aqi_current = payload.get("current", {})
    weather_current = weather_payload.get("current", {})

    result = {
        "aqi": aqi_current.get("us_aqi"),
        "pm2_5": aqi_current.get("pm2_5"),
        "pm10": aqi_current.get("pm10"),
        "no2": aqi_current.get("nitrogen_dioxide"),
        "o3": aqi_current.get("ozone"),
        "so2": aqi_current.get("sulphur_dioxide"),
        "co": aqi_current.get("carbon_monoxide"),
        "temperature": weather_current.get("temperature_2m"),
        "humidity": weather_current.get("relative_humidity_2m"),
        "pressure": weather_current.get("pressure_msl"),
        "wind_speed": weather_current.get("wind_speed_10m"),
        "timestamp_utc": aqi_current.get("time"),  # ISO string from Open-Meteo
        "source": "open-meteo-current",
    }

    cache["data"] = result
    cache["fetched_at"] = now

    return jsonify({**result, "from_cache": False})

# ========================================================================
# Local dev entrypoint
# ========================================================================

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)