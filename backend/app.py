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

import json
import os
import sys
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, request

# Make project-root imports work
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.auth import require_api_key
from utils.hopsworks_client import get_feature_store

# ========================================================================
# Constants and paths
# ========================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHAMPION_DIR = PROJECT_ROOT / "models" / "champion"
CQR_DIR = PROJECT_ROOT / "models" / "cqr"

HORIZONS = [24, 48, 72]
CHAMPION_FEATURES = ["aqi", "month", "aqi_lag_24h", "aqi_lag_72h", "humidity"]


# ========================================================================
# App setup + model loading at startup
# ========================================================================

app = Flask(__name__)


def load_all_models():
    """Load all 6 models into memory once at startup."""
    champion = {}
    cqr = {}
    for h in HORIZONS:
        ch_dir = CHAMPION_DIR / f"h{h}"
        cq_dir = CQR_DIR / f"h{h}"
        champion[h] = {
            "model": joblib.load(ch_dir / "ridge_model.joblib"),
            "scaler": joblib.load(ch_dir / "scaler.joblib"),
        }
        cqr[h] = {
            "p10": joblib.load(cq_dir / "quantile_p10.joblib"),
            "p90": joblib.load(cq_dir / "quantile_p90.joblib"),
            "scaler": joblib.load(cq_dir / "scaler.joblib"),
            "calibration": json.loads((cq_dir / "calibration.json").read_text()),
        }
    return champion, cqr


def load_metadata():
    return json.loads((CHAMPION_DIR / "champion_metadata.json").read_text())


# Loaded once when the Flask process starts. On Render, this happens
# during cold start — adds ~2-3s but only on first request after sleep.
print("[startup] Loading models...")
CHAMPION, CQR = load_all_models()
METADATA = load_metadata()
print(f"[startup] Loaded {len(CHAMPION)} champion + {len(CQR)} CQR models")


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

    latest = features_df.iloc[-1]
    base_time = latest["timestamp"]
    X = latest[CHAMPION_FEATURES].to_frame().T.astype(float)

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


# ========================================================================
# Local dev entrypoint
# ========================================================================

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)