"""
Dashboard helpers — model loading, prediction, AQI category mapping.

Separated from app.py so the UI file stays focused on layout.
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# Make project-root imports work
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.hopsworks_client import get_feature_store

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHAMPION_DIR = PROJECT_ROOT / "models" / "champion"
CQR_DIR = PROJECT_ROOT / "models" / "cqr"

HORIZONS = [24, 48, 72]
CHAMPION_FEATURES = ["aqi", "month", "aqi_lag_24h", "aqi_lag_72h", "humidity"]

# Pakistan Standard Time is UTC+5 (no DST)
PKT_OFFSET = pd.Timedelta(hours=5)


def to_pkt(ts: pd.Timestamp) -> pd.Timestamp:
    """Convert a UTC timestamp to Pakistan Standard Time (UTC+5)."""
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts + PKT_OFFSET


def format_pkt(ts: pd.Timestamp, fmt: str = "%a %d %b · %H:%M") -> str:
    """Format a UTC timestamp in PKT for display."""
    return to_pkt(ts).strftime(fmt) + " PKT"

# ========================================================================
# AQI category mapping (US EPA standard)
# ========================================================================

AQI_CATEGORIES = [
    (0, 50, "Good", "#00e400", "Air quality is satisfactory."),
    (51, 100, "Moderate", "#ffff00",
     "Air quality is acceptable. Sensitive groups may experience minor effects."),
    (101, 150, "Unhealthy for Sensitive Groups", "#ff7e00",
     "Sensitive groups should reduce prolonged outdoor exertion."),
    (151, 200, "Unhealthy", "#ff0000",
     "Everyone may experience health effects. Limit outdoor activity."),
    (201, 300, "Very Unhealthy", "#8f3f97",
     "Health alert: serious effects possible. Avoid outdoor activity."),
    (301, 500, "Hazardous", "#7e0023",
     "Health warning: emergency conditions. Stay indoors."),
]


def aqi_to_category(aqi_value: float) -> dict:
    """Map an AQI value to its category, color, and health advisory."""
    aqi = max(0, min(500, int(round(aqi_value))))
    for low, high, name, color, advice in AQI_CATEGORIES:
        if low <= aqi <= high:
            return {
                "value": aqi,
                "name": name,
                "color": color,
                "advice": advice,
            }
    return {
        "value": aqi,
        "name": "Hazardous",
        "color": "#7e0023",
        "advice": "Health warning: emergency conditions.",
    }


# ========================================================================
# Model loading (cached)
# ========================================================================

def load_champion_models() -> dict:
    """
    Load the 3 champion Ridge models (24h/48h/72h) plus their scalers.

    Returns dict like:
        {
            24: {"model": ..., "scaler": ...},
            48: {...},
            72: {...},
        }
    """
    models = {}
    for h in HORIZONS:
        h_dir = CHAMPION_DIR / f"h{h}"
        models[h] = {
            "model": joblib.load(h_dir / "ridge_model.joblib"),
            "scaler": joblib.load(h_dir / "scaler.joblib"),
        }
    return models


def load_cqr_models() -> dict:
    """
    Load the 3 CQR systems for prediction intervals.

    Returns dict like:
        {
            24: {"p10": ..., "p90": ..., "scaler": ..., "calibration": {...}},
            48: {...},
            72: {...},
        }
    """
    cqr = {}
    for h in HORIZONS:
        h_dir = CQR_DIR / f"h{h}"
        cqr[h] = {
            "p10": joblib.load(h_dir / "quantile_p10.joblib"),
            "p90": joblib.load(h_dir / "quantile_p90.joblib"),
            "scaler": joblib.load(h_dir / "scaler.joblib"),
            "calibration": json.loads(
                (h_dir / "calibration.json").read_text()
            ),
        }
    return cqr


def load_champion_metadata() -> dict:
    """Load the champion training metadata (R^2, RMSE, etc.)."""
    return json.loads((CHAMPION_DIR / "champion_metadata.json").read_text())


# ========================================================================
# Data fetching
# ========================================================================

def fetch_recent_features(hours: int = 168) -> pd.DataFrame:
    """
    Pull the most recent `hours` rows from Hopsworks (default: 7 days).

    Includes the Arrow-Flight-fallback pattern from the feature pipeline
    so dashboard loads don't randomly fail.
    """
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

    # Filter to last N hours
    cutoff = df["timestamp"].max() - pd.Timedelta(hours=hours)
    df = df[df["timestamp"] >= cutoff].reset_index(drop=True)
    return df


# ========================================================================
# Prediction
# ========================================================================

def make_predictions(features_df: pd.DataFrame,
                     champion: dict,
                     cqr: dict) -> dict:
    """
    Generate point + interval predictions for 24h/48h/72h horizons
    using the most recent row in features_df.

    Returns:
        {
            24: {"point": float, "lower": float, "upper": float, "target_time": Timestamp},
            48: {...},
            72: {...},
        }
    """
    if features_df.empty:
        raise ValueError("No features available for prediction.")

    latest = features_df.iloc[-1]
    base_time = latest["timestamp"]

    X = latest[CHAMPION_FEATURES].to_frame().T.astype(float)

    predictions = {}
    for h in HORIZONS:
        # Champion point prediction
        scaler = champion[h]["scaler"]
        model = champion[h]["model"]
        X_scaled = scaler.transform(X)
        point = float(model.predict(X_scaled)[0])

        # CQR interval (Romano et al. 2019, hybrid):
        #   lower = champion_point + QR_low_residual - Q_widen
        #   upper = champion_point + QR_high_residual + Q_widen
        # The QR models predict RESIDUALS off the local Ridge, then
        # Q_widen calibrates to the target coverage.
        cqr_scaler = cqr[h]["scaler"]
        cqr_X_scaled = cqr_scaler.transform(X)
        resid_lower = float(cqr[h]["p10"].predict(cqr_X_scaled)[0])
        resid_upper = float(cqr[h]["p90"].predict(cqr_X_scaled)[0])
        q_widen = cqr[h]["calibration"]["Q_widen"]
        lower = point + resid_lower - q_widen
        upper = point + resid_upper + q_widen

        predictions[h] = {
            "point": max(0, point),
            "lower": max(0, lower),
            "upper": max(0, upper),
            "target_time": base_time + pd.Timedelta(hours=h),
        }

    return predictions