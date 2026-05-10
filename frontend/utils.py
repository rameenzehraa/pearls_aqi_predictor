"""
Dashboard helpers — Flask API client + display utilities.

The dashboard talks to the Flask backend (deployed on Render) for all
data and predictions. This keeps the dashboard a thin presentation
layer with no Hopsworks credentials needed at runtime.

Backend URL and API key are read from environment variables (or
Streamlit Cloud secrets at runtime):
    BACKEND_URL      — e.g., https://pearls-aqi-backend.onrender.com
    BACKEND_API_KEY  — shared secret matching Render's env var
"""

import json
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv

# Project root for legacy path setup
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load .env so BACKEND_URL / BACKEND_API_KEY are available locally.
# On Streamlit Cloud, secrets come from st.secrets instead — this is a no-op there.
load_dotenv(PROJECT_ROOT / ".env")
HORIZONS = [24, 48, 72]

# Pakistan Standard Time is UTC+5 (no DST)
PKT_OFFSET = pd.Timedelta(hours=5)

# Backend timeout — Render free tier cold starts can take 30-60s,
# and Hopsworks reads on top of that take another 30-60s. Be generous.
REQUEST_TIMEOUT = 120


# ========================================================================
# Backend config (env vars locally, st.secrets on Streamlit Cloud)
# ========================================================================

def _get_secret(key: str, default: str = "") -> str:
    """
    Read a secret from Streamlit Cloud's st.secrets first, then fall
    back to environment variables (for local dev).
    """
    try:
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass  # st.secrets unavailable in some contexts
    return os.getenv(key, default)


def get_backend_url() -> str:
    url = _get_secret("BACKEND_URL", "")
    if not url:
        raise RuntimeError(
            "BACKEND_URL is not configured. Set it in .env (local) or "
            "Streamlit Cloud secrets."
        )
    return url.rstrip("/")


def get_api_key() -> str:
    key = _get_secret("BACKEND_API_KEY", "")
    if not key:
        raise RuntimeError(
            "BACKEND_API_KEY is not configured. Set it in .env (local) "
            "or Streamlit Cloud secrets."
        )
    return key


def _api_get(path: str, params: dict | None = None) -> dict:
    """GET request to the Flask backend with the API key header."""
    url = get_backend_url() + path
    headers = {"X-API-Key": get_api_key()}
    response = requests.get(
        url, headers=headers, params=params or {}, timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    return response.json()


# ========================================================================
# Time helpers
# ========================================================================

def to_pkt(ts) -> pd.Timestamp:
    """Convert a UTC timestamp (or ISO string) to PKT (UTC+5)."""
    ts = pd.Timestamp(ts)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts + PKT_OFFSET


def format_pkt(ts, fmt: str = "%a %d %b · %H:%M") -> str:
    """Format a timestamp in PKT for display."""
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
                "value": aqi, "name": name,
                "color": color, "advice": advice,
            }
    return {
        "value": aqi, "name": "Hazardous",
        "color": "#7e0023",
        "advice": "Health warning: emergency conditions.",
    }


# ========================================================================
# Backend-backed equivalents of the old functions
# (signatures preserved so dashboard/app.py doesn't change)
# ========================================================================

def fetch_predictions_payload() -> dict:
    """Call /predictions and return the full JSON payload."""
    return _api_get("/predictions")


def fetch_recent_features(hours: int = 168) -> pd.DataFrame:
    """
    Pull historical readings from the Flask backend.

    Returns a DataFrame with columns: timestamp, aqi.
    Note: this lighter payload (timestamp + aqi only) is enough for
    the 7-day chart. Other current-condition fields come via
    /predictions (current block).
    """
    payload = _api_get("/history", params={"hours": hours})
    rows = payload.get("rows", [])
    if not rows:
        return pd.DataFrame(columns=["timestamp", "aqi"])
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    df = df[["timestamp", "aqi"]].sort_values("timestamp").reset_index(drop=True)
    return df


def load_champion_metadata() -> dict:
    """Load model metadata via the backend."""
    return _api_get("/metadata")


# Backwards-compatible shims — app.py still calls these names.
# load_champion_models / load_cqr_models / make_predictions are now
# folded into fetch_predictions_payload, but app.py expects them to
# return objects it passes to make_predictions(). We keep them as
# no-ops returning None and have make_predictions ignore those args.

def load_champion_models() -> Any:
    return None


def load_cqr_models() -> Any:
    return None


def make_predictions(features_df: pd.DataFrame, champion=None, cqr=None) -> dict:
    """
    Returns predictions in the same shape as the old local-compute version,
    but sources them from the backend instead of computing them here.

    The features_df argument is now used only to derive `target_time`
    bases — it's still the latest reading from the backend's history
    response, so timestamps are consistent.
    """
    payload = fetch_predictions_payload()
    forecasts = payload["forecasts"]
    out = {}
    for h in HORIZONS:
        f = forecasts[str(h)]
        out[h] = {
            "point": float(f["point"]),
            "lower": float(f["lower"]),
            "upper": float(f["upper"]),
            "target_time": pd.to_datetime(f["target_time_utc"], utc=True),
        }
    return out


def fetch_current_conditions() -> dict:
    """
    Get the current air-quality snapshot.

    Tries the live Open-Meteo `current` endpoint first (~15 min lag).
    Falls back to the latest validated hourly row (the `current` block
    of /predictions, which can be 6-24h stale) if the live endpoint
    fails or returns nothing useful.
    """
    # Primary path: live snapshot from /current_live
    try:
        live = _api_get("/current_live")
        if live and live.get("aqi") is not None:
            return {
                "aqi": float(live["aqi"]),
                "pm2_5": float(live["pm2_5"]),
                "pm10": float(live["pm10"]),
                "temperature": float(live["temperature"]),
                "humidity": float(live["humidity"]),
                "timestamp": pd.to_datetime(live["timestamp_utc"], utc=True),
                "source": "live",  # Open-Meteo current endpoint
            }
    except Exception:
        # Network error, 503, or backend issue — fall through to hourly fallback
        pass

    # Fallback: latest hourly row (can be 6-24h stale)
    payload = fetch_predictions_payload()
    cur = payload["current"]
    return {
        "aqi": float(cur["aqi"]),
        "pm2_5": float(cur["pm2_5"]),
        "pm10": float(cur["pm10"]),
        "temperature": float(cur["temperature"]),
        "humidity": float(cur["humidity"]),
        "timestamp": pd.to_datetime(cur["timestamp_utc"], utc=True),
        "source": "hourly",  # Validated hourly endpoint, may be stale
    }