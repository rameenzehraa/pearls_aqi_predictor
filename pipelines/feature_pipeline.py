"""
Feature pipeline — hourly ingestion.

Runs on GitHub Actions every hour. Designed to be small and fast
(target: <30 sec end-to-end) to stay under the free-tier minute budget.

Flow:
    1. Pull the last ~6 hours of fresh data from Open-Meteo (covers
       publishing lag; Open-Meteo can be 30-60 min behind real-time).
    2. Pull ~80 hours of recent history from the Hopsworks feature
       group (needed to compute the 72h lag and 24h rolling features
       for the new rows).
    3. Combine, clean, engineer features.
    4. Insert ONLY the new rows into Hopsworks. The primary key
       (`timestamp`) deduplicates anything we already have.

Hopsworks-only — no local file write. Local backups exist for the
backfill case; hourly delta lives in the feature store.

Usage:
    python -m pipelines.feature_pipeline
"""

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry

# Make project-root imports work when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.config import (
    API_RETRY_BASE_DELAY,
    API_RETRY_COUNT,
    KARACHI_LAT,
    KARACHI_LON,
)
from utils.features import clean_raw_data, engineer_features
from utils.hopsworks_client import get_feature_store
from utils.logger import get_logger

logger = get_logger(__name__)

AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"

FEATURE_GROUP_NAME = "aqi_features"
FEATURE_GROUP_VERSION = 1

# How many recent hours to pull from Open-Meteo. Need a small window so
# we cover Open-Meteo's publishing lag (30-60 min in practice).
RECENT_API_HOURS = 6

# How many recent hours to read from Hopsworks for lag/rolling features.
# 72h is the longest lag we compute, plus a small safety buffer.
HISTORY_LOOKBACK_HOURS = 80


# ========================================================================
# API client (cache disabled so GHA runners always get fresh data)
# ========================================================================

def _make_client() -> openmeteo_requests.Client:
    cache_session = requests_cache.CachedSession(".cache", expire_after=300)
    retry_session = retry(
        cache_session,
        retries=API_RETRY_COUNT,
        backoff_factor=API_RETRY_BASE_DELAY / 10,
    )
    return openmeteo_requests.Client(session=retry_session)


# ========================================================================
# Fetchers (forecast endpoints — for very recent data, not archive)
# ========================================================================

def fetch_recent_air_quality(client) -> pd.DataFrame:
    """Pull the last RECENT_API_HOURS of air quality from Open-Meteo."""
    params = {
        "latitude": KARACHI_LAT,
        "longitude": KARACHI_LON,
        "hourly": [
            "pm2_5",
            "pm10",
            "nitrogen_dioxide",
            "ozone",
            "sulphur_dioxide",
            "carbon_monoxide",
        ],
        "past_days": 1,        # past_days=1 covers the last 24h
        "forecast_days": 0,
        "timezone": "UTC",
    }
    responses = client.weather_api(AIR_QUALITY_URL, params=params)
    response = responses[0]
    hourly = response.Hourly()

    df = pd.DataFrame({
        "timestamp": pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left",
        ),
        "pm2_5": hourly.Variables(0).ValuesAsNumpy(),
        "pm10":  hourly.Variables(1).ValuesAsNumpy(),
        "no2":   hourly.Variables(2).ValuesAsNumpy(),
        "o3":    hourly.Variables(3).ValuesAsNumpy(),
        "so2":   hourly.Variables(4).ValuesAsNumpy(),
        "co":    hourly.Variables(5).ValuesAsNumpy(),
    })
    logger.info("Air quality from API: %d rows", len(df))
    return df


def fetch_recent_weather(client) -> pd.DataFrame:
    """Pull the last 24 hours of weather from Open-Meteo's forecast endpoint."""
    params = {
        "latitude": KARACHI_LAT,
        "longitude": KARACHI_LON,
        "hourly": [
            "temperature_2m",
            "relative_humidity_2m",
            "wind_speed_10m",
            "pressure_msl",
        ],
        "past_days": 1,
        "forecast_days": 0,
        "timezone": "UTC",
    }
    responses = client.weather_api(WEATHER_URL, params=params)
    response = responses[0]
    hourly = response.Hourly()

    df = pd.DataFrame({
        "timestamp": pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left",
        ),
        "temperature": hourly.Variables(0).ValuesAsNumpy(),
        "humidity":    hourly.Variables(1).ValuesAsNumpy(),
        "wind_speed":  hourly.Variables(2).ValuesAsNumpy(),
        "pressure":    hourly.Variables(3).ValuesAsNumpy(),
    })
    logger.info("Weather from API: %d rows", len(df))
    return df


# ========================================================================
# History from Hopsworks
# ========================================================================

def fetch_recent_history(fg, hours: int = HISTORY_LOOKBACK_HOURS) -> pd.DataFrame:
    """
    Pull the most recent `hours` rows from Hopsworks. Used to provide
    enough context for lag and rolling feature computation on new rows.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    logger.info("Reading Hopsworks rows from %s onward", cutoff.isoformat())

    df = fg.read()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df[df["timestamp"] >= cutoff].sort_values("timestamp")
    logger.info("History from Hopsworks: %d rows", len(df))
    return df


# ========================================================================
# Orchestration
# ========================================================================

def run_feature_pipeline() -> int:
    """
    Run one cycle of the hourly feature pipeline.

    Returns the number of new rows inserted into Hopsworks.
    """
    client = _make_client()

    logger.info("Step 1/4: fetching latest data from Open-Meteo")
    aq_df = fetch_recent_air_quality(client)
    weather_df = fetch_recent_weather(client)
    fresh = pd.merge(aq_df, weather_df, on="timestamp", how="inner")
    logger.info("Fresh API rows merged: %d", len(fresh))

    logger.info("Step 2/4: connecting to Hopsworks")
    fs = get_feature_store()
    fg = fs.get_feature_group(FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)

    logger.info("Step 3/4: pulling recent history for lag/rolling context")
    history = fetch_recent_history(fg)

    # We only need the raw columns from history for feature engineering;
    # the rest will be recomputed.
    raw_cols = [
        "timestamp", "pm2_5", "pm10", "no2", "o3", "so2", "co",
        "temperature", "humidity", "wind_speed", "pressure",
    ]
    history_raw = history[raw_cols] if not history.empty else pd.DataFrame(columns=raw_cols)

    combined = pd.concat([history_raw, fresh], ignore_index=True)
    combined = clean_raw_data(combined)
    engineered = engineer_features(combined)

    # Identify which rows are NEW (not already in Hopsworks)
    existing_timestamps = set(history["timestamp"]) if not history.empty else set()
    new_rows = engineered[~engineered["timestamp"].isin(existing_timestamps)]
    logger.info("Step 4/4: %d new rows to insert", len(new_rows))

    if len(new_rows) == 0:
        logger.info("No new rows to insert. API is up-to-date with feature store.")
        return 0

    fg.insert(new_rows, write_options={"wait_for_job": True})
    logger.info("Insert complete: %d rows materialized", len(new_rows))
    return len(new_rows)


def main() -> None:
    try:
        n_inserted = run_feature_pipeline()
        logger.info("Feature pipeline OK. Inserted %d rows.", n_inserted)
    except Exception as exc:
        logger.exception("Feature pipeline failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()