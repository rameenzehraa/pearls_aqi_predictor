"""
Backfill script — one-time seeding of historical data.

Fetches ~1 year of hourly air quality + weather data for Karachi from
Open-Meteo's archive endpoints, cleans it, engineers features, and
writes to:

    1. Hopsworks feature group `aqi_features` v1  (PRIMARY destination,
       used by all production training and serving code)
    2. Local parquet at data/backfill.parquet     (BACKUP only — never
       read by training, feature, backend, or frontend code paths.
       Exists for disaster recovery and EDA notebook speed.)

Architecture rule (Draft 6, coordinator MOM): production code reads
from Hopsworks only. The local file is a cold backup, not a cache.

Usage:
    python -m pipelines.backfill
    python -m pipelines.backfill --start 2025-04-25 --end 2026-04-24
    python -m pipelines.backfill --skip-hopsworks   (dev/debug only)
    python -m pipelines.backfill --skip-local       (Hopsworks only)
"""

import argparse
import sys
from datetime import date, timedelta
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
    BACKFILL_START_DATE,
    KARACHI_LAT,
    KARACHI_LON,
)
from utils.features import clean_raw_data, engineer_features
from utils.logger import get_logger

logger = get_logger(__name__)

# Open-Meteo archive endpoints (undocumented but stable for air quality;
# official for weather).
AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
WEATHER_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

# Local backup path. Gitignored. NEVER read by production code.
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data"
LOCAL_BACKUP_PATH = OUTPUT_DIR / "backfill.parquet"

# Hopsworks feature group config
FEATURE_GROUP_NAME = "aqi_features"
FEATURE_GROUP_VERSION = 1
PRIMARY_KEY = ["timestamp"]
EVENT_TIME_COLUMN = "timestamp"
FEATURE_GROUP_DESCRIPTION = (
    "Hourly Karachi AQI features: pollutants, weather, time-of-day, "
    "rolling stats, lagged values, and 24/48/72h AQI targets."
)


# ========================================================================
# API client
# ========================================================================

def _make_client() -> openmeteo_requests.Client:
    """Cached + retrying Open-Meteo client."""
    cache_session = requests_cache.CachedSession(".cache", expire_after=3600)
    retry_session = retry(
        cache_session,
        retries=API_RETRY_COUNT,
        backoff_factor=API_RETRY_BASE_DELAY / 10,
    )
    return openmeteo_requests.Client(session=retry_session)


# ========================================================================
# Fetchers
# ========================================================================

def fetch_air_quality(
    client: openmeteo_requests.Client,
    lat: float,
    lon: float,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Pull historical hourly air quality for the given date range."""
    logger.info(
        "Fetching air quality: lat=%s lon=%s start=%s end=%s",
        lat, lon, start_date, end_date,
    )

    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": [
            "pm2_5",
            "pm10",
            "nitrogen_dioxide",
            "ozone",
            "sulphur_dioxide",
            "carbon_monoxide",
        ],
        "start_date": start_date,
        "end_date": end_date,
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

    logger.info("Air quality: %d rows fetched", len(df))
    return df


def fetch_weather(
    client: openmeteo_requests.Client,
    lat: float,
    lon: float,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Pull historical hourly weather (ERA5 reanalysis)."""
    logger.info(
        "Fetching weather archive: lat=%s lon=%s start=%s end=%s",
        lat, lon, start_date, end_date,
    )

    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": [
            "temperature_2m",
            "relative_humidity_2m",
            "wind_speed_10m",
            "pressure_msl",
        ],
        "start_date": start_date,
        "end_date": end_date,
        "timezone": "UTC",
    }

    responses = client.weather_api(WEATHER_ARCHIVE_URL, params=params)
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

    logger.info("Weather: %d rows fetched", len(df))
    return df


# ========================================================================
# Orchestration
# ========================================================================

def run_backfill(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Full backfill: fetch, merge, clean, engineer features.
    Returns the final feature DataFrame.
    """
    client = _make_client()

    aq_df = fetch_air_quality(client, KARACHI_LAT, KARACHI_LON, start_date, end_date)
    weather_df = fetch_weather(client, KARACHI_LAT, KARACHI_LON, start_date, end_date)

    logger.info("Merging air quality + weather on timestamp")
    merged = pd.merge(aq_df, weather_df, on="timestamp", how="inner")
    logger.info("Merged: %d rows, %d columns", len(merged), len(merged.columns))

    logger.info("Cleaning raw data")
    cleaned = clean_raw_data(merged)
    logger.info("After cleaning: %d rows (%d dropped)",
                len(cleaned), len(merged) - len(cleaned))

    logger.info("Engineering features")
    features = engineer_features(cleaned)
    logger.info(
        "Feature DataFrame: %d rows, %d columns, has_target sum = %d",
        len(features), len(features.columns), int(features["has_target"].sum()),
    )

    return features


def write_to_hopsworks(df: pd.DataFrame) -> None:
    """Write feature DataFrame to the Hopsworks feature store (PRIMARY)."""
    # Imported lazily so the script can still run with --skip-hopsworks
    # if Hopsworks credentials aren't available.
    from utils.hopsworks_client import get_feature_store

    logger.info("Connecting to Hopsworks feature store")
    fs = get_feature_store()

    logger.info(
        "Getting or creating feature group: %s (v%d)",
        FEATURE_GROUP_NAME, FEATURE_GROUP_VERSION,
    )
    fg = fs.get_or_create_feature_group(
        name=FEATURE_GROUP_NAME,
        version=FEATURE_GROUP_VERSION,
        primary_key=PRIMARY_KEY,
        event_time=EVENT_TIME_COLUMN,
        description=FEATURE_GROUP_DESCRIPTION,
        online_enabled=False,
    )

    logger.info("Inserting %d rows into Hopsworks", len(df))
    fg.insert(df, write_options={"wait_for_job": True})
    logger.info("Hopsworks insert complete.")


def write_local_backup(df: pd.DataFrame, path: Path = LOCAL_BACKUP_PATH) -> None:
    """
    Write feature DataFrame to a local parquet file.

    BACKUP ONLY. Never read by training, feature, backend, or frontend
    code paths. Used for disaster recovery and EDA notebook speed.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    logger.info("Local backup written to %s (%d rows)", path, len(df))


# ========================================================================
# CLI
# ========================================================================

def _parse_args() -> argparse.Namespace:
    default_end = (date.today() - timedelta(days=1)).isoformat()
    parser = argparse.ArgumentParser(description="Backfill Karachi AQI features")
    parser.add_argument("--start", default=BACKFILL_START_DATE,
                        help="Start date (YYYY-MM-DD). Default from config.")
    parser.add_argument("--end", default=default_end,
                        help="End date (YYYY-MM-DD). Default: yesterday.")
    parser.add_argument("--skip-hopsworks", action="store_true",
                        help="Skip Hopsworks write (dev/debug only).")
    parser.add_argument("--skip-local", action="store_true",
                        help="Skip local backup write (Hopsworks only).")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    logger.info("Starting backfill: %s to %s", args.start, args.end)

    if args.skip_hopsworks and args.skip_local:
        logger.error("Cannot skip both Hopsworks and local. Aborting.")
        sys.exit(1)

    try:
        features = run_backfill(args.start, args.end)

        if not args.skip_hopsworks:
            write_to_hopsworks(features)
        else:
            logger.warning("--skip-hopsworks set; Hopsworks write skipped.")

        if not args.skip_local:
            write_local_backup(features)
        else:
            logger.warning("--skip-local set; local backup skipped.")

        logger.info("Backfill complete.")
    except Exception as exc:
        logger.exception("Backfill failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()