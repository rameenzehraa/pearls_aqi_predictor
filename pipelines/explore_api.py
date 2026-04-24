"""
Exploration script — first Open-Meteo API call.

Purpose: prove we can fetch air quality + weather data for one Karachi
subdivision, and inspect the shape of the response before building anything
downstream.

Not part of the production pipeline. Delete or move to notebooks/ later.
"""

import sys
from pathlib import Path

# Add project root to Python path so we can import from config/
sys.path.insert(0, str(Path(__file__).parent.parent))

import openmeteo_requests
import requests_cache
from retry_requests import retry
import pandas as pd

from config.config import SUBDIVISIONS


def main():
    # Pick one subdivision to start — SITE is a good choice
    # (industrial area, AQI variation should be visible)
    test_subdivision = "SITE"
    lat, lon = SUBDIVISIONS[test_subdivision]
    print(f"Testing with: {test_subdivision} ({lat}, {lon})")
    print()

    # Set up Open-Meteo client with caching (1 hour) and retries
    cache_session = requests_cache.CachedSession(".cache", expire_after=3600)
    retry_session = retry(cache_session, retries=3, backoff_factor=0.5)
    client = openmeteo_requests.Client(session=retry_session)

    # ============================================================
    # 1. Air Quality API — the 6 pollutants + AQI
    # ============================================================
    aq_url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    aq_params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": [
            "pm2_5",
            "pm10",
            "nitrogen_dioxide",
            "ozone",
            "sulphur_dioxide",
            "carbon_monoxide",
            "us_aqi",
        ],
        "past_days": 2,
        "forecast_days": 1,
        "timezone": "UTC",
    }

    print("Fetching air quality data...")
    aq_responses = client.weather_api(aq_url, params=aq_params)
    aq_response = aq_responses[0]
    print(f"  Coordinates returned: {aq_response.Latitude()}, {aq_response.Longitude()}")
    print(f"  Elevation: {aq_response.Elevation()} m")
    print(f"  Timezone: {aq_response.Timezone()}")
    print()

    # Extract hourly data
    aq_hourly = aq_response.Hourly()
    aq_df = pd.DataFrame({
        "timestamp": pd.date_range(
            start=pd.to_datetime(aq_hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(aq_hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=aq_hourly.Interval()),
            inclusive="left",
        ),
        "pm2_5":               aq_hourly.Variables(0).ValuesAsNumpy(),
        "pm10":                aq_hourly.Variables(1).ValuesAsNumpy(),
        "no2":                 aq_hourly.Variables(2).ValuesAsNumpy(),
        "o3":                  aq_hourly.Variables(3).ValuesAsNumpy(),
        "so2":                 aq_hourly.Variables(4).ValuesAsNumpy(),
        "co":                  aq_hourly.Variables(5).ValuesAsNumpy(),
        "us_aqi":              aq_hourly.Variables(6).ValuesAsNumpy(),
    })

    print("=" * 60)
    print("AIR QUALITY — first 5 rows:")
    print("=" * 60)
    print(aq_df.head())
    print()
    print("AIR QUALITY — summary stats:")
    print(aq_df.describe())
    print()
    print(f"Rows: {len(aq_df)}   Columns: {list(aq_df.columns)}")
    print(f"Timestamp range: {aq_df['timestamp'].min()} to {aq_df['timestamp'].max()}")
    print(f"Null counts:\n{aq_df.isnull().sum()}")
    print()

    # ============================================================
    # 2. Weather API — temperature, humidity, wind, pressure
    # ============================================================
    weather_url = "https://api.open-meteo.com/v1/forecast"
    weather_params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": [
            "temperature_2m",
            "relative_humidity_2m",
            "wind_speed_10m",
            "pressure_msl",
        ],
        "past_days": 2,
        "forecast_days": 1,
        "timezone": "UTC",
    }

    print("Fetching weather data...")
    weather_responses = client.weather_api(weather_url, params=weather_params)
    weather_response = weather_responses[0]

    weather_hourly = weather_response.Hourly()
    weather_df = pd.DataFrame({
        "timestamp": pd.date_range(
            start=pd.to_datetime(weather_hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(weather_hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=weather_hourly.Interval()),
            inclusive="left",
        ),
        "temperature":  weather_hourly.Variables(0).ValuesAsNumpy(),
        "humidity":     weather_hourly.Variables(1).ValuesAsNumpy(),
        "wind_speed":   weather_hourly.Variables(2).ValuesAsNumpy(),
        "pressure":     weather_hourly.Variables(3).ValuesAsNumpy(),
    })

    print("=" * 60)
    print("WEATHER — first 5 rows:")
    print("=" * 60)
    print(weather_df.head())
    print()
    print(f"Rows: {len(weather_df)}   Columns: {list(weather_df.columns)}")
    print()

    # ============================================================
    # 3. Merge on timestamp
    # ============================================================
    merged = pd.merge(aq_df, weather_df, on="timestamp", how="inner")
    print("=" * 60)
    print("MERGED — first 5 rows:")
    print("=" * 60)
    print(merged.head())
    print()
    print(f"Total rows: {len(merged)}")
    print(f"Columns: {list(merged.columns)}")

    # Plausibility check — Karachi PM2.5 should usually be 30-250
    mean_pm25 = merged["pm2_5"].mean()
    print()
    print(f"Mean PM2.5 over this window: {mean_pm25:.1f} μg/m³")
    if 10 < mean_pm25 < 500:
        print("  -> Plausible range for Karachi. API working correctly.")
    else:
        print("  -> WARNING: value outside expected range. Investigate.")


if __name__ == "__main__":
    main()