"""
Experiment: do the 11 unique grid cells actually return different AQI values,
or are they all effectively the same?

If cells vary meaningfully → the case for multi-zone forecasting is valid.
If cells are near-identical → even 11 zones is performative; city-wide is honest.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import openmeteo_requests
import requests_cache
from retry_requests import retry
import pandas as pd
import numpy as np

# The 11 unique cells discovered by grid_resolution_test.py
# (cell_lat, cell_lon) with a representative human label
CELLS = {
    "Central Karachi":     (24.9, 67.0),
    "Mid-East":            (24.9, 67.1),
    "East Corridor":       (24.9, 67.2),
    "Northwest":           (25.0, 67.0),
    "New Karachi":         (25.0, 67.1),
    "Shah Murad area":     (25.0, 67.2),
    "Gadap North":         (25.1, 67.2),
    "Harbour":             (24.8, 67.0),
    "Ibrahim Hyderi area": (24.8, 67.1),
    "Korangi Core":        (24.8, 67.2),
    "Bin Qasim":           (24.8, 67.4),
}


def main():
    cache_session = requests_cache.CachedSession(".cache", expire_after=3600)
    retry_session = retry(cache_session, retries=3, backoff_factor=0.5)
    client = openmeteo_requests.Client(session=retry_session)

    url = "https://air-quality-api.open-meteo.com/v1/air-quality"

    # Fetch 7 days of past data for each cell
    results = {}
    print(f"Fetching 7-day AQI data for {len(CELLS)} cells...\n")

    for name, (lat, lon) in CELLS.items():
        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": ["pm2_5", "pm10", "us_aqi"],
            "past_days": 7,
            "forecast_days": 0,
            "timezone": "UTC",
        }
        responses = client.weather_api(url, params=params)
        resp = responses[0]
        hourly = resp.Hourly()

        df = pd.DataFrame({
            "timestamp": pd.date_range(
                start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
                end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
                freq=pd.Timedelta(seconds=hourly.Interval()),
                inclusive="left",
            ),
            "pm2_5": hourly.Variables(0).ValuesAsNumpy(),
            "pm10":  hourly.Variables(1).ValuesAsNumpy(),
            "aqi":   hourly.Variables(2).ValuesAsNumpy(),
        })
        results[name] = df
        print(f"  {name:22s}  rows: {len(df)}  mean AQI: {df['aqi'].mean():6.1f}")

    # Build a matrix of hourly AQI per cell
    combined = pd.DataFrame({
        "timestamp": results[list(CELLS.keys())[0]]["timestamp"]
    })
    for name, df in results.items():
        combined[name] = df["aqi"].values

    combined = combined.set_index("timestamp")

    print("\n" + "=" * 70)
    print("PER-CELL STATISTICS (over 7 days)")
    print("=" * 70)
    stats = pd.DataFrame({
        "mean":   combined.mean(),
        "std":    combined.std(),
        "min":    combined.min(),
        "max":    combined.max(),
        "range":  combined.max() - combined.min(),
    })
    print(stats.round(1))

    # Cross-cell spread per hour — how much do cells differ at any given time?
    print("\n" + "=" * 70)
    print("CROSS-CELL VARIATION (how much do cells differ at any given hour?)")
    print("=" * 70)
    hourly_spread = combined.max(axis=1) - combined.min(axis=1)
    print(f"  Mean hourly spread across cells:   {hourly_spread.mean():.1f} AQI units")
    print(f"  Median hourly spread:              {hourly_spread.median():.1f}")
    print(f"  Max hourly spread:                 {hourly_spread.max():.1f}")
    print(f"  Hours with spread < 10:            {(hourly_spread < 10).sum()} / {len(hourly_spread)}")
    print(f"  Hours with spread > 30:            {(hourly_spread > 30).sum()} / {len(hourly_spread)}")

    # Mean AQI gap: highest-mean cell vs lowest-mean cell
    highest = stats['mean'].idxmax()
    lowest = stats['mean'].idxmin()
    gap = stats['mean'].max() - stats['mean'].min()
    print(f"\n  Highest-mean cell:   {highest} ({stats.loc[highest, 'mean']:.1f})")
    print(f"  Lowest-mean cell:    {lowest} ({stats.loc[lowest, 'mean']:.1f})")
    print(f"  Mean-AQI gap:        {gap:.1f} AQI units")

    # Correlation matrix — are cells moving together or independently?
    print("\n" + "=" * 70)
    print("CELL-TO-CELL CORRELATION (values closer to 1.0 = cells move together)")
    print("=" * 70)
    corr = combined.corr()
    print(corr.round(2))
    mean_offdiag_corr = (corr.values.sum() - len(corr)) / (len(corr)**2 - len(corr))
    print(f"\n  Mean off-diagonal correlation: {mean_offdiag_corr:.2f}")

    # Verdict
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    if gap > 20 and mean_offdiag_corr < 0.90:
        v = "Cells meaningfully differ. Multi-zone forecasting is justified."
    elif gap > 10:
        v = "Cells differ moderately. Multi-zone forecasting has some value."
    else:
        v = "Cells are near-identical. City-wide forecast is more honest."
    print(f"  {v}")


if __name__ == "__main__":
    main()