"""
Experiment: how many unique pollution grid cells do our 31 subdivisions
actually resolve to on Open-Meteo's CAMS grid?

This is a one-off diagnostic to inform the 31-vs-7 architectural decision.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from collections import defaultdict
import openmeteo_requests
import requests_cache
from retry_requests import retry

from config.config import SUBDIVISIONS, SUBDIVISION_DISTRICTS


def main():
    # Set up client (same as explore_api.py)
    cache_session = requests_cache.CachedSession(".cache", expire_after=3600)
    retry_session = retry(cache_session, retries=3, backoff_factor=0.5)
    client = openmeteo_requests.Client(session=retry_session)

    url = "https://air-quality-api.open-meteo.com/v1/air-quality"

    # For each subdivision, make a minimal air-quality API call and record
    # the coordinates Open-Meteo returns (these are snapped to their grid)
    results = []
    print(f"Querying {len(SUBDIVISIONS)} subdivisions...\n")

    for name, (lat, lon) in SUBDIVISIONS.items():
        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": ["pm2_5"],  # minimal — just need to trigger grid snap
            "past_days": 0,
            "forecast_days": 1,
        }
        try:
            responses = client.weather_api(url, params=params)
            resp = responses[0]
            returned_lat = round(resp.Latitude(), 4)
            returned_lon = round(resp.Longitude(), 4)
            sent_str = f"({lat:.4f}, {lon:.4f})"
            got_str = f"({returned_lat}, {returned_lon})"
            print(f"  {name:22s}  sent {sent_str}  →  got {got_str}")
            results.append((name, lat, lon, returned_lat, returned_lon))
        except Exception as e:
            print(f"  {name:22s}  ERROR: {e}")
            results.append((name, lat, lon, None, None))

    # Group by returned grid cell
    print("\n" + "=" * 70)
    print("GRID CELL GROUPING")
    print("=" * 70)

    grid_groups = defaultdict(list)
    for name, _, _, glat, glon in results:
        if glat is not None:
            grid_groups[(glat, glon)].append(name)

    # Sort groups by size (biggest first) for readability
    sorted_groups = sorted(grid_groups.items(), key=lambda x: -len(x[1]))

    for i, (cell, members) in enumerate(sorted_groups, 1):
        districts = sorted(set(SUBDIVISION_DISTRICTS[m] for m in members))
        print(f"\nCell {i}: {cell}  ({len(members)} subdivisions, districts: {', '.join(districts)})")
        for m in members:
            print(f"   - {m:22s}  [{SUBDIVISION_DISTRICTS[m]}]")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    total = len(SUBDIVISIONS)
    unique = len(grid_groups)
    shared = sum(len(v) for v in grid_groups.values() if len(v) > 1)
    alone = sum(1 for v in grid_groups.values() if len(v) == 1)

    print(f"Total subdivisions:        {total}")
    print(f"Unique pollution cells:    {unique}")
    print(f"Subdivisions in their own cell:            {alone}")
    print(f"Subdivisions sharing with others:          {shared}")
    print(f"Ratio unique/total:        {unique/total:.2%}")

    if unique / total >= 0.85:
        verdict = "31 is legitimate — minimal grid-sharing."
    elif unique / total >= 0.45:
        verdict = "Middle ground — 31 is justified but Option C (visual collapse) helps."
    else:
        verdict = "Heavy grid-sharing — consider switching to 7 districts or the unique cells."

    print(f"\nVerdict: {verdict}")


if __name__ == "__main__":
    main()