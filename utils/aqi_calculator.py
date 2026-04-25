"""
AQI calculation from raw pollutant concentrations.

Uses the US EPA piecewise linear interpolation formula:

    AQI = ((I_hi - I_lo) / (C_hi - C_lo)) * (C - C_lo) + I_lo

where (C_lo, C_hi) is the concentration breakpoint bracketing C, and
(I_lo, I_hi) is the corresponding AQI breakpoint.

The overall AQI at a timestamp is the MAXIMUM of the sub-indices across
all pollutants (US EPA "dominant pollutant" rule).

Units expected from Open-Meteo:
    pm2_5, pm10, so2, no2, o3   — µg/m³
    co                           — µg/m³ (converted to mg/m³ internally)

Averaging windows per EPA:
    pm2_5   — 24h
    pm10    — 24h
    o3      — 8h
    co      — 8h
    so2     — 1h
    no2     — 1h

The feature pipeline is responsible for supplying already-averaged values.
This module applies the breakpoint lookup only.
"""

from typing import Optional

from config.config import AQI_CATEGORIES

# ========================================================================
# EPA breakpoints
# Each entry: (C_lo, C_hi, I_lo, I_hi)
# Concentrations in µg/m³ except CO which is in mg/m³
# Source: US EPA Technical Assistance Document (May 2024)
# ========================================================================

BREAKPOINTS = {
    "pm2_5": [
        (0.0,   9.0,   0,   50),
        (9.1,   35.4,  51,  100),
        (35.5,  55.4,  101, 150),
        (55.5,  125.4, 151, 200),
        (125.5, 225.4, 201, 300),
        (225.5, 325.4, 301, 500),
    ],
    "pm10": [
        (0,   54,   0,   50),
        (55,  154,  51,  100),
        (155, 254,  101, 150),
        (255, 354,  151, 200),
        (355, 424,  201, 300),
        (425, 604,  301, 500),
    ],
    # O3 8-hour (µg/m³). EPA uses ppb; µg/m³ values derived at 25°C, 1 atm.
    "o3": [
        (0,   108, 0,   50),
        (109, 140, 51,  100),
        (141, 170, 101, 150),
        (171, 210, 151, 200),
        (211, 400, 201, 300),
    ],
    # CO 8-hour (mg/m³). EPA breakpoints in ppm converted at 25°C, 1 atm.
    "co": [
        (0.0,  5.0,  0,   50),
        (5.1,  10.5, 51,  100),
        (10.6, 14.0, 101, 150),
        (14.1, 17.5, 151, 200),
        (17.6, 34.9, 201, 300),
        (35.0, 57.5, 301, 500),
    ],
    # SO2 1-hour (µg/m³). Converted from ppb at 25°C, 1 atm.
    "so2": [
        (0,   92,   0,   50),
        (93,  197,  51,  100),
        (198, 484,  101, 150),
        (485, 796,  151, 200),
        (797, 1583, 201, 300),
    ],
    # NO2 1-hour (µg/m³). Converted from ppb at 25°C, 1 atm.
    "no2": [
        (0,    100,  0,   50),
        (101,  188,  51,  100),
        (189,  677,  101, 150),
        (678,  1220, 151, 200),
        (1221, 2349, 201, 300),
    ],
}


def _sub_index(concentration: float, breakpoints: list) -> Optional[float]:
    """
    Compute the AQI sub-index for a single pollutant via piecewise linear
    interpolation against its breakpoint table.

    The breakpoint table has small gaps between adjacent bands (e.g. PM10's
    upper bound 54 and next lower bound 55). To avoid a concentration
    falling into none of the bands, we treat each band as covering the
    range [c_lo, next_c_lo), with the final band closed on both ends.

    Returns None if concentration is negative or NaN.
    Concentrations above the top breakpoint are capped at 500.
    """
    if concentration is None:
        return None
    try:
        c = float(concentration)
    except (TypeError, ValueError):
        return None
    if c != c:            # NaN check
        return None
    if c < 0:
        return None

    n = len(breakpoints)
    for i, (c_lo, c_hi, i_lo, i_hi) in enumerate(breakpoints):
        # Use the next band's lower bound as this band's exclusive upper
        # bound, closing the gap. For the final band, use its own c_hi
        # inclusively.
        upper = breakpoints[i + 1][0] if i + 1 < n else c_hi
        is_final = i + 1 == n
        in_band = (c_lo <= c <= upper) if is_final else (c_lo <= c < upper)
        if in_band:
            return ((i_hi - i_lo) / (c_hi - c_lo)) * (c - c_lo) + i_lo

    # Above the highest breakpoint — cap at 500
    return 500.0


def compute_aqi(
    pm2_5: Optional[float] = None,
    pm10:  Optional[float] = None,
    o3:    Optional[float] = None,
    co:    Optional[float] = None,
    so2:   Optional[float] = None,
    no2:   Optional[float] = None,
) -> Optional[float]:
    """
    Compute the overall AQI as the maximum sub-index across available
    pollutants (US EPA dominant pollutant rule).

    CO is accepted in µg/m³ (as returned by Open-Meteo) and converted to
    mg/m³ internally for the breakpoint lookup.

    Returns None if no valid pollutant readings are provided.
    """
    # Open-Meteo returns CO in µg/m³; EPA breakpoints are in mg/m³
    co_mg = None if co is None else co / 1000.0

    sub_indices = [
        _sub_index(pm2_5, BREAKPOINTS["pm2_5"]),
        _sub_index(pm10,  BREAKPOINTS["pm10"]),
        _sub_index(o3,    BREAKPOINTS["o3"]),
        _sub_index(co_mg, BREAKPOINTS["co"]),
        _sub_index(so2,   BREAKPOINTS["so2"]),
        _sub_index(no2,   BREAKPOINTS["no2"]),
    ]

    valid = [s for s in sub_indices if s is not None]
    if not valid:
        return None

    return round(max(valid), 2)


def get_category(aqi: Optional[float]) -> Optional[int]:
    """
    Return the AQI category ID (1-6) for a given AQI value.

    Returns None if aqi is None or out of range.
    """
    if aqi is None:
        return None
    try:
        aqi_val = float(aqi)
    except (TypeError, ValueError):
        return None
    if aqi_val != aqi_val:
        return None

    for (lo, hi), (cat_id, _label) in AQI_CATEGORIES.items():
        if lo <= aqi_val <= hi:
            return cat_id

    # Above 500 — treat as Hazardous
    if aqi_val > 500:
        return 6
    return None


def get_category_label(aqi: Optional[float]) -> Optional[str]:
    """Return the AQI category label (e.g. 'Moderate') for a given AQI."""
    if aqi is None:
        return None
    try:
        aqi_val = float(aqi)
    except (TypeError, ValueError):
        return None
    if aqi_val != aqi_val:
        return None

    for (lo, hi), (_cat_id, label) in AQI_CATEGORIES.items():
        if lo <= aqi_val <= hi:
            return label

    if aqi_val > 500:
        return "Hazardous"
    return None