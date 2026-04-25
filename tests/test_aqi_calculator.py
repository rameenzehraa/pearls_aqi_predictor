"""
Tests for utils.aqi_calculator.

Covers:
    - Breakpoint boundary conditions (AQI at exact EPA breakpoints)
    - Dominant pollutant rule (max across sub-indices)
    - Graceful handling of None, NaN, negative, and out-of-range inputs
    - Category lookup
"""

import math

import pytest

from utils.aqi_calculator import (
    compute_aqi,
    get_category,
    get_category_label,
)


class TestComputeAQI:

    def test_pm25_lower_bound_of_good(self):
        # PM2.5 = 0 → AQI = 0 (bottom of Good band)
        assert compute_aqi(pm2_5=0.0) == 0.0

    def test_pm25_upper_bound_of_good(self):
        # PM2.5 = 9.0 → AQI = 50 (top of Good band)
        assert compute_aqi(pm2_5=9.0) == 50.0

    def test_pm25_lower_bound_of_moderate(self):
        # PM2.5 = 9.1 → AQI = 51 (bottom of Moderate band)
        assert compute_aqi(pm2_5=9.1) == 51.0

    def test_pm25_midpoint_of_moderate(self):
        # PM2.5 midway through Moderate band → AQI within Moderate band
        aqi = compute_aqi(pm2_5=22.0)
        assert 51 <= aqi <= 100

    def test_dominant_pollutant_rule(self):
        # PM2.5 at Good level, PM10 at Unhealthy level → AQI = max
        aqi = compute_aqi(pm2_5=5.0, pm10=255)
        assert aqi >= 151   # should reflect PM10, not PM2.5

    def test_single_pollutant_none(self):
        # Providing only one pollutant should still work
        aqi = compute_aqi(pm2_5=35.5)
        assert aqi == 101.0

    def test_all_pollutants_none_returns_none(self):
        assert compute_aqi() is None

    def test_negative_values_ignored(self):
        # Negative PM2.5 is invalid — should be skipped, not crash
        aqi = compute_aqi(pm2_5=-5, pm10=100)
        assert aqi is not None
        assert aqi > 0

    def test_nan_ignored(self):
        aqi = compute_aqi(pm2_5=float("nan"), pm10=100)
        assert aqi is not None

    def test_above_top_breakpoint_capped_at_500(self):
        # PM2.5 = 1000 is above the top breakpoint → capped at 500
        aqi = compute_aqi(pm2_5=1000)
        assert aqi == 500.0

    def test_co_unit_conversion(self):
        # CO input in µg/m³ (Open-Meteo units). 5000 µg/m³ = 5.0 mg/m³
        # which is the top of CO's Good band → AQI = 50
        aqi = compute_aqi(co=5000)
        assert aqi == 50.0


class TestGetCategory:

    def test_good(self):
        assert get_category(25) == 1

    def test_moderate(self):
        assert get_category(75) == 2

    def test_unhealthy_for_sensitive(self):
        assert get_category(125) == 3

    def test_unhealthy(self):
        assert get_category(175) == 4

    def test_very_unhealthy(self):
        assert get_category(250) == 5

    def test_hazardous(self):
        assert get_category(400) == 6

    def test_above_500_still_hazardous(self):
        assert get_category(600) == 6

    def test_none_returns_none(self):
        assert get_category(None) is None

    def test_nan_returns_none(self):
        assert get_category(float("nan")) is None

    def test_boundary_50_is_good(self):
        # Exactly 50 is upper bound of Good
        assert get_category(50) == 1

    def test_boundary_51_is_moderate(self):
        assert get_category(51) == 2


class TestGetCategoryLabel:

    def test_moderate_label(self):
        assert get_category_label(75) == "Moderate"

    def test_hazardous_label(self):
        assert get_category_label(400) == "Hazardous"

    def test_none_returns_none(self):
        assert get_category_label(None) is None


class TestBreakpointGaps:
    """
    Regression tests: the EPA breakpoint tables have small numeric gaps
    between adjacent bands (e.g. PM10 (0-54) and (55-154) leave 54.0-55.0
    uncovered). The sub-index function must close these gaps; otherwise
    real-world readings can fall through and incorrectly cap at AQI 500.
    """

    def test_pm10_in_band_gap_does_not_cap_at_500(self):
        # 54.9 sits between (0,54) and (55,154). Should be ~50, NOT 500.
        aqi = compute_aqi(pm10=54.9)
        assert aqi is not None
        assert aqi < 100   # well within Good band, nowhere near 500

    def test_pm25_in_band_gap(self):
        # 9.05 sits between (0,9.0) and (9.1,35.4)
        aqi = compute_aqi(pm2_5=9.05)
        assert aqi is not None
        assert aqi < 60

    def test_o3_in_band_gap(self):
        # 108.5 sits between (0,108) and (109,140)
        aqi = compute_aqi(o3=108.5)
        assert aqi is not None
        assert aqi < 60

    def test_so2_in_band_gap(self):
        # 92.5 sits between (0,92) and (93,197)
        aqi = compute_aqi(so2=92.5)
        assert aqi is not None
        assert aqi < 60

    def test_no2_in_band_gap(self):
        # 100.5 sits between (0,100) and (101,188)
        aqi = compute_aqi(no2=100.5)
        assert aqi is not None
        assert aqi < 60

    def test_co_in_band_gap(self):
        # CO of 5050 µg/m³ → 5.05 mg/m³, between (0,5.0) and (5.1,10.5)
        aqi = compute_aqi(co=5050)
        assert aqi is not None
        assert aqi < 60

    def test_pm10_top_of_good_band(self):
        # PM10 = 54 → AQI exactly 50
        aqi = compute_aqi(pm10=54)
        assert aqi == 50.0

    def test_pm10_bottom_of_moderate_band(self):
        # PM10 = 55 → AQI exactly 51
        aqi = compute_aqi(pm10=55)
        assert aqi == 51.0