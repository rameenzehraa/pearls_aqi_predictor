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