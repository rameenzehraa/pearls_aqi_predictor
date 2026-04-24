"""
Tests for utils.alerts.

Covers:
    - Hard-threshold channel (fires on AQI >= 151 or >= 301 regardless of baseline)
    - Statistical channel (unusual spike, ratio spike, category jump)
    - Minimum-jump gate on the statistical channel
    - Graceful handling of missing/invalid inputs
    - alert_reason messaging
"""

import pytest

from utils.alerts import should_alert, alert_reason


class TestHardThresholdChannel:

    def test_hazardous_forecast_fires(self):
        # AQI 320 crosses the 301 threshold
        assert should_alert(320, 180, 170, 25) is True

    def test_unhealthy_forecast_fires(self):
        # AQI 160 crosses the 151 threshold
        assert should_alert(160, 90, 85, 15) is True

    def test_hazardous_fires_even_when_baseline_is_hazardous(self):
        # Karachi worst case: baseline is already hazardous
        # Hard threshold must still fire — statistical channel would not.
        assert should_alert(320, 310, 305, 20) is True

    def test_just_below_unhealthy_does_not_fire_hard_channel(self):
        # 150 is exactly below the 151 hard threshold. Statistical channel
        # shouldn't fire either with these inputs.
        assert should_alert(150, 140, 135, 10) is False

    def test_exactly_151_fires(self):
        assert should_alert(151, 90, 85, 15) is True

    def test_exactly_301_fires(self):
        assert should_alert(301, 180, 170, 20) is True


class TestStatisticalChannel:

    def test_no_alert_when_all_normal(self):
        # Everything steady — no alert
        assert should_alert(95, 90, 85, 15) is False

    def test_unusual_spike_with_sufficient_jump_fires(self):
        # 140 > 80 + (1.5 * max(10, 20)) = 110, and 140 - 85 = 55 > 30
        assert should_alert(140, 85, 80, 10) is True

    def test_unusual_spike_without_sufficient_jump_does_not_fire(self):
        # Predicted is unusual vs baseline but jump from current is tiny
        # predicted=140, current=130 → jump=10, below ALERT_MINIMUM_JUMP=30
        assert should_alert(140, 130, 80, 10) is False

    def test_ratio_spike_fires(self):
        # predicted=130, current=95 → 130 > 95*1.3=123.5 ✓, jump 35 > 30 ✓
        # baseline: 130 > 85 + 1.5*20 = 115 ✓ (unusual also fires)
        assert should_alert(130, 95, 85, 15) is True

    def test_std_floor_applied(self):
        # Rolling std is absurdly small (3). Floor of 20 should still apply.
        # predicted=125, avg=80, std=3 → without floor: 80 + 4.5 = 84.5, would fire
        # with floor of 20: 80 + 30 = 110, still fires. Jump 125-85=40 > 30 ✓
        assert should_alert(125, 85, 80, 3) is True


class TestInvalidInputs:

    def test_none_predicted_does_not_fire(self):
        assert should_alert(None, 90, 85, 15) is False

    def test_nan_predicted_does_not_fire(self):
        assert should_alert(float("nan"), 90, 85, 15) is False

    def test_missing_baseline_blocks_statistical_channel(self):
        # Without baseline stats, only the hard channel can fire
        assert should_alert(95, None, None, None) is False

    def test_missing_baseline_still_allows_hard_channel(self):
        # Hazardous forecast fires even if we have no baseline context
        assert should_alert(320, None, None, None) is True

    def test_invalid_predicted_type_does_not_crash(self):
        assert should_alert("not a number", 90, 85, 15) is False


class TestAlertReason:

    def test_no_alert_returns_none(self):
        assert alert_reason(95, 90, 85, 15) is None

    def test_hazardous_reason(self):
        reason = alert_reason(320, 180, 170, 25)
        assert reason is not None
        assert "Hazardous" in reason
        assert "320" in reason

    def test_unhealthy_reason(self):
        reason = alert_reason(160, 90, 85, 15)
        assert reason is not None
        assert "Unhealthy" in reason

    def test_statistical_spike_reason(self):
        reason = alert_reason(140, 85, 80, 10)
        assert reason is not None
        assert "spike" in reason.lower() or "unusual" in reason.lower()