"""
Alert decision logic for forecasted AQI.

Two independent alert channels:

    1. Hard threshold — fires whenever predicted AQI enters a dangerous
       EPA category (Unhealthy or above), regardless of baseline. Needed
       because Karachi's baseline is often already elevated and a purely
       statistical alert could miss objectively dangerous air.

    2. Statistical — fires when the forecast is unusual relative to the
       30-day rolling baseline AND represents a meaningful jump from the
       current reading. Designed to catch spikes against a person's
       normal experience of the air.

An alert fires if EITHER channel triggers.

All thresholds come from config.config — never hardcoded here.
"""

from typing import Optional

from config.config import (
    ALERT_HARD_THRESHOLDS,
    ALERT_MINIMUM_JUMP,
    ALERT_SPIKE_RATIO,
    ALERT_STD_FLOOR,
    ALERT_STD_MULTIPLIER,
)
from utils.aqi_calculator import get_category


def _is_valid_number(x) -> bool:
    """True if x is a finite real number."""
    if x is None:
        return False
    try:
        val = float(x)
    except (TypeError, ValueError):
        return False
    return val == val and val not in (float("inf"), float("-inf"))


def should_alert(
    predicted_aqi: Optional[float],
    current_aqi: Optional[float],
    rolling_avg: Optional[float],
    rolling_std: Optional[float],
) -> bool:
    """
    Decide whether the predicted AQI warrants an alert.

    Args:
        predicted_aqi: Model's forecast for the horizon in question.
        current_aqi:   Most recent observed AQI.
        rolling_avg:   30-day rolling mean AQI (baseline).
        rolling_std:   30-day rolling std of AQI.

    Returns:
        True if either the hard-threshold or statistical channel fires.
    """
    if not _is_valid_number(predicted_aqi):
        return False

    predicted_aqi = float(predicted_aqi)

    # ----- Channel 1: hard threshold -----
    # Fires if predicted AQI crosses any EPA danger threshold, regardless
    # of current/baseline conditions.
    if any(predicted_aqi >= t for t in ALERT_HARD_THRESHOLDS):
        return True

    # ----- Channel 2: statistical spike -----
    # Requires all three baseline inputs to be valid.
    if not all(_is_valid_number(x) for x in (current_aqi, rolling_avg, rolling_std)):
        return False

    current_aqi = float(current_aqi)
    rolling_avg = float(rolling_avg)
    rolling_std = float(rolling_std)

    effective_std = max(rolling_std, ALERT_STD_FLOOR)
    unusual       = predicted_aqi > rolling_avg + (ALERT_STD_MULTIPLIER * effective_std)
    spike         = predicted_aqi > current_aqi * ALERT_SPIKE_RATIO
    category_jump = (
        (get_category(predicted_aqi) or 0) > (get_category(current_aqi) or 0)
    )
    minimum_jump  = (predicted_aqi - current_aqi) > ALERT_MINIMUM_JUMP

    return (unusual or spike or category_jump) and minimum_jump


def alert_reason(
    predicted_aqi: Optional[float],
    current_aqi: Optional[float],
    rolling_avg: Optional[float],
    rolling_std: Optional[float],
) -> Optional[str]:
    """
    Return a short human-readable reason for the alert, or None if no
    alert would fire. Useful for the dashboard banner.
    """
    if not should_alert(predicted_aqi, current_aqi, rolling_avg, rolling_std):
        return None

    predicted_aqi = float(predicted_aqi)

    # Hard threshold takes precedence in messaging
    for t in sorted(ALERT_HARD_THRESHOLDS, reverse=True):
        if predicted_aqi >= t:
            if t >= 301:
                return f"Hazardous air predicted (AQI {predicted_aqi:.0f})"
            if t >= 151:
                return f"Unhealthy air predicted (AQI {predicted_aqi:.0f})"

    # Statistical reasons
    if _is_valid_number(current_aqi):
        jump = predicted_aqi - float(current_aqi)
        return f"Significant spike predicted (+{jump:.0f} AQI from current)"

    return "Unusual air quality predicted"