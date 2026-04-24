"""
Tests for utils.features.

Covers:
    - clean_raw_data: negative clipping, cap clipping, ffill, dedup, sort
    - engineer_features: time features, AQI column, rolling, lags, targets
    - has_target flag behavior
    - Idempotency (running twice gives same result)
"""

import numpy as np
import pandas as pd
import pytest

from utils.features import (
    clean_raw_data,
    engineer_features,
    get_feature_columns,
    get_target_columns,
    POLLUTANTS,
    TARGET_HORIZONS_HOURS,
)


def _make_raw_df(n_hours: int = 100, seed: int = 42) -> pd.DataFrame:
    """Build a synthetic raw DataFrame with all required columns."""
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2025-01-01", periods=n_hours, freq="h", tz="UTC")
    return pd.DataFrame({
        "timestamp":   ts,
        "pm2_5":       rng.uniform(20, 80, n_hours),
        "pm10":        rng.uniform(40, 150, n_hours),
        "no2":         rng.uniform(20, 100, n_hours),
        "o3":          rng.uniform(50, 120, n_hours),
        "so2":         rng.uniform(10, 50, n_hours),
        "co":          rng.uniform(500, 2000, n_hours),
        "temperature": rng.uniform(20, 35, n_hours),
        "humidity":    rng.uniform(40, 80, n_hours),
        "wind_speed":  rng.uniform(1, 10, n_hours),
        "pressure":    rng.uniform(1005, 1020, n_hours),
    })


class TestCleanRawData:

    def test_empty_df_returns_empty(self):
        result = clean_raw_data(pd.DataFrame())
        assert result.empty

    def test_missing_timestamp_column_raises(self):
        df = pd.DataFrame({"pm2_5": [10, 20]})
        with pytest.raises(ValueError):
            clean_raw_data(df)

    def test_negative_pollutants_clipped_to_zero(self):
        df = _make_raw_df(10)
        df.loc[5, "pm2_5"] = -50
        cleaned = clean_raw_data(df)
        assert (cleaned["pm2_5"] >= 0).all()

    def test_pm25_capped_at_physical_limit(self):
        df = _make_raw_df(10)
        df.loc[3, "pm2_5"] = 5000   # absurd sensor reading
        cleaned = clean_raw_data(df)
        assert cleaned["pm2_5"].max() <= 1000

    def test_pm10_capped_at_physical_limit(self):
        df = _make_raw_df(10)
        df.loc[3, "pm10"] = 9999
        cleaned = clean_raw_data(df)
        assert cleaned["pm10"].max() <= 1500

    def test_forward_fill_short_gaps(self):
        df = _make_raw_df(10)
        df.loc[5, "pm2_5"] = np.nan
        # Single-row NaN should be forward-filled, so row count stays 10
        cleaned = clean_raw_data(df)
        assert len(cleaned) == 10
        assert not cleaned["pm2_5"].isna().any()

    def test_long_gaps_dropped(self):
        df = _make_raw_df(10)
        df.loc[3:7, "pm2_5"] = np.nan   # 5 consecutive NaNs, > ffill limit
        cleaned = clean_raw_data(df)
        # ffill(limit=3) fills rows 3,4,5 then 6,7 remain NaN and are dropped
        assert len(cleaned) < 10
        assert not cleaned["pm2_5"].isna().any()

    def test_duplicate_timestamps_removed(self):
        df = _make_raw_df(10)
        dup = df.iloc[[0]].copy()
        df = pd.concat([df, dup], ignore_index=True)
        cleaned = clean_raw_data(df)
        assert cleaned["timestamp"].is_unique

    def test_output_sorted_by_timestamp(self):
        df = _make_raw_df(10)
        df = df.iloc[::-1].reset_index(drop=True)   # reverse order
        cleaned = clean_raw_data(df)
        ts = cleaned["timestamp"].reset_index(drop=True)
        assert ts.is_monotonic_increasing

    def test_does_not_mutate_input(self):
        df = _make_raw_df(10)
        original = df.copy()
        clean_raw_data(df)
        pd.testing.assert_frame_equal(df, original)


class TestEngineerFeatures:

    def test_missing_required_column_raises(self):
        df = pd.DataFrame({
            "timestamp": pd.date_range("2025-01-01", periods=5, freq="h", tz="UTC"),
            "pm2_5":     [10, 20, 30, 40, 50],
            # missing pm10 and others
        })
        with pytest.raises(ValueError):
            engineer_features(df)

    def test_time_features_present(self):
        df = _make_raw_df(100)
        feats = engineer_features(clean_raw_data(df))
        for col in ["hour", "day_of_week", "month", "is_weekend"]:
            assert col in feats.columns

    def test_aqi_column_computed(self):
        df = _make_raw_df(100)
        feats = engineer_features(clean_raw_data(df))
        assert "aqi" in feats.columns
        assert feats["aqi"].notna().all()

    def test_rolling_features_present(self):
        df = _make_raw_df(100)
        feats = engineer_features(clean_raw_data(df))
        for col in ["aqi_rolling_6h", "aqi_rolling_24h",
                    "rolling_30day_avg", "rolling_30day_std"]:
            assert col in feats.columns

    def test_lag_features_present(self):
        df = _make_raw_df(200)
        feats = engineer_features(clean_raw_data(df))
        for lag in [24, 48, 72]:
            assert f"aqi_lag_{lag}h" in feats.columns
            assert f"pm2_5_lag_{lag}h" in feats.columns

    def test_lag_values_are_actually_shifted(self):
        df = _make_raw_df(200)
        feats = engineer_features(clean_raw_data(df))
        # aqi at row 24 should equal aqi_lag_24h at row 48
        assert feats["aqi"].iloc[24] == feats["aqi_lag_24h"].iloc[48]

    def test_target_columns_present(self):
        df = _make_raw_df(200)
        feats = engineer_features(clean_raw_data(df))
        for h in TARGET_HORIZONS_HOURS:
            assert f"target_aqi_{h}h" in feats.columns

    def test_targets_are_future_values(self):
        df = _make_raw_df(200)
        feats = engineer_features(clean_raw_data(df))
        # target_aqi_24h at row 0 should equal aqi at row 24
        assert feats["target_aqi_24h"].iloc[0] == feats["aqi"].iloc[24]

    def test_has_target_flag_end_rows_false(self):
        df = _make_raw_df(200)
        feats = engineer_features(clean_raw_data(df))
        # Last 72 rows cannot have a 72h target → has_target must be 0
        assert feats["has_target"].tail(72).sum() == 0

    def test_has_target_flag_count(self):
        n = 200
        df = _make_raw_df(n)
        feats = engineer_features(clean_raw_data(df))
        # Exactly n - 72 rows should have all three targets available
        assert feats["has_target"].sum() == n - 72

    def test_change_rate_present(self):
        df = _make_raw_df(100)
        feats = engineer_features(clean_raw_data(df))
        assert "aqi_change_rate" in feats.columns
        # First row has no prior row → NaN
        assert pd.isna(feats["aqi_change_rate"].iloc[0])

    def test_idempotent(self):
        df = _make_raw_df(100)
        cleaned = clean_raw_data(df)
        once = engineer_features(cleaned)
        twice = engineer_features(cleaned)
        pd.testing.assert_frame_equal(once, twice)


class TestFeatureAndTargetColumnLists:

    def test_feature_columns_all_exist_after_engineering(self):
        df = _make_raw_df(200)
        feats = engineer_features(clean_raw_data(df))
        for col in get_feature_columns():
            assert col in feats.columns, f"missing feature column: {col}"

    def test_target_columns_all_exist_after_engineering(self):
        df = _make_raw_df(200)
        feats = engineer_features(clean_raw_data(df))
        for col in get_target_columns():
            assert col in feats.columns

    def test_no_overlap_between_features_and_targets(self):
        features = set(get_feature_columns())
        targets = set(get_target_columns())
        assert features.isdisjoint(targets)