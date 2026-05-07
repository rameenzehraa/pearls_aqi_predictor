"""
Day 11 Task 3 — Quantile prediction intervals around Ridge champion.

Approach: residual-based quantile regression.
  1. Load saved Ridge champion (point predictions already validated).
  2. Compute residuals on training data: r = y_true - y_pred.
  3. Train two QuantileRegressors to predict p10 and p90 of those
     residuals from the same 5 features.
  4. Final 80% interval: [point + p10_residual, point + p90_residual].
  5. Evaluate coverage on holdout.

Usage:
    python -m models.quantile_intervals
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import QuantileRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.hopsworks_client import get_feature_store
from utils.logger import get_logger
from utils.metrics import quantile_coverage

logger = get_logger(__name__)

CHAMPION_DIR = Path("models/champion")
CHAMPION_FEATURES = ["aqi", "month", "aqi_lag_24h", "aqi_lag_72h", "humidity"]
HORIZONS = [24, 48, 72]
HOLDOUT_FRAC = 0.20
LOW_Q = 0.10
HIGH_Q = 0.90
NOMINAL_COVERAGE = HIGH_Q - LOW_Q   # 0.80

# QuantileRegressor needs an alpha (regularization). Higher = wider intervals.
# Start at 0.001 (very mild) — we want to let the data speak.
QR_ALPHA = 0.001


def load_data() -> pd.DataFrame:
    fs = get_feature_store()
    fg = fs.get_feature_group("aqi_features", version=1)
    df = fg.read()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    logger.info("Loaded %d rows from Hopsworks", len(df))
    return df


def load_champion(horizon: int):
    horizon_dir = CHAMPION_DIR / f"h{horizon}"
    model = joblib.load(horizon_dir / "ridge_model.joblib")
    scaler = joblib.load(horizon_dir / "scaler.joblib")
    with open(horizon_dir / "feature_medians.json") as f:
        medians_dict = json.load(f)
    medians = pd.Series(medians_dict)
    return model, scaler, medians


def split_and_impute(df: pd.DataFrame, target_col: str, medians: pd.Series):
    work = df.dropna(subset=[target_col]).reset_index(drop=True)
    holdout_start = int(len(work) * (1 - HOLDOUT_FRAC))
    train = work.iloc[:holdout_start].copy()
    test  = work.iloc[holdout_start:].copy()
    train[CHAMPION_FEATURES] = train[CHAMPION_FEATURES].fillna(medians)
    test[CHAMPION_FEATURES]  = test[CHAMPION_FEATURES].fillna(medians)
    return train, test


def fit_quantile(X: np.ndarray, residuals: np.ndarray, q: float) -> QuantileRegressor:
    """Train a single quantile regressor."""
    qr = QuantileRegressor(quantile=q, alpha=QR_ALPHA, solver="highs")
    qr.fit(X, residuals)
    return qr


def evaluate_horizon(df: pd.DataFrame, horizon: int) -> dict:
    target_col = f"target_aqi_{horizon}h"
    logger.info("─" * 50)
    logger.info("Horizon %dh", horizon)

    ridge, scaler, medians = load_champion(horizon)
    train, test = split_and_impute(df, target_col, medians)

    X_train = train[CHAMPION_FEATURES].values
    y_train = train[target_col].values
    X_test  = test[CHAMPION_FEATURES].values
    y_test  = test[target_col].values

    X_train_s = scaler.transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # Ridge point predictions (used as the centre of intervals)
    y_train_pred = ridge.predict(X_train_s)
    y_test_pred  = ridge.predict(X_test_s)

    # Residuals on training data — what the quantile regressors learn from
    residuals_train = y_train - y_train_pred

    logger.info(
        "Residual stats (train) — mean=%.2f std=%.2f min=%.2f max=%.2f",
        residuals_train.mean(),
        residuals_train.std(),
        residuals_train.min(),
        residuals_train.max(),
    )

    # Train two quantile regressors on the residuals
    qr_lo = fit_quantile(X_train_s, residuals_train, LOW_Q)
    qr_hi = fit_quantile(X_train_s, residuals_train, HIGH_Q)

    # Predict residual quantiles on holdout, build intervals
    resid_lo = qr_lo.predict(X_test_s)
    resid_hi = qr_hi.predict(X_test_s)

    y_lo = y_test_pred + resid_lo
    y_hi = y_test_pred + resid_hi

    # Coverage on holdout
    coverage = quantile_coverage(y_test, y_lo, y_hi)
    avg_width = float(np.mean(y_hi - y_lo))

    # Save the quantile models
    horizon_dir = CHAMPION_DIR / f"h{horizon}"
    joblib.dump(qr_lo, horizon_dir / "quantile_p10.joblib")
    joblib.dump(qr_hi, horizon_dir / "quantile_p90.joblib")

    return {
        "horizon": horizon,
        "coverage": coverage,
        "nominal": NOMINAL_COVERAGE,
        "avg_interval_width": round(avg_width, 2),
        "n_test": len(y_test),
    }


def main() -> None:
    logger.info("=" * 60)
    logger.info("Day 11 Task 3 — Quantile prediction intervals")
    logger.info("=" * 60)

    df = load_data()

    results = []
    for h in HORIZONS:
        res = evaluate_horizon(df, h)
        results.append(res)

    print("\n" + "=" * 70)
    print("QUANTILE INTERVAL COVERAGE (target = 80%)")
    print("=" * 70)
    print(f"{'Horizon':<10} {'Coverage':<12} {'Nominal':<10} {'Avg width':<12} {'N':<8}")
    print("-" * 70)
    for r in results:
        cov_pct = round(r["coverage"] * 100, 2)
        nom_pct = round(r["nominal"] * 100, 1)
        marker = "✓" if 0.75 <= r["coverage"] <= 0.85 else "✗"
        print(
            f"{r['horizon']}h{'':<8} "
            f"{cov_pct:<6}% {marker:<4} "
            f"{nom_pct:<6}%   "
            f"{r['avg_interval_width']:<12} "
            f"{r['n_test']:<8}"
        )
    print("=" * 70)
    print(f"Day 9 baseline (XGBoost on raw target, 24h only): 57.66% coverage")


if __name__ == "__main__":
    main()