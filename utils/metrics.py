"""
Regression metrics + tabular reporting helpers.

Used by all model training scripts to evaluate forecasts on the holdout
set and print a clean, scannable per-horizon comparison.
"""

from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(
    y_true: Iterable[float],
    y_pred: Iterable[float],
) -> Dict[str, float]:
    """
    Compute RMSE, MAE, R² for a single (y_true, y_pred) pair.

    Returns a dict with rounded values:
        {"rmse": ..., "mae": ..., "r2": ...}
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if len(y_true) == 0:
        return {"rmse": float("nan"), "mae": float("nan"), "r2": float("nan")}

    return {
        "rmse": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 3),
        "mae":  round(float(mean_absolute_error(y_true, y_pred)), 3),
        "r2":   round(float(r2_score(y_true, y_pred)), 3),
    }


def metrics_table(results: Dict[str, Dict[str, Dict[str, float]]]) -> pd.DataFrame:
    """
    Build a tidy DataFrame from a nested results dict.

    Input shape:
        {
            "Naive":   {"24h": {"rmse": .., "mae": .., "r2": ..}, "48h": {...}, "72h": {...}},
            "Ridge":   {...},
            "RF":      {...},
            "XGBoost": {...},
        }

    Output: long-form DataFrame with columns
        model | horizon | rmse | mae | r2
    """
    rows = []
    for model_name, by_horizon in results.items():
        for horizon, metrics in by_horizon.items():
            rows.append({
                "model":   model_name,
                "horizon": horizon,
                "rmse":    metrics.get("rmse"),
                "mae":     metrics.get("mae"),
                "r2":      metrics.get("r2"),
            })
    return pd.DataFrame(rows)


def print_metrics_by_horizon(
    results: Dict[str, Dict[str, Dict[str, float]]],
    horizons: Optional[Iterable[str]] = None,
) -> None:
    """
    Print one narrow table per horizon for easy scanning.

    Each horizon's table has columns: Model | RMSE | MAE | R²
    """
    df = metrics_table(results)
    horizons = horizons or sorted(df["horizon"].unique())

    for h in horizons:
        sub = df[df["horizon"] == h][["model", "rmse", "mae", "r2"]].copy()
        sub.columns = ["Model", "RMSE", "MAE", "R²"]
        print(f"\nHorizon = {h}")
        print("-" * 40)
        print(sub.to_string(index=False))


def quantile_coverage(
    y_true: Iterable[float],
    y_lo:   Iterable[float],
    y_hi:   Iterable[float],
) -> float:
    """
    Fraction of y_true values that fall within [y_lo, y_hi].

    For a properly calibrated 80% prediction interval (p10 to p90),
    this should be close to 0.80.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_lo   = np.asarray(y_lo,   dtype=float)
    y_hi   = np.asarray(y_hi,   dtype=float)

    mask = ~np.isnan(y_true) & ~np.isnan(y_lo) & ~np.isnan(y_hi)
    if mask.sum() == 0:
        return float("nan")

    inside = (y_true[mask] >= y_lo[mask]) & (y_true[mask] <= y_hi[mask])
    return round(float(inside.mean()), 4)

evaluate = regression_metrics
print_metrics_table = print_metrics_by_horizon