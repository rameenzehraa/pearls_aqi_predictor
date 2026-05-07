"""
Day 11 Task 4 — Out-of-time sanity check on the champion.

Loads the saved Ridge champion, evaluates it on the most recent
~120 rows that landed AFTER the original holdout split. Compares
out-of-time metrics to the holdout metrics from training.

Usage:
    python -m models.oot_sanity_check
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.hopsworks_client import get_feature_store
from utils.logger import get_logger

logger = get_logger(__name__)

CHAMPION_DIR = Path("models/champion")
CHAMPION_FEATURES = ["aqi", "month", "aqi_lag_24h", "aqi_lag_72h", "humidity"]
HORIZONS = [24, 48, 72]
HOLDOUT_FRAC = 0.20


def load_data() -> pd.DataFrame:
    logger.info("Connecting to Hopsworks …")
    fs = get_feature_store()
    fg = fs.get_feature_group("aqi_features", version=1)
    df = fg.read()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    logger.info("Loaded %d rows from Hopsworks", len(df))
    return df


def load_champion(horizon: int):
    """Load the saved Ridge model, scaler, and feature medians."""
    horizon_dir = CHAMPION_DIR / f"h{horizon}"
    model = joblib.load(horizon_dir / "ridge_model.joblib")
    scaler = joblib.load(horizon_dir / "scaler.joblib")
    with open(horizon_dir / "feature_medians.json") as f:
        medians_dict = json.load(f)
    medians = pd.Series(medians_dict)
    return model, scaler, medians


def predict_and_score(
    df_oot: pd.DataFrame, target_col: str, model, scaler, medians
) -> dict:
    """Apply same preprocessing as training, predict, score."""
    work = df_oot.dropna(subset=[target_col]).copy()
    if len(work) == 0:
        return {"n": 0, "r2": None, "rmse": None, "mae": None}

    work[CHAMPION_FEATURES] = work[CHAMPION_FEATURES].fillna(medians)

    X = work[CHAMPION_FEATURES].values
    y_true = work[target_col].values
    X_s = scaler.transform(X)
    y_pred = model.predict(X_s)

    return {
        "n": len(work),
        "r2":   round(float(r2_score(y_true, y_pred)), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 2),
        "mae":  round(float(mean_absolute_error(y_true, y_pred)), 2),
    }


def main() -> None:
    logger.info("=" * 60)
    logger.info("Day 11 Task 4 — Out-of-time sanity check")
    logger.info("=" * 60)

    # Load metadata to know what timestamp the champion was trained against
    with open(CHAMPION_DIR / "champion_metadata.json") as f:
        metadata = json.load(f)
    n_rows_at_training = metadata["n_rows_loaded"]
    logger.info("Champion was trained on %d rows", n_rows_at_training)

    df = load_data()
    n_now = len(df)
    n_new = n_now - n_rows_at_training
    logger.info("Current rows: %d | New since training: %d", n_now, n_new)

    if n_new <= 0:
        logger.warning(
            "No new rows since champion was trained. OOT check is meaningless."
        )
        return

    # OOT slice: everything after the training cutoff
    df_oot = df.iloc[n_rows_at_training:].reset_index(drop=True)
    logger.info(
        "OOT window: %s → %s",
        df_oot["timestamp"].min(),
        df_oot["timestamp"].max(),
    )

    # Holdout-baseline metrics (from training metadata) for side-by-side
    holdout_metrics = {h["horizon"]: h["metrics"] for h in metadata["horizons"]}

    print("\n" + "=" * 70)
    print("OUT-OF-TIME PERFORMANCE vs HOLDOUT BASELINE")
    print("=" * 70)
    print(f"{'Horizon':<8} {'Holdout R²':<12} {'OOT R²':<10} {'Δ R²':<10} {'OOT N':<8}")
    print("-" * 70)

    oot_results = []
    for h in HORIZONS:
        target_col = f"target_aqi_{h}h"
        model, scaler, medians = load_champion(h)
        oot = predict_and_score(df_oot, target_col, model, scaler, medians)

        holdout_r2 = holdout_metrics[h]["r2"]
        oot_r2 = oot["r2"] if oot["r2"] is not None else float("nan")
        delta = round(oot_r2 - holdout_r2, 4) if oot["r2"] is not None else None

        print(
            f"{h}h{'':<6} "
            f"{holdout_r2:<12} "
            f"{oot_r2:<10} "
            f"{delta if delta is not None else 'n/a':<10} "
            f"{oot['n']:<8}"
        )
        oot_results.append({"horizon": h, "holdout_r2": holdout_r2, **oot, "delta": delta})

    print("=" * 70)
    print()
    print("Full OOT metrics:")
    print(pd.DataFrame(oot_results).to_string(index=False))


if __name__ == "__main__":
    main()