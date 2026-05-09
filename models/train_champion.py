"""
Day 11 Task 2 (final) — Train the production Ridge champion.

Uses the 5 features identified by SHAP analysis + ablation study:
    aqi, month, aqi_lag_24h, aqi_lag_72h, humidity

Trains one Ridge model per horizon (24h, 48h, 72h), saves models
and scalers as joblib artifacts, and writes a metadata JSON describing
the training run.

Usage:
    python -m models.train_champion
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.hopsworks_client import get_feature_store
from utils.logger import get_logger

logger = get_logger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────
FEATURE_GROUP_NAME = "aqi_features"
FEATURE_GROUP_VERSION = 1

CHAMPION_FEATURES = [
    "aqi",
    "month",
    "aqi_lag_24h",
    "aqi_lag_72h",
    "humidity",
]

HORIZONS = [24, 48, 72]
HOLDOUT_FRAC = 0.20
RIDGE_ALPHA = 10.0

ARTIFACTS_DIR = Path("models/champion")


def load_data() -> pd.DataFrame:
    logger.info("Connecting to Hopsworks …")
    fs = get_feature_store()
    fg = fs.get_feature_group(FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)
    df = fg.read()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    logger.info("Loaded %d rows from Hopsworks", len(df))
    return df


def split_and_impute(df: pd.DataFrame, target_col: str):
    """Same chronological split as Day 9. Median imputation fitted on train."""
    work = df.dropna(subset=[target_col]).reset_index(drop=True)
    holdout_start = int(len(work) * (1 - HOLDOUT_FRAC))
    train = work.iloc[:holdout_start].copy()
    test  = work.iloc[holdout_start:].copy()

    medians = train[CHAMPION_FEATURES].median()
    train[CHAMPION_FEATURES] = train[CHAMPION_FEATURES].fillna(medians)
    test[CHAMPION_FEATURES]  = test[CHAMPION_FEATURES].fillna(medians)
    return train, test, medians


def train_for_horizon(df: pd.DataFrame, horizon: int) -> dict:
    """Train Ridge on a single horizon, save artifacts, return metrics."""
    target_col = f"target_aqi_{horizon}h"
    logger.info("─" * 50)
    logger.info("Horizon %dh", horizon)

    train, test, medians = split_and_impute(df, target_col)

    X_train = train[CHAMPION_FEATURES].values
    y_train = train[target_col].values
    X_test  = test[CHAMPION_FEATURES].values
    y_test  = test[target_col].values

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    model = Ridge(alpha=RIDGE_ALPHA, random_state=42)
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)

    metrics = {
        "r2":   round(float(r2_score(y_test, y_pred)), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 2),
        "mae":  round(float(mean_absolute_error(y_test, y_pred)), 2),
    }

    # Save artifacts
    horizon_dir = ARTIFACTS_DIR / f"h{horizon}"
    horizon_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, horizon_dir / "ridge_model.joblib")
    joblib.dump(scaler, horizon_dir / "scaler.joblib")
    medians.to_json(horizon_dir / "feature_medians.json")

    logger.info(
        "  R²=%.4f  RMSE=%.2f  MAE=%.2f  | saved %s",
        metrics["r2"], metrics["rmse"], metrics["mae"], horizon_dir
    )
    return {
        "horizon": horizon,
        "metrics": metrics,
        "n_train": len(train),
        "n_test": len(test),
        "coefficients": dict(zip(CHAMPION_FEATURES, model.coef_.round(4).tolist())),
        "intercept": round(float(model.intercept_), 4),
    }


def main() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("=" * 60)
    logger.info("Day 11 — Training Ridge champion (5 features)")
    logger.info("=" * 60)

    df = load_data()

    summary = {
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_rows_loaded": len(df),
        "features": CHAMPION_FEATURES,
        "ridge_alpha": RIDGE_ALPHA,
        "holdout_frac": HOLDOUT_FRAC,
        "horizons": [],
    }

    for h in HORIZONS:
        result = train_for_horizon(df, h)
        summary["horizons"].append(result)

    metadata_path = ARTIFACTS_DIR / "champion_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Saved %s", metadata_path)

    # Also copy metadata into each per-horizon directory so the Hopsworks
    # Model Registry artifact includes it (registry uploads per-horizon
    # dirs, not the parent). Lets the backend display training metrics
    # without needing a separate metadata-fetching step.
    for h in HORIZONS:
        horizon_dir = ARTIFACTS_DIR / f"h{h}"
        if horizon_dir.exists():
            with open(horizon_dir / "champion_metadata.json", "w") as f:
                json.dump(summary, f, indent=2)

    # Print final summary
    print("\n" + "=" * 60)
    print("CHAMPION TRAINING COMPLETE")
    print("=" * 60)
    print(f"Features: {CHAMPION_FEATURES}")
    print(f"Rows used: {summary['n_rows_loaded']}")
    print(f"Ridge alpha: {RIDGE_ALPHA}")
    print()
    print(f"{'Horizon':<10} {'R²':<8} {'RMSE':<8} {'MAE':<8}")
    print("-" * 40)
    for h_data in summary["horizons"]:
        m = h_data["metrics"]
        print(f"{h_data['horizon']}h{'':<8} {m['r2']:<8} {m['rmse']:<8} {m['mae']:<8}")


if __name__ == "__main__":
    main()