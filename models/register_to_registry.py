"""
Day 11 Task 5 — Register champion + CQR models to Hopsworks Model Registry.

Registers six models, version 1 each:
  karachi_aqi_ridge_{24,48,72}h  — production champion (point prediction)
  karachi_aqi_cqr_{24,48,72}h    — conformal interval system

Each model is registered with:
  - All required artifact files
  - Per-horizon metrics (R², RMSE, MAE, coverage where applicable)
  - Feature schema and model metadata

Usage:
    python -m models.register_to_registry
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from utils.hopsworks_client import get_model_registry

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.logger import get_logger

logger = get_logger(__name__)

CHAMPION_DIR = Path("models/champion")
CQR_DIR = Path("models/cqr")
HORIZONS = [24, 48, 72]
CHAMPION_FEATURES = ["aqi", "month", "aqi_lag_24h", "aqi_lag_72h", "humidity"]


def register_champion(mr, horizon: int, champion_meta: dict) -> None:
    """Register a single horizon's production Ridge champion."""
    name = f"karachi_aqi_ridge_{horizon}h"
    horizon_dir = CHAMPION_DIR / f"h{horizon}"

    # Pull the metrics for this horizon from the metadata
    h_data = next(h for h in champion_meta["horizons"] if h["horizon"] == horizon)
    metrics = h_data["metrics"]

    description = (
        f"Ridge regression champion for Karachi AQI prediction at {horizon}h horizon. "
        f"Trained on {champion_meta['n_rows_loaded']} hourly rows from Hopsworks "
        f"feature group aqi_features v1. Features: {', '.join(CHAMPION_FEATURES)}. "
        f"Selected via SHAP analysis + ablation study (Day 11)."
    )

    logger.info("Registering %s …", name)
    model = mr.python.create_model(
        name=name,
        version=1,
        metrics=metrics,
        description=description,
        feature_view=None,  # we use raw feature group
        input_example=None,
    )
    model.save(str(horizon_dir), keep_original_files=True)
    logger.info("  ✓ Registered %s v%d  (R²=%.4f)", name, model.version, metrics["r2"])


def register_cqr(mr, horizon: int) -> None:
    """Register a single horizon's CQR (interval) system."""
    name = f"karachi_aqi_cqr_{horizon}h"
    horizon_dir = CQR_DIR / f"h{horizon}"

    with open(horizon_dir / "calibration.json") as f:
        cal = json.load(f)

    metrics = {
        "coverage": cal["holdout_coverage"],
        "avg_width": cal["holdout_avg_width"],
        "Q_widen": cal["Q_widen"],
        "point_r2": cal.get("production_point_r2", cal.get("ridge_cqr_holdout_r2", 0)),
    }

    description = (
        f"Conformalized Quantile Regression (Romano et al. 2019, hybrid variant) "
        f"for Karachi AQI prediction intervals at {horizon}h horizon. "
        f"Nominal 80% coverage, achieved {cal['holdout_coverage']*100:.1f}% on holdout. "
        f"Point predictions from production champion (karachi_aqi_ridge_{horizon}h v1); "
        f"intervals from QR + calibration. "
        f"Train/calibration split: {cal['n_train_proper']}/{cal['n_calibration']}."
    )

    logger.info("Registering %s …", name)
    model = mr.python.create_model(
        name=name,
        version=1,
        metrics=metrics,
        description=description,
        feature_view=None,
        input_example=None,
    )
    model.save(str(horizon_dir), keep_original_files=True)
    logger.info(
        "  ✓ Registered %s v%d  (coverage=%.1f%%)",
        name, model.version, metrics["coverage"] * 100
    )


def main() -> None:
    logger.info("=" * 60)
    logger.info("Day 11 Task 5 — Register models to Hopsworks Model Registry")
    logger.info("=" * 60)

    mr = get_model_registry()
    logger.info("Connected to model registry: %s", mr.project_name)

    # Load champion metadata once
    with open(CHAMPION_DIR / "champion_metadata.json") as f:
        champion_meta = json.load(f)

    # Register all 6 models
    for h in HORIZONS:
        register_champion(mr, h, champion_meta)

    for h in HORIZONS:
        register_cqr(mr, h)

    print("\n" + "=" * 60)
    print("REGISTRATION COMPLETE — 6 models registered, v1 each")
    print("=" * 60)
    print("Champion (point predictions):")
    for h in HORIZONS:
        print(f"  karachi_aqi_ridge_{h}h v1")
    print("CQR (interval predictions):")
    for h in HORIZONS:
        print(f"  karachi_aqi_cqr_{h}h v1")
    print()
    print(f"View at: https://eu-west.cloud.hopsworks.ai:443/p/32001")


if __name__ == "__main__":
    main()