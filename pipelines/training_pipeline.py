"""
Training pipeline — daily model retraining.

Runs daily at 02:00 UTC via GitHub Actions. Pulls fresh features from
Hopsworks, retrains the production champion (3 Ridge models, one per
horizon), then recalibrates the CQR conformal interval system on the
fresh data + new champion.

Hard rule (Draft 6, MOM): training pipeline reads from Hopsworks only.
No local CSV/parquet caching in this code path.

Usage:
    python -m pipelines.training_pipeline
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.hopsworks_client import get_feature_store
from utils.logger import get_logger

# Import the actual training entry points.
from models import train_champion, conformal_intervals, register_to_registry

logger = get_logger(__name__)

FEATURE_GROUP_NAME = "aqi_features"
FEATURE_GROUP_VERSION = 1
MIN_ROWS_FOR_TRAINING = 500


def preflight_check() -> None:
    """
    Verify Hopsworks is reachable and we have enough data with targets
    before kicking off the (slow) training step. Fail fast if not.
    """
    logger.info("Preflight: checking feature store access and data sufficiency")

    fs = get_feature_store()
    fg = fs.get_feature_group(FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)

    try:
        df = fg.read()
    except Exception as exc:
        msg = str(exc)
        if any(s in msg for s in ("Flight", "Socket closed", "Query Service")):
            logger.warning(
                "Arrow Flight read failed (%s). Falling back to JDBC/Hive.",
                type(exc).__name__,
            )
            df = fg.read(read_options={"use_hive": True})
        else:
            raise

    n_total = len(df)
    n_with_target = int(df["has_target"].sum()) if "has_target" in df.columns else 0
    logger.info("Feature group rows:    %d", n_total)
    logger.info("Rows with all targets: %d", n_with_target)

    if n_with_target < MIN_ROWS_FOR_TRAINING:
        raise RuntimeError(
            f"Not enough rows with targets ({n_with_target} < {MIN_ROWS_FOR_TRAINING}). "
            f"Aborting before training."
        )

    logger.info("Preflight OK — proceeding to training.")


def main() -> None:
    logger.info("=" * 60)
    logger.info("Daily training pipeline starting")
    logger.info("=" * 60)

    try:
        # 1) Sanity check
        preflight_check()

        # 2) Retrain the 3 Ridge champions (one per horizon)
        logger.info("=" * 60)
        logger.info("Step 1/3: retraining Ridge champions")
        logger.info("=" * 60)
        train_champion.main()

        # 3) Recalibrate CQR intervals on the fresh champion
        logger.info("=" * 60)
        logger.info("Step 2/3: recalibrating CQR conformal intervals")
        logger.info("=" * 60)
        conformal_intervals.main()

        # 4) Push fresh artifacts to Hopsworks Model Registry (auto-versioned)
        logger.info("=" * 60)
        logger.info("Step 3/3: registering models to Hopsworks Model Registry")
        logger.info("=" * 60)
        register_to_registry.main()

        logger.info("=" * 60)
        logger.info("Daily training pipeline complete.")
        logger.info("=" * 60)

    except Exception as exc:
        logger.exception("Training pipeline failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()