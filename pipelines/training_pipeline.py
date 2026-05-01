"""
Training pipeline — daily model retraining.

DAY 7 STUB: this file currently only verifies that:
    1. Hopsworks credentials are present and valid
    2. The aqi_features feature group can be read
    3. We have enough rows (with non-null targets) to train

The actual model training logic lands on Day 9. The stub exists so the
Day 8 GitHub Actions training workflow has something callable.

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

logger = get_logger(__name__)

FEATURE_GROUP_NAME = "aqi_features"
FEATURE_GROUP_VERSION = 1
MIN_ROWS_FOR_TRAINING = 500   # arbitrary floor; real check on Day 9


def main() -> None:
    logger.info("Training pipeline (stub) — verifying feature store access")

    try:
        fs = get_feature_store()
        fg = fs.get_feature_group(FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)
        df = fg.read()
        n_total = len(df)
        n_with_target = int(df["has_target"].sum()) if "has_target" in df.columns else 0

        logger.info("Feature group rows:    %d", n_total)
        logger.info("Rows with all targets: %d", n_with_target)

        if n_with_target < MIN_ROWS_FOR_TRAINING:
            logger.error(
                "Not enough rows with targets (%d < %d). Aborting.",
                n_with_target, MIN_ROWS_FOR_TRAINING,
            )
            sys.exit(1)

        logger.info("Stub OK — feature store reachable, sufficient training data.")
        logger.info("Real model training to be implemented on Day 9.")
    except Exception as exc:
        logger.exception("Training pipeline failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()