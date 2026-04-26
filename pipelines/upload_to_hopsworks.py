"""
One-time script: upload the local backfill parquet to Hopsworks.

Reads `data/backfill.parquet` and writes it into the `aqi_features`
feature group in Hopsworks. Run this once after the local backfill
to seed the feature store.

After this is verified, `backfill.py` will be modified to write to
Hopsworks directly. This script remains as a recovery tool.

Usage:
    python -m pipelines.upload_to_hopsworks
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.hopsworks_client import get_feature_store
from utils.logger import get_logger

logger = get_logger(__name__)

PARQUET_PATH = Path(__file__).resolve().parent.parent / "data" / "backfill.parquet"

FEATURE_GROUP_NAME = "aqi_features"
FEATURE_GROUP_VERSION = 1
PRIMARY_KEY = ["timestamp"]
EVENT_TIME_COLUMN = "timestamp"
DESCRIPTION = (
    "Hourly Karachi AQI features: pollutants, weather, time-of-day, "
    "rolling stats, lagged values, and 24/48/72h AQI targets."
)


def main() -> None:
    if not PARQUET_PATH.exists():
        logger.error("Parquet file not found at %s. Run backfill.py first.", PARQUET_PATH)
        sys.exit(1)

    logger.info("Reading local backfill: %s", PARQUET_PATH)
    df = pd.read_parquet(PARQUET_PATH)
    logger.info("Loaded %d rows, %d columns", len(df), len(df.columns))

    logger.info("Connecting to Hopsworks feature store")
    fs = get_feature_store()

    logger.info(
        "Getting or creating feature group: %s (v%d)",
        FEATURE_GROUP_NAME, FEATURE_GROUP_VERSION,
    )
    fg = fs.get_or_create_feature_group(
        name=FEATURE_GROUP_NAME,
        version=FEATURE_GROUP_VERSION,
        primary_key=PRIMARY_KEY,
        event_time=EVENT_TIME_COLUMN,
        description=DESCRIPTION,
        online_enabled=False,    # offline is enough for our needs
    )

    logger.info("Inserting %d rows into Hopsworks", len(df))
    fg.insert(df, write_options={"wait_for_job": True})
    logger.info("Insert complete.")

    logger.info("Verifying read-back...")
    read_df = fg.read()
    logger.info("Feature group now contains %d rows", len(read_df))

    if len(read_df) == 0:
        logger.warning("Read returned 0 rows — write may still be processing.")
    else:
        logger.info(
            "Date range in feature store: %s → %s",
            read_df["timestamp"].min(), read_df["timestamp"].max(),
        )


if __name__ == "__main__":
    main()