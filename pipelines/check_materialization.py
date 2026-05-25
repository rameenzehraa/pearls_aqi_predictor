"""
One-off diagnostic: check the state of the offline feature group
materialization job. Read-only — does not modify anything.

Usage:
    python -m pipelines.check_materialization
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.hopsworks_client import get_feature_store
from utils.logger import get_logger

logger = get_logger(__name__)

FEATURE_GROUP_NAME = "aqi_features"
FEATURE_GROUP_VERSION = 1


def main() -> None:
    logger.info("Connecting to Hopsworks...")
    fs = get_feature_store()
    fg = fs.get_feature_group(FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)

    job = fg.materialization_job
    logger.info("Materialization job name: %s", job.name)

    # Current state of the most recent execution
    try:
        state = job.get_state()
        logger.info("Current state: %s", state)
    except Exception as exc:
        logger.warning("get_state() failed: %s", exc)

    # Final state of the last finished execution (if any)
    try:
        final_state = job.get_final_state()
        logger.info("Last final state: %s", final_state)
    except Exception as exc:
        logger.warning("get_final_state() failed: %s", exc)

    # List recent executions so we can see history
    try:
        executions = job.get_executions()
        logger.info("Found %d executions total", len(executions))
        # Show the 5 most recent
        for ex in executions[:5]:
            logger.info(
                "  exec id=%s state=%s final_state=%s submitted=%s",
                getattr(ex, "id", "?"),
                getattr(ex, "state", "?"),
                getattr(ex, "final_status", getattr(ex, "finalStatus", "?")),
                getattr(ex, "submission_time", "?"),
            )
    except Exception as exc:
        logger.warning("get_executions() failed: %s", exc)


if __name__ == "__main__":
    main()