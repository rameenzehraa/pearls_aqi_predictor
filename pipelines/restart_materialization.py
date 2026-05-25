"""
One-off fix: kill the stuck SUBMITTED materialization execution
and trigger a fresh run.

Background: 2026-05-21 00:15 UTC execution has been queued for
2.5+ days and is blocking all subsequent inserts from materializing
to the offline store. This is a Hopsworks platform issue, not a
pipeline defect.

Usage:
    python -m pipelines.restart_materialization
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.hopsworks_client import get_feature_store
from utils.logger import get_logger

logger = get_logger(__name__)

FEATURE_GROUP_NAME = "aqi_features"
FEATURE_GROUP_VERSION = 1

# Execution we expect to kill — sanity check before stopping
STUCK_EXEC_ID = 163936


def main() -> None:
    logger.info("Connecting to Hopsworks...")
    fs = get_feature_store()
    fg = fs.get_feature_group(FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)
    job = fg.materialization_job

    # Step 1: list current executions and find the stuck one
    executions = job.get_executions()
    if not executions:
        logger.error("No executions found — something is off, aborting.")
        return

    stuck = executions[0]
    logger.info(
        "Most recent execution: id=%s state=%s submitted=%s",
        getattr(stuck, "id", "?"),
        getattr(stuck, "state", "?"),
        getattr(stuck, "submission_time", "?"),
    )

    if getattr(stuck, "id", None) != STUCK_EXEC_ID:
        logger.warning(
            "Top execution id (%s) doesn't match expected stuck id (%s). "
            "Aborting to be safe — re-run check_materialization first.",
            getattr(stuck, "id", "?"), STUCK_EXEC_ID,
        )
        return

    if str(getattr(stuck, "state", "")) != "SUBMITTED":
        logger.warning(
            "Top execution state is %s, not SUBMITTED. Aborting — "
            "something may have changed since diagnosis.",
            getattr(stuck, "state", "?"),
        )
        return

    # Step 2: stop the stuck execution
    logger.info("Stopping stuck execution %s ...", stuck.id)
    try:
        stuck.stop()
        logger.info("Stop request sent.")
    except Exception as exc:
        logger.exception("stop() failed: %s", exc)
        return

    # Step 3: wait a few seconds for Hopsworks to register the stop
    logger.info("Waiting 10s for state to update...")
    time.sleep(10)

    # Step 4: confirm it's no longer in SUBMITTED
    executions = job.get_executions()
    top = executions[0]
    logger.info(
        "After stop: id=%s state=%s final_state=%s",
        getattr(top, "id", "?"),
        getattr(top, "state", "?"),
        getattr(top, "final_status", getattr(top, "finalStatus", "?")),
    )

    # Step 5: trigger a fresh materialization run
    logger.info("Triggering fresh materialization job...")
    try:
        new_exec = job.run()
        logger.info("New execution started: id=%s state=%s",
                    getattr(new_exec, "id", "?"),
                    getattr(new_exec, "state", "?"))
    except Exception as exc:
        logger.exception("job.run() failed: %s", exc)
        return

    logger.info("Done. Re-run check_materialization in a few minutes to see progress.")


if __name__ == "__main__":
    main()