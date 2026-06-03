# pipelines/check_latest_features.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from utils.hopsworks_client import get_feature_store
from utils.logger import get_logger

logger = get_logger(__name__)

CHAMPION_FEATURES = ["aqi", "month", "aqi_lag_24h", "aqi_lag_72h", "humidity"]


def main():
    fs = get_feature_store()
    fg = fs.get_feature_group("aqi_features", version=1)
    df = fg.read()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    
    last_24 = df.tail(24)
    logger.info("Latest 24 rows — NaN status per champion feature:")
    logger.info("-" * 70)
    
    for _, row in last_24.iterrows():
        ts = row["timestamp"]
        status = []
        for f in CHAMPION_FEATURES:
            mark = "NaN" if pd.isna(row[f]) else f"{row[f]:.2f}" if isinstance(row[f], (int, float)) else str(row[f])
            status.append(f"{f}={mark}")
        logger.info("%s | %s", ts, " | ".join(status))
    
    logger.info("-" * 70)
    logger.info("NaN counts per feature in last 24 rows:")
    for f in CHAMPION_FEATURES:
        n = last_24[f].isna().sum()
        logger.info("  %s: %d NaN", f, n)


if __name__ == "__main__":
    main()