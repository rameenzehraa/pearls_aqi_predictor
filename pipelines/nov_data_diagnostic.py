"""Quick diagnostic: confirm Nov 2025 data exists in feature store + local parquet."""
import os
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

print("=" * 60)
print("LOCAL PARQUET CHECK")
print("=" * 60)
parquet_path = Path("data/backfill.parquet")
if parquet_path.exists():
    df_local = pd.read_parquet(parquet_path)
    df_local["timestamp"] = pd.to_datetime(df_local["timestamp"], utc=True)
    print(f"Rows: {len(df_local)}")
    print(f"Date range: {df_local['timestamp'].min()} → {df_local['timestamp'].max()}")
    nov_mask = (df_local["timestamp"] >= "2025-10-01") & (df_local["timestamp"] < "2025-12-01")
    print(f"Oct-Nov 2025 rows: {nov_mask.sum()}")
else:
    print(f"No local parquet at {parquet_path}")

print()
print("=" * 60)
print("HOPSWORKS FEATURE STORE CHECK")
print("=" * 60)
try:
    import hopsworks
    project = hopsworks.login(
        api_key_value=os.getenv("HOPSWORKS_API_KEY"),
        project=os.getenv("HOPSWORKS_PROJECT"),
    )
    fs = project.get_feature_store()
    fg = fs.get_feature_group("aqi_features", version=1)
    df_fs = fg.read()
    df_fs["timestamp"] = pd.to_datetime(df_fs["timestamp"], utc=True)
    df_fs = df_fs.sort_values("timestamp")
    print(f"Rows: {len(df_fs)}")
    print(f"Date range: {df_fs['timestamp'].min()} → {df_fs['timestamp'].max()}")
    nov_mask = (df_fs["timestamp"] >= "2025-10-01") & (df_fs["timestamp"] < "2025-12-01")
    print(f"Oct-Nov 2025 rows: {nov_mask.sum()}")
    print(f"Columns: {list(df_fs.columns)}")
except Exception as e:
    print(f"Hopsworks check failed: {e}")