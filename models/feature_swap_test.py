"""
Quick A/B test: original SHAP top-6 vs swapped (with day_of_week)
vs reduced (5 features). Three minutes of computation, real answer.
"""
from __future__ import annotations
import sys
from pathlib import Path

import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.hopsworks_client import get_feature_store
from utils.logger import get_logger

logger = get_logger(__name__)

HORIZONS = [24, 48, 72]
HOLDOUT_FRAC = 0.20
RIDGE_ALPHA = 10.0

FEATURE_SETS = {
    "A_original_top6": ["aqi", "month", "aqi_lag_24h", "aqi_lag_72h", "humidity", "pm2_5_lag_24h"],
    "B_swap_dow":      ["aqi", "month", "aqi_lag_24h", "aqi_lag_72h", "humidity", "day_of_week"],
    "C_top5_only":     ["aqi", "month", "aqi_lag_24h", "aqi_lag_72h", "humidity"],
}


def evaluate_set(df, target_col, feature_cols):
    work = df.dropna(subset=[target_col]).reset_index(drop=True)
    holdout_start = int(len(work) * (1 - HOLDOUT_FRAC))
    train = work.iloc[:holdout_start].copy()
    test  = work.iloc[holdout_start:].copy()

    medians = train[feature_cols].median()
    train[feature_cols] = train[feature_cols].fillna(medians)
    test[feature_cols]  = test[feature_cols].fillna(medians)

    X_train = train[feature_cols].values
    y_train = train[target_col].values
    X_test  = test[feature_cols].values
    y_test  = test[target_col].values

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    model = Ridge(alpha=RIDGE_ALPHA, random_state=42)
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)
    return {
        "r2":   round(r2_score(y_test, y_pred), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 2),
        "mae":  round(mean_absolute_error(y_test, y_pred), 2),
    }


def main():
    fs = get_feature_store()
    fg = fs.get_feature_group("aqi_features", version=1)
    df = fg.read()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    rows = []
    for set_name, feats in FEATURE_SETS.items():
        for h in HORIZONS:
            target = f"target_aqi_{h}h"
            m = evaluate_set(df, target, feats)
            rows.append({"set": set_name, "horizon": h, **m})

    results = pd.DataFrame(rows)
    pivot = results.pivot(index="set", columns="horizon", values="r2")
    pivot.columns = [f"R²_{h}h" for h in pivot.columns]
    print("\n" + "=" * 60)
    print("FEATURE SET COMPARISON (R² by horizon)")
    print("=" * 60)
    print(pivot.to_string())

    print("\nFull metrics:")
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()