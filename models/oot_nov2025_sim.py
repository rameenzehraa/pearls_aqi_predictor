"""
Day 17 — Historical noisy-window OOT simulation.

Trains a fresh Ridge champion on data ending 2025-10-31 23:00 UTC, then
predicts the Nov 1-15 2025 window — Karachi's seasonal AQI peak (high
variance) per EDA findings. Validates the model on the kind of conditions
it's actually meant to forecast well in.

Mirrors models/train_champion.py exactly except:
  - Reads from local parquet (data/backfill.parquet) for fast iteration
  - Date-based train/test split instead of chronological fraction
  - Includes Naive baseline comparison and saves predictions + plots

Usage:
    python -m models.oot_nov2025_sim
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.logger import get_logger

logger = get_logger(__name__)

# ── Constants (mirror train_champion.py) ─────────────────────────────────────
CHAMPION_FEATURES = [
    "aqi",
    "month",
    "aqi_lag_24h",
    "aqi_lag_72h",
    "humidity",
]
HORIZONS = [24, 48, 72]
RIDGE_ALPHA = 10.0

# ── Sim-specific ─────────────────────────────────────────────────────────────
PARQUET_PATH = Path("data/backfill.parquet")
TRAIN_END    = pd.Timestamp("2025-10-31 23:00", tz="UTC")
WINDOW_START = pd.Timestamp("2025-11-01 00:00", tz="UTC")
WINDOW_END   = pd.Timestamp("2025-11-16 00:00", tz="UTC")  # exclusive
ARTIFACTS_DIR = Path("artifacts/oot_nov2025")


def load_data() -> pd.DataFrame:
    logger.info("Loading local parquet: %s", PARQUET_PATH)
    df = pd.read_parquet(PARQUET_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    logger.info("Loaded %d rows (%s → %s)",
                len(df), df["timestamp"].min(), df["timestamp"].max())
    return df


def split_by_date(df: pd.DataFrame, target_col: str):
    """Train: timestamp <= TRAIN_END. Test: WINDOW_START <= timestamp < WINDOW_END."""
    work = df.dropna(subset=[target_col]).reset_index(drop=True)

    train = work[work["timestamp"] <= TRAIN_END].copy()
    test  = work[(work["timestamp"] >= WINDOW_START) &
                 (work["timestamp"] <  WINDOW_END)].copy()

    medians = train[CHAMPION_FEATURES].median()
    train[CHAMPION_FEATURES] = train[CHAMPION_FEATURES].fillna(medians)
    test[CHAMPION_FEATURES]  = test[CHAMPION_FEATURES].fillna(medians)
    return train, test, medians


def train_for_horizon(df: pd.DataFrame, horizon: int) -> dict:
    target_col = f"target_aqi_{horizon}h"
    logger.info("─" * 50)
    logger.info("Horizon %dh", horizon)

    train, test, medians = split_by_date(df, target_col)

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

    # Naive baseline: predict y at t+horizon as aqi at t
    y_naive = test["aqi"].values

    ridge_metrics = {
        "r2":   round(float(r2_score(y_test, y_pred)), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 2),
        "mae":  round(float(mean_absolute_error(y_test, y_pred)), 2),
    }
    naive_metrics = {
        "r2":   round(float(r2_score(y_test, y_naive)), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(y_test, y_naive))), 2),
        "mae":  round(float(mean_absolute_error(y_test, y_naive)), 2),
    }

    # Save model + scaler + medians
    horizon_dir = ARTIFACTS_DIR / f"h{horizon}"
    horizon_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, horizon_dir / "ridge_model.joblib")
    joblib.dump(scaler, horizon_dir / "scaler.joblib")
    medians.to_json(horizon_dir / "feature_medians.json")

    # Save predictions CSV
    pred_df = pd.DataFrame({
        "timestamp": test["timestamp"].values,
        "actual":    y_test,
        "ridge":     y_pred.round(2),
        "naive":     y_naive,
    })
    pred_df.to_csv(horizon_dir / "predictions.csv", index=False)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(pred_df["timestamp"], pred_df["actual"], "k-", label="Actual",  lw=2)
    ax.plot(pred_df["timestamp"], pred_df["ridge"],  "b--", label="Ridge",  lw=1.5)
    ax.plot(pred_df["timestamp"], pred_df["naive"],  "r:",  label="Naive",  lw=1.5)
    ax.set_title(f"Nov 2025 OOT — {horizon}h horizon  "
                 f"(Ridge R²={ridge_metrics['r2']}, Naive R²={naive_metrics['r2']})")
    ax.set_xlabel("Timestamp (UTC)")
    ax.set_ylabel("AQI")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(horizon_dir / f"plot_h{horizon}.png", dpi=120)
    plt.close(fig)

    logger.info(
        "  Ridge R²=%.4f RMSE=%.2f MAE=%.2f  |  Naive R²=%.4f RMSE=%.2f MAE=%.2f",
        ridge_metrics["r2"], ridge_metrics["rmse"], ridge_metrics["mae"],
        naive_metrics["r2"], naive_metrics["rmse"], naive_metrics["mae"],
    )

    return {
        "horizon": horizon,
        "n_train": len(train),
        "n_test": len(test),
        "ridge": ridge_metrics,
        "naive": naive_metrics,
        "ridge_coefficients": dict(zip(CHAMPION_FEATURES,
                                       model.coef_.round(4).tolist())),
        "ridge_intercept": round(float(model.intercept_), 4),
    }


def main() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("=" * 60)
    logger.info("Nov 2025 noisy-window OOT simulation")
    logger.info("Train end: %s  |  Window: %s → %s",
                TRAIN_END, WINDOW_START, WINDOW_END)
    logger.info("=" * 60)

    df = load_data()

    # Window stats for honest framing
    window_df = df[(df["timestamp"] >= WINDOW_START) &
                   (df["timestamp"] <  WINDOW_END)]
    window_stats = {
        "n_rows":   int(len(window_df)),
        "aqi_mean": round(float(window_df["aqi"].mean()), 2),
        "aqi_std":  round(float(window_df["aqi"].std()), 2),
        "aqi_min":  round(float(window_df["aqi"].min()), 2),
        "aqi_max":  round(float(window_df["aqi"].max()), 2),
    }
    logger.info("Window AQI stats: %s", window_stats)

    summary = {
        "ran_at_utc": datetime.now(timezone.utc).isoformat(),
        "train_end_utc":    TRAIN_END.isoformat(),
        "window_start_utc": WINDOW_START.isoformat(),
        "window_end_utc":   WINDOW_END.isoformat(),
        "features": CHAMPION_FEATURES,
        "ridge_alpha": RIDGE_ALPHA,
        "window_stats": window_stats,
        "horizons": [],
    }

    for h in HORIZONS:
        summary["horizons"].append(train_for_horizon(df, h))

    metrics_path = ARTIFACTS_DIR / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Saved %s", metrics_path)

    # Final table
    print("\n" + "=" * 70)
    print("NOV 2025 NOISY-WINDOW OOT — RESULTS")
    print("=" * 70)
    print(f"Window: {WINDOW_START.date()} → {WINDOW_END.date()}  "
          f"(n={window_stats['n_rows']}, "
          f"AQI mean={window_stats['aqi_mean']}, std={window_stats['aqi_std']})")
    print()
    print(f"{'Horizon':<8} {'Ridge R²':<10} {'Naive R²':<10} "
          f"{'Ridge RMSE':<12} {'Naive RMSE':<12} {'n_test':<8}")
    print("-" * 70)
    for h_data in summary["horizons"]:
        r, n = h_data["ridge"], h_data["naive"]
        print(f"{h_data['horizon']}h{'':<6} "
              f"{r['r2']:<10} {n['r2']:<10} "
              f"{r['rmse']:<12} {n['rmse']:<12} {h_data['n_test']:<8}")


if __name__ == "__main__":
    main()