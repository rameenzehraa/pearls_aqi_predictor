"""
Day 11 — SHAP analysis on Ridge champion.

Computes Shapley values using shap.LinearExplainer (exact closed-form
Shapley values for linear models). Generates summary and bar plots
per horizon and saves them as PNGs for the report.

Usage:
    python -m models.shap_analysis
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.hopsworks_client import get_feature_store
from utils.logger import get_logger

logger = get_logger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────
FEATURE_GROUP_NAME = "aqi_features"
FEATURE_GROUP_VERSION = 1
TARGET_COLS = ["target_aqi_24h", "target_aqi_48h", "target_aqi_72h"]
HORIZONS = [24, 48, 72]
HOLDOUT_FRAC = 0.20
RIDGE_ALPHA = 10.0          # match Day 9
ARTIFACTS_DIR = Path("artifacts/shap")


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    exclude = {"timestamp", "has_target"} | set(TARGET_COLS)
    return [c for c in df.columns if c not in exclude]


def load_data() -> pd.DataFrame:
    logger.info("Connecting to Hopsworks …")
    fs = get_feature_store()
    fg = fs.get_feature_group(FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)
    df = fg.read()

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)

    logger.info("Loaded %d rows from Hopsworks", len(df))
    return df


def chronological_split(df: pd.DataFrame, target_col: str) -> tuple:
    """Same split logic as Day 9. Drops rows where target is NaN."""
    work = df.dropna(subset=[target_col]).reset_index(drop=True)
    n = len(work)
    holdout_start = int(n * (1 - HOLDOUT_FRAC))

    train = work.iloc[:holdout_start].copy()
    test  = work.iloc[holdout_start:].copy()
    return train, test


def impute_features(
    train: pd.DataFrame, test: pd.DataFrame, feature_cols: list[str]
) -> tuple:
    """Median imputation matching Day 9. Fit on train, apply to both."""
    feature_medians = train[feature_cols].median()
    train[feature_cols] = train[feature_cols].fillna(feature_medians)
    test[feature_cols]  = test[feature_cols].fillna(feature_medians)
    return train, test


def run_shap_for_horizon(df: pd.DataFrame, horizon: int, feature_cols: list[str]) -> dict:
    """Train Ridge on horizon, compute SHAP on holdout, save plots."""
    target_col = f"target_aqi_{horizon}h"
    logger.info("─" * 50)
    logger.info("Horizon %dh — training Ridge and computing SHAP", horizon)

    train, test = chronological_split(df, target_col)
    train, test = impute_features(train, test, feature_cols)

    X_train = train[feature_cols].values
    y_train = train[target_col].values
    X_test  = test[feature_cols].values

    # Scale (Ridge prefers scaled features for fair coefficient comparison)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    model = Ridge(alpha=RIDGE_ALPHA, random_state=42)
    model.fit(X_train_s, y_train)

    # SHAP
    explainer = shap.LinearExplainer(model, X_train_s)
    shap_values = explainer.shap_values(X_test_s)
    logger.info("SHAP values shape: %s", shap_values.shape)

    # Convert to a DataFrame for plotting with feature names
    X_test_df = pd.DataFrame(X_test_s, columns=feature_cols)

    # ── Summary plot (dot plot) ──
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test_df, show=False)
    plt.title(f"SHAP Summary — Ridge {horizon}h")
    plt.tight_layout()
    summary_path = ARTIFACTS_DIR / f"shap_summary_{horizon}h.png"
    plt.savefig(summary_path, dpi=120, bbox_inches="tight")
    plt.close()
    logger.info("Saved %s", summary_path)

    # ── Bar plot (mean |SHAP|) ──
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test_df, plot_type="bar", show=False)
    plt.title(f"SHAP Importance — Ridge {horizon}h")
    plt.tight_layout()
    bar_path = ARTIFACTS_DIR / f"shap_bar_{horizon}h.png"
    plt.savefig(bar_path, dpi=120, bbox_inches="tight")
    plt.close()
    logger.info("Saved %s", bar_path)

    # ── Ranked importance table ──
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    ranking = pd.DataFrame({
        "feature":      feature_cols,
        "mean_abs_shap": mean_abs_shap,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    return {"horizon": horizon, "ranking": ranking}


def main() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("=" * 60)
    logger.info("Day 11 — SHAP analysis on Ridge champion")
    logger.info("=" * 60)

    df = load_data()
    feature_cols = get_feature_cols(df)
    logger.info("Feature count: %d", len(feature_cols))

    rankings = {}
    for h in HORIZONS:
        result = run_shap_for_horizon(df, h, feature_cols)
        rankings[h] = result["ranking"]

    # Save all rankings to one CSV
    combined = []
    for h, r in rankings.items():
        r = r.copy()
        r.insert(0, "horizon", h)
        combined.append(r)
    full = pd.concat(combined, ignore_index=True)
    csv_path = ARTIFACTS_DIR / "shap_rankings.csv"
    full.to_csv(csv_path, index=False)
    logger.info("Saved %s", csv_path)

    # Print top 10 per horizon for the console report
    print("\n" + "=" * 60)
    print("TOP 10 FEATURES BY MEAN |SHAP| — per horizon")
    print("=" * 60)
    for h in HORIZONS:
        print(f"\nHorizon {h}h:")
        print(rankings[h].head(10).to_string(index=False))


if __name__ == "__main__":
    main()