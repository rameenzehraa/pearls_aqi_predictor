"""
Day 11 Task 2 — Feature ablation study on Ridge.

Ranks features by mean |SHAP| (averaged across horizons), then trains
Ridge with progressively fewer features. Records R² at each step
and saves a plot showing how performance degrades.

Usage:
    python -m models.feature_ablation
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
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
RIDGE_ALPHA = 10.0
ARTIFACTS_DIR = Path("artifacts/ablation")
SHAP_RANKINGS_PATH = Path("artifacts/shap/shap_rankings.csv")
MIN_FEATURES = 3   # don't go below this


# ── Reuse helpers from shap_analysis ─────────────────────────────────────────
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


def split_and_impute(df: pd.DataFrame, target_col: str, feature_cols: list[str]) -> tuple:
    work = df.dropna(subset=[target_col]).reset_index(drop=True)
    holdout_start = int(len(work) * (1 - HOLDOUT_FRAC))
    train = work.iloc[:holdout_start].copy()
    test  = work.iloc[holdout_start:].copy()
    medians = train[feature_cols].median()
    train[feature_cols] = train[feature_cols].fillna(medians)
    test[feature_cols]  = test[feature_cols].fillna(medians)
    return train, test


def fit_and_score(train, test, feature_cols: list[str], target_col: str) -> float:
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
    return r2_score(y_test, y_pred)


# ── Build cross-horizon ranking from SHAP CSV ────────────────────────────────
def get_global_ranking() -> list[str]:
    """
    Average mean_abs_shap across horizons, return features ranked
    most-important to least-important.
    """
    df = pd.read_csv(SHAP_RANKINGS_PATH)
    avg = df.groupby("feature")["mean_abs_shap"].mean().sort_values(ascending=False)
    ranking = avg.index.tolist()
    logger.info("Global SHAP ranking (top 5): %s", ranking[:5])
    logger.info("Global SHAP ranking (bottom 5): %s", ranking[-5:])
    return ranking


# ── Ablation loop ────────────────────────────────────────────────────────────
def run_ablation(df: pd.DataFrame, ranking: list[str]) -> pd.DataFrame:
    """
    For each horizon and for each feature count from N down to MIN_FEATURES,
    fit Ridge using the top-K features by global SHAP rank.
    """
    rows = []
    for horizon in HORIZONS:
        target_col = f"target_aqi_{horizon}h"
        logger.info("─" * 50)
        logger.info("Horizon %dh", horizon)

        for k in range(len(ranking), MIN_FEATURES - 1, -1):
            kept_features = ranking[:k]
            train, test = split_and_impute(df, target_col, kept_features)
            r2 = fit_and_score(train, test, kept_features, target_col)
            rows.append({
                "horizon": horizon,
                "n_features": k,
                "r2": r2,
                "dropped": ", ".join(ranking[k:]) if k < len(ranking) else "",
            })
            logger.info("  k=%2d → R²=%.4f", k, r2)

    return pd.DataFrame(rows)


# ── Plot ─────────────────────────────────────────────────────────────────────
def plot_curves(results: pd.DataFrame, save_path: Path) -> None:
    plt.figure(figsize=(10, 6))
    for h in HORIZONS:
        sub = results[results["horizon"] == h].sort_values("n_features")
        plt.plot(sub["n_features"], sub["r2"], marker="o", label=f"{h}h")
    plt.xlabel("Number of features (top-K by mean |SHAP|)")
    plt.ylabel("R² on holdout")
    plt.title("Feature ablation — Ridge")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.gca().invert_xaxis()  # so we read left-to-right as features-being-dropped
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    logger.info("Saved %s", save_path)


def main() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("=" * 60)
    logger.info("Day 11 Task 2 — Feature ablation study")
    logger.info("=" * 60)

    df = load_data()
    feature_cols = get_feature_cols(df)
    logger.info("Total feature count: %d", len(feature_cols))

    ranking = get_global_ranking()
    if set(ranking) != set(feature_cols):
        missing = set(feature_cols) - set(ranking)
        extra = set(ranking) - set(feature_cols)
        raise ValueError(
            f"SHAP ranking mismatch. Missing: {missing}. Extra: {extra}. "
            f"Re-run shap_analysis.py first."
        )

    results = run_ablation(df, ranking)
    csv_path = ARTIFACTS_DIR / "ablation_results.csv"
    results.to_csv(csv_path, index=False)
    logger.info("Saved %s", csv_path)

    plot_curves(results, ARTIFACTS_DIR / "ablation_curves.png")

    # Print pivoted view for the console
    pivot = results.pivot(index="n_features", columns="horizon", values="r2").round(4)
    pivot.columns = [f"R²_{h}h" for h in pivot.columns]
    print("\n" + "=" * 60)
    print("ABLATION RESULTS (R² by feature count, all horizons)")
    print("=" * 60)
    print(pivot.sort_index(ascending=False).to_string())


if __name__ == "__main__":
    main()