"""
Day 9 — train and evaluate classical + XGBoost models for AQI forecasting.

Models trained per horizon (24h, 48h, 72h):
    - Naive baseline   (predict tomorrow = today's AQI)
    - Ridge regression (with StandardScaler + alpha grid search)
    - Random Forest    (small grid: n_estimators, max_depth)
    - XGBoost          (RandomizedSearchCV, 20 trials)
    - XGBoost p10      (quantile, no HPT)
    - XGBoost p90      (quantile, no HPT)

Validation:
    - Chronological 80/20 split
    - 5-fold walk-forward CV on the 80% training pool for HPT
    - Metrics reported on the 20% holdout (touched once at the end)

Hard rule (Draft 6, MOM): training reads from Hopsworks only.

Models are NOT pushed to the registry today. That's Day 11 after SHAP
feature selection.

Usage:
    python -m models.train_classical
    python -m models.train_classical --quick   (smaller grids for testing)
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.config import (
    CV_N_SPLITS,
    HPT_N_ITER,
    HPT_CV_FOLDS,
    TRAIN_TEST_SPLIT,
    XGBOOST_QUANTILE_LOW,
    XGBOOST_QUANTILE_HIGH,
)
from utils.cv import chronological_holdout_split, walk_forward_splits
from utils.features import get_feature_columns, get_target_columns
from utils.hopsworks_client import get_feature_store
from utils.logger import get_logger
from utils.metrics import (
    print_metrics_by_horizon,
    quantile_coverage,
    regression_metrics,
)

logger = get_logger(__name__)

FEATURE_GROUP_NAME = "aqi_features"
FEATURE_GROUP_VERSION = 1

HORIZONS = ["24h", "48h", "72h"]


# ========================================================================
# Data loading & prep
# ========================================================================

def load_training_data() -> pd.DataFrame:
    """Read the feature group, keep only rows with full target coverage."""
    logger.info("Reading aqi_features from Hopsworks")
    fs = get_feature_store()
    fg = fs.get_feature_group(FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)
    df = fg.read()

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)

    logger.info("Total rows: %d", len(df))
    df = df[df["has_target"] == 1].copy()
    logger.info("Rows with full targets: %d", len(df))
    return df


def split_xy(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, pd.Series]]:
    # SHAP-pruned 5 features (matches train_champion.py)
    feature_cols = ["aqi", "month", "aqi_lag_24h", "aqi_lag_72h", "humidity"]
    target_cols = get_target_columns()

    X = df[feature_cols].copy()
    y = {
        h: df[f"target_aqi_{h}"]
        for h in HORIZONS
        if f"target_aqi_{h}" in target_cols
    }
    return X, y


# ========================================================================
# Model builders (per horizon — call these for each horizon target)
# ========================================================================

def fit_naive(_X_train, y_train, X_holdout) -> np.ndarray:
    """
    Naive baseline: predict the holdout's CURRENT aqi as the future aqi.

    Since `aqi` is a feature column, we can just read it directly off the
    holdout features. No fitting needed.
    """
    return X_holdout["aqi"].to_numpy()


def fit_ridge(X_train, y_train, X_holdout, quick: bool = False):
    """Ridge with StandardScaler + alpha grid search via walk-forward CV."""
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge()),
    ])

    alpha_grid = (
        [0.1, 1.0, 10.0]
        if quick
        else [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    )
    param_grid = {"ridge__alpha": alpha_grid}

    cv_splits = walk_forward_splits(len(X_train), n_splits=CV_N_SPLITS)
    search = GridSearchCV(
        pipe,
        param_grid=param_grid,
        cv=cv_splits,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    logger.info("    best alpha=%s | CV RMSE=%.3f",
                search.best_params_["ridge__alpha"],
                -search.best_score_)
    return search.best_estimator_, search.best_estimator_.predict(X_holdout)


def fit_rf(X_train, y_train, X_holdout, quick: bool = False):
    """Random Forest with small grid search."""
    if quick:
        param_grid = {"n_estimators": [50], "max_depth": [10]}
    else:
        param_grid = {
            "n_estimators": [100, 200],
            "max_depth": [10, 20, None],
        }

    cv_splits = walk_forward_splits(len(X_train), n_splits=CV_N_SPLITS)
    search = GridSearchCV(
        RandomForestRegressor(random_state=42, n_jobs=-1),
        param_grid=param_grid,
        cv=cv_splits,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    logger.info("    best params=%s | CV RMSE=%.3f",
                search.best_params_, -search.best_score_)
    return search.best_estimator_, search.best_estimator_.predict(X_holdout)


def fit_xgb_main(X_train, y_train, X_holdout, quick: bool = False):
    """XGBoost with RandomizedSearchCV — the heavy HPT step."""
    n_iter = 5 if quick else HPT_N_ITER

    param_dist = {
        "n_estimators":     [200, 400, 600, 800],
        "max_depth":        [4, 6, 8, 10],
        "learning_rate":    [0.01, 0.03, 0.05, 0.1],
        "subsample":        [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        "min_child_weight": [1, 3, 5, 7],
    }

    cv_splits = walk_forward_splits(len(X_train), n_splits=HPT_CV_FOLDS)
    search = RandomizedSearchCV(
        XGBRegressor(
            objective="reg:squarederror",
            random_state=42,
            n_jobs=1,   # parallelize at the search level instead
            tree_method="hist",
        ),
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=cv_splits,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        random_state=42,
        verbose=0,
    )
    search.fit(X_train, y_train)
    logger.info("    best params=%s | CV RMSE=%.3f",
                search.best_params_, -search.best_score_)
    return search.best_estimator_, search.best_estimator_.predict(X_holdout)


def fit_xgb_quantile(X_train, y_train, X_holdout, alpha: float):
    """XGBoost quantile regression at the given quantile (no HPT)."""
    model = XGBRegressor(
        objective="reg:quantileerror",
        quantile_alpha=alpha,
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
    )
    model.fit(X_train, y_train)
    return model, model.predict(X_holdout)


# ========================================================================
# Main orchestration
# ========================================================================

def train_all_models(quick: bool = False) -> None:
    """Train all main models and quantile models, print evaluation table."""
    df = load_training_data()

    train_df, holdout_df = chronological_holdout_split(
        df, train_fraction=TRAIN_TEST_SPLIT
    )
    logger.info("Train pool: %d rows | Holdout: %d rows",
                len(train_df), len(holdout_df))
    logger.info("Train ends: %s | Holdout starts: %s",
                train_df["timestamp"].iloc[-1],
                holdout_df["timestamp"].iloc[0])

    X_train, y_train_dict = split_xy(train_df)
    X_holdout, y_holdout_dict = split_xy(holdout_df)

    # Replace any NaNs in the FEATURE matrix with the median of each column.
    # Lag features for the very first rows of the dataset are NaN.
    feature_medians = X_train.median()
    X_train = X_train.fillna(feature_medians)
    X_holdout = X_holdout.fillna(feature_medians)

    results: Dict[str, Dict[str, Dict[str, float]]] = {
        "Naive":   {},
        "Ridge":   {},
        "RF":      {},
        "XGBoost": {},
    }

    for h in HORIZONS:
        logger.info("\n=== Horizon: %s ===", h)
        y_train = y_train_dict[h]
        y_holdout = y_holdout_dict[h]

        logger.info("  [Naive] no fit needed")
        pred = fit_naive(X_train, y_train, X_holdout)
        results["Naive"][h] = regression_metrics(y_holdout, pred)

        logger.info("  [Ridge] fitting with grid search")
        t0 = time.time()
        _, pred = fit_ridge(X_train, y_train, X_holdout, quick=quick)
        logger.info("    fit time: %.1fs", time.time() - t0)
        results["Ridge"][h] = regression_metrics(y_holdout, pred)

        logger.info("  [RF] fitting with grid search")
        t0 = time.time()
        _, pred = fit_rf(X_train, y_train, X_holdout, quick=quick)
        logger.info("    fit time: %.1fs", time.time() - t0)
        results["RF"][h] = regression_metrics(y_holdout, pred)

        logger.info("  [XGBoost] fitting with RandomizedSearchCV")
        t0 = time.time()
        _, pred = fit_xgb_main(X_train, y_train, X_holdout, quick=quick)
        logger.info("    fit time: %.1fs", time.time() - t0)
        results["XGBoost"][h] = regression_metrics(y_holdout, pred)

    # ============================================================
    # Quantile evaluation — only at 24h horizon for now
    # (extending to 48h/72h is straightforward but doubles work)
    # ============================================================
    logger.info("\n=== Quantile models (24h horizon) ===")
    y_train = y_train_dict["24h"]
    y_holdout = y_holdout_dict["24h"]

    logger.info("  [XGBoost p10] fitting")
    _, pred_p10 = fit_xgb_quantile(
        X_train, y_train, X_holdout, alpha=XGBOOST_QUANTILE_LOW
    )

    logger.info("  [XGBoost p90] fitting")
    _, pred_p90 = fit_xgb_quantile(
        X_train, y_train, X_holdout, alpha=XGBOOST_QUANTILE_HIGH
    )

    coverage = quantile_coverage(y_holdout, pred_p10, pred_p90)
    nominal = XGBOOST_QUANTILE_HIGH - XGBOOST_QUANTILE_LOW
    logger.info("  Empirical coverage: %.2f%% (nominal: %.0f%%)",
                coverage * 100, nominal * 100)

    # ============================================================
    # Final report
    # ============================================================
    print("\n" + "=" * 50)
    print("HOLDOUT METRICS — main models")
    print("=" * 50)
    print_metrics_by_horizon(results)

    print("\n" + "=" * 50)
    print("QUANTILE COVERAGE (24h horizon)")
    print("=" * 50)
    print(f"  Empirical: {coverage:.2%}  Nominal: {nominal:.0%}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train classical + XGBoost models for AQI forecasting"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use smaller hyperparameter grids for testing (~1-2 min)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.quick:
        logger.info("QUICK MODE: small grids, 5 trials for XGBoost")
    try:
        t0 = time.time()
        train_all_models(quick=args.quick)
        logger.info("\nTotal training time: %.1fs", time.time() - t0)
    except Exception as exc:
        logger.exception("Training failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()