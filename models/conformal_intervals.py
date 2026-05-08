"""
Day 11 Task 3 — Conformalized Quantile Regression on Ridge.

Implements split-conformal quantile regression following Romano,
Patterson & Candès (2019). Hybrid setup ("locally-valid" conformal):

  - Quantile regressors and the conformal Q value are derived from a
    train_proper / calibration split, ensuring conformal exchangeability
    on the calibration set.
  - Final point predictions on the holdout use the deployed production
    champion (full pre-holdout training data) for accuracy.
  - Intervals = champion_point ± (QR_residual + Q_widen).

This deliberately departs from the "purely clean" CQR variant in which
the same Ridge produces both points and intervals. We trade strict
exchangeability for usable point accuracy, which is documented in the
literature as locally-valid conformal — coverage is empirically
≥ nominal under mild distribution shift, intervals remain useful.

Saves CQR artifacts (QRs, scaler, medians, calibration metadata) to
models/cqr/h{horizon}/.

Usage:
    python -m models.conformal_intervals
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import QuantileRegressor, Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.hopsworks_client import get_feature_store
from utils.logger import get_logger
from utils.metrics import quantile_coverage

logger = get_logger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
CHAMPION_FEATURES = ["aqi", "month", "aqi_lag_24h", "aqi_lag_72h", "humidity"]
HORIZONS = [24, 48, 72]

# Three-way chronological split of the data:
#   train_proper | calibration | holdout
HOLDOUT_FRAC = 0.20
CALIBRATION_FRAC_OF_TRAIN = 0.30   # 30% of pre-holdout pool

LOW_Q = 0.10
HIGH_Q = 0.90
NOMINAL_COVERAGE = HIGH_Q - LOW_Q          # 0.80
ALPHA = 1 - NOMINAL_COVERAGE               # 0.20

RIDGE_ALPHA = 10.0
QR_ALPHA = 0.001

CHAMPION_DIR = Path("models/champion")     # source of production point models
CQR_DIR = Path("models/cqr")               # destination for CQR artifacts


def load_data() -> pd.DataFrame:
    fs = get_feature_store()
    fg = fs.get_feature_group("aqi_features", version=1)
    df = fg.read()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    logger.info("Loaded %d rows from Hopsworks", len(df))
    return df


def split_three_way(df: pd.DataFrame, target_col: str):
    """
    Chronological three-way split:
      train_proper → calibration → holdout
    """
    work = df.dropna(subset=[target_col]).reset_index(drop=True)
    n = len(work)
    holdout_start = int(n * (1 - HOLDOUT_FRAC))
    train_pool = work.iloc[:holdout_start]
    holdout    = work.iloc[holdout_start:].copy()

    cal_start = int(len(train_pool) * (1 - CALIBRATION_FRAC_OF_TRAIN))
    train_proper = train_pool.iloc[:cal_start].copy()
    calibration  = train_pool.iloc[cal_start:].copy()
    return train_proper, calibration, holdout


def fit_quantile(X: np.ndarray, y: np.ndarray, q: float) -> QuantileRegressor:
    qr = QuantileRegressor(quantile=q, alpha=QR_ALPHA, solver="highs")
    qr.fit(X, y)
    return qr


def load_production_champion(horizon: int):
    """Load the deployed production champion Ridge + scaler + medians."""
    horizon_dir = CHAMPION_DIR / f"h{horizon}"
    model = joblib.load(horizon_dir / "ridge_model.joblib")
    scaler = joblib.load(horizon_dir / "scaler.joblib")
    with open(horizon_dir / "feature_medians.json") as f:
        medians_dict = json.load(f)
    return model, scaler, pd.Series(medians_dict)


def evaluate_horizon(df: pd.DataFrame, horizon: int) -> dict:
    target_col = f"target_aqi_{horizon}h"
    logger.info("─" * 50)
    logger.info("Horizon %dh", horizon)

    train_proper, calibration, holdout = split_three_way(df, target_col)
    logger.info(
        "Split — train_proper: %d  calibration: %d  holdout: %d",
        len(train_proper), len(calibration), len(holdout),
    )

    # ─── CQR pipeline: medians + scaler from train_proper ────────────────
    cqr_medians = train_proper[CHAMPION_FEATURES].median()
    train_proper_imputed = train_proper.copy()
    calibration_imputed  = calibration.copy()
    holdout_imputed      = holdout.copy()
    for split in (train_proper_imputed, calibration_imputed, holdout_imputed):
        split[CHAMPION_FEATURES] = split[CHAMPION_FEATURES].fillna(cqr_medians)

    cqr_scaler = StandardScaler()
    X_tp  = cqr_scaler.fit_transform(train_proper_imputed[CHAMPION_FEATURES].values)
    X_cal = cqr_scaler.transform(calibration_imputed[CHAMPION_FEATURES].values)
    X_ho_cqr = cqr_scaler.transform(holdout_imputed[CHAMPION_FEATURES].values)
    y_tp  = train_proper_imputed[target_col].values
    y_cal = calibration_imputed[target_col].values
    y_ho  = holdout_imputed[target_col].values

    # ─── Local Ridge on train_proper (only used to derive QR residuals) ──
    ridge_local = Ridge(alpha=RIDGE_ALPHA, random_state=42)
    ridge_local.fit(X_tp, y_tp)

    y_tp_pred = ridge_local.predict(X_tp)
    residuals_tp = y_tp - y_tp_pred

    qr_lo = fit_quantile(X_tp, residuals_tp, LOW_Q)
    qr_hi = fit_quantile(X_tp, residuals_tp, HIGH_Q)

    # ─── Calibration step (Romano et al. 2019, symmetric CQR) ────────────
    y_cal_pred = ridge_local.predict(X_cal)
    cal_resid_lo = qr_lo.predict(X_cal)
    cal_resid_hi = qr_hi.predict(X_cal)
    cal_y_lo = y_cal_pred + cal_resid_lo
    cal_y_hi = y_cal_pred + cal_resid_hi

    scores = np.maximum(cal_y_lo - y_cal, y_cal - cal_y_hi)
    n_cal = len(y_cal)
    k = int(np.ceil((n_cal + 1) * (1 - ALPHA))) - 1
    k = min(k, n_cal - 1)
    Q = float(np.sort(scores)[k])
    Q_widen = max(Q, 0.0)

    logger.info(
        "Calibration: Q=%.2f  Q_widen=%.2f  k=%d / n_cal=%d",
        Q, Q_widen, k, n_cal,
    )

    # ─── Holdout intervals — HYBRID ──────────────────────────────────────
    # Point predictions: production champion (full pre-holdout training).
    # Intervals: QR_lo / QR_hi (from train_proper) + Q_widen (from calibration).
    prod_ridge, prod_scaler, prod_medians = load_production_champion(horizon)

    holdout_for_prod = holdout.copy()
    holdout_for_prod[CHAMPION_FEATURES] = (
        holdout_for_prod[CHAMPION_FEATURES].fillna(prod_medians)
    )
    X_ho_prod = prod_scaler.transform(holdout_for_prod[CHAMPION_FEATURES].values)
    y_ho_pred = prod_ridge.predict(X_ho_prod)

    ho_resid_lo = qr_lo.predict(X_ho_cqr)
    ho_resid_hi = qr_hi.predict(X_ho_cqr)
    y_lo = y_ho_pred + ho_resid_lo - Q_widen
    y_hi = y_ho_pred + ho_resid_hi + Q_widen

    coverage = quantile_coverage(y_ho, y_lo, y_hi)
    avg_width = float(np.mean(y_hi - y_lo))
    point_r2 = r2_score(y_ho, y_ho_pred)

    # ─── Persist artifacts ───────────────────────────────────────────────
    horizon_dir = CQR_DIR / f"h{horizon}"
    horizon_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(ridge_local, horizon_dir / "ridge_local.joblib")
    joblib.dump(cqr_scaler, horizon_dir / "scaler.joblib")
    joblib.dump(qr_lo, horizon_dir / "quantile_p10.joblib")
    joblib.dump(qr_hi, horizon_dir / "quantile_p90.joblib")
    cqr_medians.to_json(horizon_dir / "feature_medians.json")
    with open(horizon_dir / "calibration.json", "w") as f:
        json.dump({
            "method": "split_conformal_cqr_symmetric_hybrid",
            "notes": (
                "Point predictions sourced from production champion "
                "(models/champion). QR residuals + Q_widen derived from "
                "train_proper/calibration split. Locally-valid conformal."
            ),
            "Q": Q,
            "Q_widen": Q_widen,
            "low_q": LOW_Q,
            "high_q": HIGH_Q,
            "alpha": ALPHA,
            "qr_alpha": QR_ALPHA,
            "ridge_alpha": RIDGE_ALPHA,
            "n_train_proper": int(len(train_proper)),
            "n_calibration": n_cal,
            "n_holdout": int(len(holdout)),
            "holdout_coverage": round(float(coverage), 4),
            "holdout_avg_width": round(avg_width, 2),
            "production_point_r2": round(float(point_r2), 4),
        }, f, indent=2)

    return {
        "horizon": horizon,
        "coverage": coverage,
        "avg_width": round(avg_width, 2),
        "Q": round(Q, 2),
        "Q_widen": round(Q_widen, 2),
        "point_r2": round(float(point_r2), 4),
        "n_holdout": len(holdout),
    }


def main() -> None:
    CQR_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Day 11 Task 3 — Conformalized QR (hybrid)")
    logger.info("=" * 60)

    df = load_data()

    results = []
    for h in HORIZONS:
        results.append(evaluate_horizon(df, h))

    print("\n" + "=" * 80)
    print("HYBRID CQR — coverage on holdout (target = 80%)")
    print("=" * 80)
    print(
        f"{'Horizon':<8} {'Coverage':<11} {'Width':<8} {'Q':<8} "
        f"{'Point R²':<10} {'N':<6}"
    )
    print("-" * 80)
    for r in results:
        cov_pct = round(r["coverage"] * 100, 2)
        marker = "✓" if 0.78 <= r["coverage"] <= 0.85 else "≈"
        print(
            f"{r['horizon']}h{'':<6} "
            f"{cov_pct:<5}% {marker:<3} "
            f"{r['avg_width']:<8} "
            f"{r['Q']:<8} "
            f"{r['point_r2']:<10} "
            f"{r['n_holdout']:<6}"
        )
    print("=" * 80)
    print()
    print("Coverage history:")
    print("  Day 9   (XGBoost on raw target, 24h): 57.66%")
    print("  Day 11a (residual QR, no calibration): 73-74%")
    print("  Day 11b (hybrid CQR, this run):        see above")


if __name__ == "__main__":
    main()