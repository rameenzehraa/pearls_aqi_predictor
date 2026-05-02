"""
Day 10 — LSTM training (multi-output, 3 horizons).

Architecture:
    Input: (48, n_features)
    LSTM(64, return_sequences=True) → Dropout(0.2)
    LSTM(32) → Dropout(0.2)
    Dense(16, relu)
    Three output heads: Dense(1) each for 24h / 48h / 72h

Loss weighting: [1.0, 1.2, 1.5] for 24h / 48h / 72h
Optimizer: Adam lr=1e-3
Early stopping: patience=10 on val_loss
Holdout: same chronological split as Day 9 (last 20% of data)

Usage:
    python -m models.train_lstm
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # suppress TF noise

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.hopsworks_client import get_feature_store
from utils.logger import get_logger
from utils.metrics import print_metrics_table, evaluate
from utils.sequences import build_sequences, LOOKBACK_HOURS

logger = get_logger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────
FEATURE_GROUP_NAME = "aqi_features"
FEATURE_GROUP_VERSION = 1
TARGET_COLS = ["target_aqi_24h", "target_aqi_48h", "target_aqi_72h"]
HORIZONS = [24, 48, 72]
HOLDOUT_FRAC = 0.20
VAL_FRAC = 0.10        # fraction of training pool used for early stopping
LOOKBACK = LOOKBACK_HOURS
BATCH_SIZE = 32
MAX_EPOCHS = 100
PATIENCE = 10
LOSS_WEIGHTS = [1.0, 1.2, 1.5]   # 24h / 48h / 72h
SEED = 42

tf.random.set_seed(SEED)
np.random.seed(SEED)


# ── Feature columns (must match Day 9 exactly) ───────────────────────────────
def get_feature_cols(df: pd.DataFrame) -> list[str]:
    exclude = {"timestamp", "has_target"} | set(TARGET_COLS)
    return [c for c in df.columns if c not in exclude]


# ── Data loading ─────────────────────────────────────────────────────────────
def load_data() -> pd.DataFrame:
    logger.info("Connecting to Hopsworks …")
    fs = get_feature_store()
    fg = fs.get_feature_group(FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)
    df = fg.read()

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)
    else:
        df = df.sort_index()

    logger.info("Loaded %d rows from Hopsworks", len(df))
    return df


# ── Chronological split (mirrors Day 9) ──────────────────────────────────────
def chronological_split(
    X: np.ndarray, y: np.ndarray, timestamps: pd.DatetimeIndex
) -> tuple:
    n = len(X)
    holdout_start = int(n * (1 - HOLDOUT_FRAC))
    val_start = int(holdout_start * (1 - VAL_FRAC))

    X_train = X[:val_start]
    y_train = y[:val_start]
    X_val   = X[val_start:holdout_start]
    y_val   = y[val_start:holdout_start]
    X_test  = X[holdout_start:]
    y_test  = y[holdout_start:]
    ts_test = timestamps[holdout_start:]

    logger.info(
        "Split — train: %d  val: %d  holdout: %d",
        len(X_train), len(X_val), len(X_test)
    )
    return X_train, y_train, X_val, y_val, X_test, y_test, ts_test


# ── Scaling ───────────────────────────────────────────────────────────────────
def scale(
    X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    n_train, t, f = X_train.shape
    scaler = StandardScaler()
    X_train_2d = scaler.fit_transform(X_train.reshape(-1, f))
    X_val_2d   = scaler.transform(X_val.reshape(-1, f))
    X_test_2d  = scaler.transform(X_test.reshape(-1, f))
    return (
        X_train_2d.reshape(n_train, t, f),
        X_val_2d.reshape(len(X_val), t, f),
        X_test_2d.reshape(len(X_test), t, f),
        scaler,
    )


# ── Model definition ──────────────────────────────────────────────────────────
def build_model(n_features: int) -> keras.Model:
    inp = keras.Input(shape=(LOOKBACK, n_features), name="sequence_input")

    x = layers.LSTM(64, return_sequences=True, name="lstm_1")(inp)
    x = layers.Dropout(0.2, name="drop_1")(x)
    x = layers.LSTM(32, return_sequences=False, name="lstm_2")(x)
    x = layers.Dropout(0.2, name="drop_2")(x)
    x = layers.Dense(16, activation="relu", name="shared_dense")(x)

    out_24h = layers.Dense(1, name="out_24h")(x)
    out_48h = layers.Dense(1, name="out_48h")(x)
    out_72h = layers.Dense(1, name="out_72h")(x)

    model = keras.Model(inputs=inp, outputs=[out_24h, out_48h, out_72h])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss={"out_24h": "mse", "out_48h": "mse", "out_72h": "mse"},
        loss_weights={"out_24h": LOSS_WEIGHTS[0], "out_48h": LOSS_WEIGHTS[1], "out_72h": LOSS_WEIGHTS[2]},
        metrics={"out_24h": "mae", "out_48h": "mae", "out_72h": "mae"},
    )
    return model


# ── Training ──────────────────────────────────────────────────────────────────
def train(
    model: keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> keras.callbacks.History:
    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=PATIENCE,
        restore_best_weights=True,
        verbose=1,
    )

    history = model.fit(
        X_train,
        [y_train[:, 0], y_train[:, 1], y_train[:, 2]],
        validation_data=(
            X_val,
            [y_val[:, 0], y_val[:, 1], y_val[:, 2]],
        ),
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop],
        verbose=1,
    )
    return history


# ── Evaluation ────────────────────────────────────────────────────────────────
def evaluate_lstm(
    model: keras.Model,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    preds = model.predict(X_test, verbose=0)  # list of 3 arrays
    results = {}
    for i, h in enumerate(HORIZONS):
        y_true = y_test[:, i]
        y_pred = preds[i].squeeze()
        results[h] = evaluate(y_true, y_pred)
    return results


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    logger.info("=" * 60)
    logger.info("Day 10 — LSTM training")
    logger.info("=" * 60)

    df = load_data()
    feature_cols = get_feature_cols(df)
    logger.info("Feature count: %d", len(feature_cols))

    X, y, timestamps = build_sequences(df, feature_cols, TARGET_COLS, lookback=LOOKBACK)

    X_train, y_train, X_val, y_val, X_test, y_test, ts_test = chronological_split(
        X, y, timestamps
    )

    X_train, X_val, X_test, scaler = scale(X_train, X_val, X_test)

    model = build_model(n_features=X_train.shape[2])
    model.summary()

    logger.info("Training …")
    history = train(model, X_train, y_train, X_val, y_val)

    epochs_run = len(history.history["loss"])
    logger.info("Stopped at epoch %d / %d", epochs_run, MAX_EPOCHS)

    results = evaluate_lstm(model, X_test, y_test)

    print("\n" + "=" * 60)
    print("LSTM RESULTS (holdout)")
    print("=" * 60)
    for h in HORIZONS:
        r = results[h]
        print(f"  {h}h  RMSE={r['rmse']:.1f}  MAE={r['mae']:.1f}  R²={r['r2']:.3f}")

    print("\nRidge baseline (Day 9 reference):")
    print("  24h  RMSE=21.2  MAE=16.1  R²=0.308")
    print("  48h  RMSE=?     MAE=?     R²=0.162")
    print("  72h  RMSE=?     MAE=?     R²=0.116")
    print("=" * 60)


if __name__ == "__main__":
    main()