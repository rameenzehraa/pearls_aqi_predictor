"""
Sequence builder for LSTM training.

Converts tabular feature dataframes into 3D tensors of shape
(samples, timesteps, features) suitable for Keras LSTM layers.

Hard rules:
- Sequences are built chronologically. No shuffling inside this module.
- A sample at row i uses features from rows [i-LOOKBACK+1, i] inclusive.
- Rows where the lookback window would go before the dataframe start are dropped.
- Rows where any target is NaN are dropped (must align with has_target flag).
- Targets are returned as a 2D array (samples, n_horizons) for multi-output training.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from utils.logger import get_logger

logger = get_logger(__name__)

LOOKBACK_HOURS = 48


def build_sequences(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_cols: list[str],
    lookback: int = LOOKBACK_HOURS,
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """
    Build (X, y, timestamps) tensors for LSTM training.

    Parameters
    ----------
    df : pd.DataFrame
        Must be sorted chronologically by timestamp ascending.
        Must contain feature_cols and target_cols.
    feature_cols : list[str]
        Columns to use as model inputs.
    target_cols : list[str]
        Columns to predict (e.g. ["target_24h", "target_48h", "target_72h"]).
    lookback : int
        Number of timesteps in each input sequence.

    Returns
    -------
    X : np.ndarray, shape (n_samples, lookback, n_features)
    y : np.ndarray, shape (n_samples, n_targets)
    timestamps : pd.DatetimeIndex
        The timestamp of each sample's *prediction origin* (the last row
        in its lookback window). Used downstream for chronological splits.
    """
    if df.index.name != "timestamp" and "timestamp" not in df.columns:
        raise ValueError("df must have 'timestamp' as index or column")

    # Standardise on timestamp as a column for slicing
    work = df.copy()
    if work.index.name == "timestamp":
        work = work.reset_index()

    # Verify chronological order
    if not work["timestamp"].is_monotonic_increasing:
        raise ValueError("df must be sorted by timestamp ascending")

    # Drop rows where any target is NaN
    target_mask = work[target_cols].notna().all(axis=1)
    valid_idx = work.index[target_mask].to_list()

    # For each valid row, check we have enough history for the lookback window
    samples_X = []
    samples_y = []
    samples_ts = []

    feature_array = work[feature_cols].to_numpy(dtype=np.float32)
    target_array = work[target_cols].to_numpy(dtype=np.float32)
    timestamp_array = work["timestamp"].to_numpy()

    for i in valid_idx:
        if i - lookback + 1 < 0:
            continue
        window = feature_array[i - lookback + 1 : i + 1]
        if np.isnan(window).any():
            continue
        samples_X.append(window)
        samples_y.append(target_array[i])
        samples_ts.append(timestamp_array[i])

    if not samples_X:
        raise ValueError(
            "No valid sequences could be built. Check lookback, NaN handling, "
            "and target column alignment."
        )

    X = np.stack(samples_X, axis=0)
    y = np.stack(samples_y, axis=0)
    timestamps = pd.DatetimeIndex(samples_ts)

    logger.info(
        "Built %d sequences | X shape %s | y shape %s | first ts %s | last ts %s",
        len(X),
        X.shape,
        y.shape,
        timestamps[0],
        timestamps[-1],
    )

    return X, y, timestamps