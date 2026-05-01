"""
Walk-forward cross-validation helpers for time-series model selection.

Standard k-fold CV leaks future into past on time series. Walk-forward
CV trains on a contiguous past window and validates on the next chunk,
so every validation sample is strictly later than every training sample.

We use this CV scheme:
  - INSIDE the training pool (first 80% of the timeline) for
    hyperparameter tuning and model comparison.
  - The final 20% holdout is touched ONCE at the end for unbiased
    metric reporting — never seen during CV.
"""

from typing import Iterator, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from config.config import CV_N_SPLITS


def walk_forward_splits(
    n_samples: int,
    n_splits: int = CV_N_SPLITS,
) -> list:
    """
    Return a list of (train_idx, val_idx) tuples for walk-forward CV.

    Wraps sklearn's TimeSeriesSplit. Returns a list (not a generator)
    so the splits can be iterated multiple times by the caller.

    Args:
        n_samples: total rows in the training pool
        n_splits:  number of CV folds (default from config)
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    return list(tscv.split(np.arange(n_samples)))


def chronological_holdout_split(
    df: pd.DataFrame,
    train_fraction: float,
    timestamp_col: str = "timestamp",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Sort a DataFrame by timestamp and split it chronologically.

    Returns:
        (train_pool, holdout) — first `train_fraction` and last
        `1 - train_fraction` of the rows by time.
    """
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1")

    sorted_df = df.sort_values(timestamp_col).reset_index(drop=True)
    cutoff = int(len(sorted_df) * train_fraction)
    return sorted_df.iloc[:cutoff].copy(), sorted_df.iloc[cutoff:].copy()