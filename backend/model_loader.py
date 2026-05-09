"""
Model loader — fetches Ridge champion + CQR models from Hopsworks
Model Registry on first run, caches them to a local directory for
fast loads on subsequent restarts.

Cache location: /tmp/aqi_models/  (Render's writable filesystem)

Caching strategy:
    - First call: download all 6 models from Hopsworks (~30-60s)
    - Subsequent calls in same process: hit in-memory cache (instant)
    - After process restart: load from disk cache (~1s)
    - Force refresh: delete the cache directory and restart

This matches "Approach B" from the design discussion — favoring fast
cold starts over absolute model freshness. Daily retraining produces
new versions in the registry; backend picks them up on the next
deploy/restart after cache invalidation.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

import joblib

# Make project-root imports work
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.hopsworks_client import get_model_registry
from utils.logger import get_logger

logger = get_logger(__name__)

# Cache directory — /tmp/ on Render is writable and ephemeral
# (cleared on every cold start, which is fine for a download-once cache)
CACHE_DIR = Path(os.getenv("MODEL_CACHE_DIR", "/tmp/aqi_models"))

HORIZONS = [24, 48, 72]


def _cache_path(model_name: str) -> Path:
    return CACHE_DIR / model_name


def _is_cached(model_name: str) -> bool:
    """Check if a model has been downloaded to local cache."""
    p = _cache_path(model_name)
    # A successfully-downloaded model dir contains the joblib + json files
    return p.exists() and any(p.iterdir())


def _get_latest_version(mr, model_name: str) -> int:
    """Find the highest version number for a model in the registry."""
    versions = mr.get_models(model_name)
    if not versions:
        raise RuntimeError(f"Model '{model_name}' not found in registry")
    return max(m.version for m in versions)


def _download_model(mr, model_name: str) -> Path:
    """
    Download the latest version of a model from Hopsworks to local cache.
    Returns the path to the cached directory.
    """
    cache = _cache_path(model_name)
    if _is_cached(model_name):
        logger.info("Cache hit: %s", model_name)
        return cache

    latest_version = _get_latest_version(mr, model_name)
    logger.info("Cache miss: downloading %s v%d from registry", model_name, latest_version)

    model = mr.get_model(model_name, version=latest_version)
    download_path = model.download()

    # download_path is a temp dir; copy into our stable cache location
    cache.mkdir(parents=True, exist_ok=True)
    for item in Path(download_path).iterdir():
        if item.is_file():
            shutil.copy2(item, cache / item.name)

    logger.info("Downloaded %s v%d to %s", model_name, model.version, cache)
    return cache


def load_all_models() -> tuple[dict, dict]:
    """
    Load all 6 models (3 champion + 3 CQR), downloading from registry
    if not already cached.

    Returns:
        (champion_dict, cqr_dict) in the same shape the Flask app expects.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Connect to registry only if we need to download anything
    needs_download = any(
        not _is_cached(f"karachi_aqi_ridge_{h}h") or not _is_cached(f"karachi_aqi_cqr_{h}h")
        for h in HORIZONS
    )

    if needs_download:
        logger.info("One or more models not cached; connecting to model registry")
        mr = get_model_registry()
        for h in HORIZONS:
            _download_model(mr, f"karachi_aqi_ridge_{h}h")
            _download_model(mr, f"karachi_aqi_cqr_{h}h")
    else:
        logger.info("All 6 models found in cache; skipping registry connection")

    # Now load from cache (always succeeds at this point)
    champion = {}
    cqr = {}
    for h in HORIZONS:
        ch_dir = _cache_path(f"karachi_aqi_ridge_{h}h")
        cq_dir = _cache_path(f"karachi_aqi_cqr_{h}h")

        champion[h] = {
            "model": joblib.load(ch_dir / "ridge_model.joblib"),
            "scaler": joblib.load(ch_dir / "scaler.joblib"),
        }
        cqr[h] = {
            "p10": joblib.load(cq_dir / "quantile_p10.joblib"),
            "p90": joblib.load(cq_dir / "quantile_p90.joblib"),
            "scaler": joblib.load(cq_dir / "scaler.joblib"),
            "calibration": json.loads((cq_dir / "calibration.json").read_text()),
        }

    return champion, cqr


def load_metadata() -> dict:
    """Load champion training metadata from the cached 24h champion."""
    metadata_path = _cache_path("karachi_aqi_ridge_24h") / "champion_metadata.json"
    if metadata_path.exists():
        return json.loads(metadata_path.read_text())

    # Fallback: training metadata wasn't in the model artifact (older registrations).
    # Return a minimal placeholder so the dashboard doesn't crash.
    logger.warning("champion_metadata.json not found in cache; using fallback")
    return {
        "trained_at_utc": "unknown",
        "n_rows_loaded": 0,
        "features": ["aqi", "month", "aqi_lag_24h", "aqi_lag_72h", "humidity"],
        "horizons": [
            {"horizon": h, "metrics": {"r2": 0, "rmse": 0, "mae": 0}}
            for h in HORIZONS
        ],
    }


def clear_cache() -> None:
    """Delete the local model cache. Forces a re-download next time."""
    if CACHE_DIR.exists():
        shutil.rmtree(CACHE_DIR)
        logger.info("Cleared model cache at %s", CACHE_DIR)