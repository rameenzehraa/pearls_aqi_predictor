"""
Hopsworks connection helper.

Centralizes login so feature pipelines, training pipelines, and the
backend API all share one auth path. Reads credentials from .env.

Usage:
    from utils.hopsworks_client import get_project, get_feature_store

    project = get_project()
    fs = get_feature_store()
"""

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root regardless of where the caller runs from
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


@lru_cache(maxsize=1)
def get_project():
    """
    Log in to Hopsworks and return the project handle.

    Cached for the lifetime of the process — repeated calls in one
    script reuse the same connection.
    """
    import hopsworks

    api_key = os.getenv("HOPSWORKS_API_KEY")
    project_name = os.getenv("HOPSWORKS_PROJECT")

    if not api_key:
        raise RuntimeError(
            "HOPSWORKS_API_KEY missing. Check .env at project root."
        )
    if not project_name:
        raise RuntimeError(
            "HOPSWORKS_PROJECT missing. Check .env at project root."
        )

    return hopsworks.login(
        api_key_value=api_key,
        project=project_name,
    )


def get_feature_store():
    """Return the project's feature store handle."""
    return get_project().get_feature_store()


def get_model_registry():
    """Return the project's model registry handle."""
    return get_project().get_model_registry()