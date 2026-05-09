"""
API key authentication for the Flask backend.

The backend exposes prediction endpoints over the public internet
(deployed on Render), so we gate access with a shared secret. The
Streamlit dashboard sends the key in an `X-API-Key` header; everyone
else gets a 401.

The actual key value lives in environment variables (Render dashboard
secrets), never in code or git.
"""

import os
from functools import wraps

from flask import jsonify, request


def get_expected_key() -> str:
    """Read the expected API key from env. Raises if missing."""
    key = os.getenv("BACKEND_API_KEY")
    if not key:
        raise RuntimeError(
            "BACKEND_API_KEY is not set. Configure it in Render's "
            "environment variables (or .env for local dev)."
        )
    return key


def require_api_key(view_func):
    """
    Decorator that rejects requests without a valid X-API-Key header.

    Usage:
        @app.route("/predictions")
        @require_api_key
        def predictions():
            ...
    """
    @wraps(view_func)
    def wrapper(*args, **kwargs):
        provided = request.headers.get("X-API-Key", "")
        expected = get_expected_key()
        if provided != expected:
            return jsonify({"error": "Invalid or missing API key"}), 401
        return view_func(*args, **kwargs)
    return wrapper