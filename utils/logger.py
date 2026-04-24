"""
Centralized logging setup for all pipelines.

Every script that wants to log should call:
    from utils.logger import get_logger
    logger = get_logger(__name__)

Logs go to BOTH stdout (for GitHub Actions / terminal visibility) and a
rotating file under logs/ (for local debugging). File logs are gitignored.
"""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

# Project root = parent of utils/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

LOG_FILE = LOG_DIR / "pearls_aqi.log"
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Track configured loggers to avoid duplicate handlers on re-import
_configured_loggers = set()


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Return a logger configured with stdout + rotating file handlers.

    Args:
        name: Logger name, typically __name__ of the calling module.
        level: Logging level (default INFO).

    Returns:
        A configured logging.Logger instance.
    """
    logger = logging.getLogger(name)

    if name in _configured_loggers:
        return logger

    logger.setLevel(level)
    logger.propagate = False

    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    # Stdout — for terminal and CI visibility
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # Rotating file — 5 MB per file, keep last 3
    file_handler = RotatingFileHandler(
        LOG_FILE,
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    _configured_loggers.add(name)
    return logger