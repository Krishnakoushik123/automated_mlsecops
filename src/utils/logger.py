"""
src/utils/logger.py
-------------------
Structured JSON logger factory for every module in the project.
"""

from __future__ import annotations

import logging
import sys
from typing import Optional

try:
    from pythonjsonlogger import jsonlogger
    _JSON_AVAILABLE = True
except ImportError:
    _JSON_AVAILABLE = False


def get_logger(
    name: str,
    level: Optional[str] = None,
    json_format: bool = False,
) -> logging.Logger:
    """Return a named logger.

    Parameters
    ----------
    name:
        Typically ``__name__`` of the calling module.
    level:
        Override log level (e.g. ``"DEBUG"``).  Falls back to the
        ``MLSECOPS_LOG_LEVEL`` env var, then ``INFO``.
    json_format:
        Emit JSON lines instead of plain text (useful in containers).
    """
    import os
    effective_level = level or os.getenv("MLSECOPS_LOG_LEVEL", "INFO")
    numeric_level = getattr(logging, effective_level.upper(), logging.INFO)

    logger = logging.getLogger(name)
    if logger.handlers:
        # Already configured – just adjust level if needed
        logger.setLevel(numeric_level)
        return logger

    logger.setLevel(numeric_level)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(numeric_level)

    if json_format and _JSON_AVAILABLE:
        fmt = jsonlogger.JsonFormatter(
            "%(asctime)s %(name)s %(levelname)s %(message)s"
        )
    else:
        fmt = logging.Formatter(
            "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    handler.setFormatter(fmt)
    logger.addHandler(handler)
    logger.propagate = False
    return logger
