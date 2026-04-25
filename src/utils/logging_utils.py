"""
Logging utilities.

Keeps logging configuration in one place so modules can do:

    from src.utils.logging_utils import get_logger, setup_logging
"""

from __future__ import annotations

import logging
import os
from typing import Optional


def setup_logging(level: str | int | None = None) -> None:
    """
    Configure root logging with a concise format.
    Call this once from CLI entrypoints.
    """
    if level is None:
        level = os.getenv("LOG_LEVEL", "INFO")

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def get_logger(name: Optional[str] = None) -> logging.Logger:
    return logging.getLogger(name if name else "funding_harvest")

