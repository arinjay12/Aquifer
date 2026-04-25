"""
Validation helpers for dataframes and config.

These are intentionally lightweight and non-opinionated.
"""

from __future__ import annotations

from typing import Iterable

import pandas as pd


def require_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def require_monotonic_increasing(df: pd.DataFrame, column: str) -> None:
    if not df[column].is_monotonic_increasing:
        raise ValueError(f"Column {column} must be monotonic increasing")

