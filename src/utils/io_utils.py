"""
IO helpers for reading/writing common artifacts.

Parquet is the canonical storage format for raw and processed datasets.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_parquet(df: pd.DataFrame, path: str | Path, *, index: bool = False) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    df.to_parquet(path, index=index)


def read_parquet(path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(Path(path))


def append_csv_row(path: str | Path, row: dict, *, header_if_new: bool = True) -> None:
    """
    Append a single row to a CSV file.
    Intended for live logging; avoids holding the entire file in memory.
    """
    path = Path(path)
    ensure_dir(path.parent)

    df = pd.DataFrame([row])
    exists = path.exists()
    df.to_csv(path, mode="a", header=(header_if_new and not exists), index=False)

