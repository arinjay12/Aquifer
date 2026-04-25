"""
Live observation logger.

Always appends every observation (one row per poll) to a CSV.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from src.utils.io_utils import append_csv_row, ensure_dir


LIVE_COLUMNS = [
    "timestamp_utc",
    "instrument",
    "direction",
    "spot_price",
    "perp_price",
    "basis_bps",
    "funding_rate_last",
    "funding_rate_next",
    "next_funding_time",
    "minutes_to_next_funding",
    "expected_edge_bps",
    "confidence_score",
    "state",
    "signal_flag",
    "entry_flag",
    "exit_flag",
    "note",
]


def append_live_row(path: str | Path, row: Dict[str, Any]) -> None:
    """
    Append a row, ensuring stable column ordering.
    """
    ordered = {c: row.get(c) for c in LIVE_COLUMNS}
    append_csv_row(path, ordered, header_if_new=True)


def read_last_rows(path: str | Path, n: int = 2000) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=LIVE_COLUMNS)
    df = pd.read_csv(path)
    if len(df) > n:
        df = df.tail(n)
    if "timestamp_utc" in df.columns:
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
    if "next_funding_time" in df.columns:
        df["next_funding_time"] = pd.to_datetime(df["next_funding_time"], utc=True, errors="coerce")
    return df
