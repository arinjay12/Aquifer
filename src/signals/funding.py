"""
Funding-related signal helpers.
"""

from __future__ import annotations

from typing import Union

import numpy as np
import pandas as pd


TimestampLike = Union[pd.Timestamp, str, int, float]


def compute_minutes_to_next_funding(timestamp_utc: TimestampLike, next_funding_time: TimestampLike) -> float:
    """
    Compute minutes from timestamp_utc to next_funding_time.

    Returns a float number of minutes (can be fractional if inputs aren't minute-aligned).
    """
    ts = pd.Timestamp(timestamp_utc)
    nft = pd.Timestamp(next_funding_time)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    if nft.tzinfo is None:
        nft = nft.tz_localize("UTC")
    else:
        nft = nft.tz_convert("UTC")

    if pd.isna(ts) or pd.isna(nft):
        return float("nan")

    return float((nft - ts).total_seconds() / 60.0)
