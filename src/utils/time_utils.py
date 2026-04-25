"""
Time utilities for UTC alignment.

No exchange-specific logic here; just helpful helpers for later steps.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd


def now_utc() -> pd.Timestamp:
    return pd.Timestamp(datetime.now(timezone.utc))


def to_utc_timestamp(value) -> pd.Timestamp:
    """
    Convert value into a timezone-aware UTC pandas Timestamp.
    Accepts pd.Timestamp, datetime, int ms, or ISO string.
    """
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def floor_to_minute_utc(ts: pd.Timestamp) -> pd.Timestamp:
    ts = to_utc_timestamp(ts)
    return ts.floor("min")

