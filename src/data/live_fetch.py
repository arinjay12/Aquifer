"""
Live fetch utilities.

Fetches spot/perp ticker prices plus the futures premium index snapshot:
- spot ticker price
- futures ticker price
- mark price, index price
- last funding rate and next funding timestamp
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from src.data.binance_client import BinanceClient


@dataclass(frozen=True)
class LiveSnapshot:
    timestamp_utc: pd.Timestamp
    spot_price: float
    perp_price: float
    mark_price: float | None
    index_price: float | None
    funding_rate_last: float | None
    funding_rate_next: float | None
    next_funding_time: pd.Timestamp | None


def fetch_live_snapshot(client: BinanceClient, symbol_spot: str, symbol_perp: str) -> LiveSnapshot:
    # Use exchange-reported prices; for v1, spot/perp ticker prices are sufficient.
    spot = client.get_spot_ticker_price(symbol_spot)
    perp = client.get_perp_ticker_price(symbol_perp)
    prem = client.get_mark_price_snapshot(symbol_perp)

    ts = prem.get("time_utc")
    if ts is None:
        ts = pd.Timestamp.utcnow().tz_localize("UTC")

    mark = prem.get("markPrice")
    idx = prem.get("indexPrice")
    last_fr = prem.get("lastFundingRate")
    next_ft = prem.get("nextFundingTime_utc")

    # Binance does not provide "next funding rate" directly in the snapshot; for v1 we proxy with lastFundingRate.
    next_fr = last_fr

    return LiveSnapshot(
        timestamp_utc=pd.Timestamp(ts),
        spot_price=float(spot),
        perp_price=float(perp),
        mark_price=float(mark) if mark is not None else None,
        index_price=float(idx) if idx is not None else None,
        funding_rate_last=float(last_fr) if last_fr is not None else None,
        funding_rate_next=float(next_fr) if next_fr is not None else None,
        next_funding_time=pd.Timestamp(next_ft) if next_ft is not None else None,
    )
