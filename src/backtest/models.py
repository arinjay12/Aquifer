"""
Backtest data models.

These are intentionally lightweight and serializable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import pandas as pd


class TradeState(str, Enum):
    FLAT = "FLAT"
    OPEN = "OPEN"


@dataclass
class TradeRecord:
    trade_id: int
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp | None
    entry_spot: float
    entry_perp: float
    exit_spot: float | None
    exit_perp: float | None
    entry_basis_bps: float
    exit_basis_bps: float | None
    qty_eth: float
    notional_usd: float | None = None
    entry_confidence: float | None = None
    funding_events: list[dict[str, Any]] = field(default_factory=list)
    funding_pnl: float = 0.0
    basis_pnl: float = 0.0
    cost_pnl: float = 0.0
    net_pnl: float = 0.0
    hold_minutes: int = 0
    exit_reason: str | None = None

    @property
    def funding_payments_collected(self) -> int:
        return len(self.funding_events)


@dataclass
class BacktestResult:
    trades: list[TradeRecord]
    trades_df: pd.DataFrame
    equity_curve: pd.DataFrame
