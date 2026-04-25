"""
Risk framework utilities.

Implements:
- position sizing (fraction of capital, optionally modulated by confidence)
- drawdown-based pause logic (stop entering new trades for a period)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd


@dataclass
class RiskState:
    peak_equity: float
    pause_until: pd.Timestamp | None = None


def position_notional_usd(
    *,
    equity_usd: float,
    position_fraction: float,
    confidence: float = 1.0,
) -> float:
    frac = max(0.0, min(1.0, float(position_fraction)))
    conf = max(0.0, min(1.0, float(confidence)))
    return float(equity_usd) * frac * conf


def maybe_update_pause(
    *,
    state: RiskState,
    now: pd.Timestamp,
    equity_usd: float,
    max_drawdown_pause_pct: float,
    pause_minutes: int,
) -> RiskState:
    peak = max(float(state.peak_equity), float(equity_usd))
    dd = (float(equity_usd) / peak) - 1.0 if peak > 0 else 0.0
    dd_pct = 100.0 * dd

    pause_until = state.pause_until
    if pause_until is not None and now >= pause_until:
        pause_until = None

    if pause_until is None and dd_pct <= -abs(float(max_drawdown_pause_pct)):
        pause_until = now + pd.Timedelta(minutes=int(pause_minutes))

    return RiskState(peak_equity=peak, pause_until=pause_until)


def entries_allowed(state: RiskState, now: pd.Timestamp) -> bool:
    return state.pause_until is None or now >= state.pause_until

