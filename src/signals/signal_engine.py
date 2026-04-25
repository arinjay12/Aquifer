"""
Signal orchestration shared by backtest and live runner.

Contains the pure rule checks for entry/exit, plus small helpers.
"""

from __future__ import annotations

from typing import Any, Tuple

import numpy as np
import pandas as pd


def is_funding_timestamp(ts: pd.Timestamp) -> bool:
    """
    Binance funding is (typically) every 8 hours at 00:00/08:00/16:00 UTC.
    """
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.minute == 0 and ts.second == 0 and ts.microsecond == 0 and ts.hour in (0, 8, 16)


def should_enter(state: str, row: Any, strategy_cfg: Any) -> bool:
    """
    Entry rule (v1):
    - state == FLAT
    - expected_edge_bps >= entry_edge_bps
    - basis_bps >= min_basis_bps
    - funding_rate_next > 0
    - minutes_to_next_funding >= min_minutes_to_next_funding
    """
    if state != "FLAT":
        return False
    if not np.isfinite(float(row.expected_edge_bps)) or not np.isfinite(float(row.basis_bps)):
        return False
    if float(row.expected_edge_bps) < float(strategy_cfg.entry_edge_bps):
        return False
    if float(row.basis_bps) < float(strategy_cfg.min_basis_bps):
        return False
    if not (row.funding_rate_next is not None and float(row.funding_rate_next) > 0.0):
        return False
    if float(row.minutes_to_next_funding) < float(strategy_cfg.min_minutes_to_next_funding):
        return False
    return True


def should_exit(trade: Any, row: Any, strategy_cfg: Any) -> Tuple[bool, str | None]:
    """
    Exit rules in priority order:
    1) Hard stop: basis widened vs entry by hard_stop_basis_widen_bps
    2) Profit/convergence: basis <= exit_basis_bps AND funding_payments_collected >= 1
    3) Funding decay: expected_edge_bps <= exit_edge_bps
    4) Time stop: holding_time_hours >= max_holding_hours
    """
    # 1) Hard stop
    if float(row.basis_bps) >= float(trade.entry_basis_bps) + float(strategy_cfg.hard_stop_basis_widen_bps):
        return True, "hard_stop"

    # 2) Profit / convergence
    if int(getattr(trade, "funding_payments_collected", 0)) >= 1 and float(row.basis_bps) <= float(
        strategy_cfg.exit_basis_bps
    ):
        return True, "profit_convergence"

    # 3) Funding decay
    if float(row.expected_edge_bps) <= float(strategy_cfg.exit_edge_bps):
        return True, "funding_decay"

    # 4) Time stop
    hold_minutes = int((row.timestamp_utc - trade.entry_time).total_seconds() / 60.0)
    hold_hours = hold_minutes / 60.0
    if hold_hours >= float(strategy_cfg.max_holding_hours):
        return True, "time_stop"

    return False, None
