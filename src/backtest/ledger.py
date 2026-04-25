"""
Trade ledger helpers.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.backtest.models import TradeRecord
from src.utils.io_utils import ensure_dir


def trades_to_dataframe(trades: list[TradeRecord]) -> pd.DataFrame:
    rows: list[dict] = []
    for t in trades:
        rows.append(
            {
                "trade_id": t.trade_id,
                "entry_time": t.entry_time,
                "exit_time": t.exit_time,
                "entry_spot": t.entry_spot,
                "entry_perp": t.entry_perp,
                "exit_spot": t.exit_spot,
                "exit_perp": t.exit_perp,
                "entry_basis_bps": t.entry_basis_bps,
                "exit_basis_bps": t.exit_basis_bps,
                "qty_eth": t.qty_eth,
                "notional_usd": t.notional_usd,
                "entry_confidence": t.entry_confidence,
                "funding_events": t.funding_payments_collected,
                "funding_pnl": t.funding_pnl,
                "basis_pnl": t.basis_pnl,
                "cost_pnl": t.cost_pnl,
                "net_pnl": t.net_pnl,
                "hold_minutes": t.hold_minutes,
                "exit_reason": t.exit_reason,
            }
        )
    return pd.DataFrame(rows)


def save_trade_ledger(trades_df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    trades_df.to_csv(path, index=False)
