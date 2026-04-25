"""
Metrics utilities.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from src.backtest.models import BacktestResult


def compute_sharpe(returns: pd.Series, periods_per_year: int = 365 * 24 * 60) -> float:
    r = pd.to_numeric(returns, errors="coerce").dropna()
    if len(r) < 2:
        return float("nan")
    mu = float(r.mean())
    sigma = float(r.std(ddof=1))
    if sigma == 0.0:
        return float("nan")
    return (mu / sigma) * float(np.sqrt(periods_per_year))


def compute_max_drawdown(equity_curve: pd.Series) -> float:
    e = pd.to_numeric(equity_curve, errors="coerce").dropna()
    if e.empty:
        return float("nan")
    peak = e.cummax()
    dd = (e - peak) / peak
    return float(dd.min())


def compute_win_rate(trades_df: pd.DataFrame) -> float:
    if trades_df is None or trades_df.empty:
        return float("nan")
    wins = (pd.to_numeric(trades_df["net_pnl"], errors="coerce") > 0).sum()
    return float(wins) / float(len(trades_df))


def compute_avg_holding_period_hours(trades_df: pd.DataFrame) -> float:
    if trades_df is None or trades_df.empty:
        return float("nan")
    hold_min = pd.to_numeric(trades_df["hold_minutes"], errors="coerce").dropna()
    if hold_min.empty:
        return float("nan")
    return float(hold_min.mean() / 60.0)


def compute_turnover(trades_df: pd.DataFrame, capital: float) -> float:
    """
    Simple gross turnover approximation:
    - Each trade has two legs (spot + perp)
    - Entry + exit => 4 * notional
    - notional approximated as qty_eth * entry_spot
    """
    if trades_df is None or trades_df.empty:
        return 0.0
    entry_notional = pd.to_numeric(trades_df["qty_eth"], errors="coerce") * pd.to_numeric(
        trades_df["entry_spot"], errors="coerce"
    )
    gross = float((4.0 * entry_notional).sum())
    return gross / float(capital) if capital > 0 else float("nan")


def compute_summary_metrics(result: BacktestResult, capital: float) -> Dict[str, Any]:
    eq = result.equity_curve.copy()
    eq = eq.sort_values("timestamp_utc")
    equity = pd.to_numeric(eq["equity_usd"], errors="coerce")
    returns = equity.pct_change().fillna(0.0)

    trades_df = result.trades_df
    total_net_pnl = float(pd.to_numeric(trades_df["net_pnl"], errors="coerce").sum()) if not trades_df.empty else 0.0
    total_return = float((equity.iloc[-1] / equity.iloc[0] - 1.0)) if len(equity) >= 2 else 0.0

    metrics: Dict[str, Any] = {
        "start": eq["timestamp_utc"].min(),
        "end": eq["timestamp_utc"].max(),
        "minutes": int(len(eq)),
        "final_equity_usd": float(equity.iloc[-1]) if len(equity) else float("nan"),
        "total_net_pnl_usd": total_net_pnl,
        "total_return_pct": 100.0 * total_return,
        "sharpe_ann": compute_sharpe(returns),
        "max_drawdown_pct": 100.0 * compute_max_drawdown(equity),
        "trade_count": int(len(trades_df)),
        "win_rate": compute_win_rate(trades_df),
        "avg_hold_hours": compute_avg_holding_period_hours(trades_df),
        "turnover_x": compute_turnover(trades_df, capital=capital),
    }

    if not trades_df.empty:
        metrics["funding_pnl_usd"] = float(pd.to_numeric(trades_df["funding_pnl"], errors="coerce").sum())
        metrics["basis_pnl_usd"] = float(pd.to_numeric(trades_df["basis_pnl"], errors="coerce").sum())
        metrics["cost_pnl_usd"] = float(pd.to_numeric(trades_df["cost_pnl"], errors="coerce").sum())
    else:
        metrics["funding_pnl_usd"] = 0.0
        metrics["basis_pnl_usd"] = 0.0
        metrics["cost_pnl_usd"] = 0.0

    return metrics
