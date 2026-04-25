"""
Minute-level backtest engine for the ETH funding-harvest strategy (single venue).

Implements:
- long ETH spot + short ETH perp
- entry/exit rules from config
- funding accrual only at real funding timestamps (00:00/08:00/16:00 UTC)
- full P&L decomposition (funding, basis, costs)
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd

from src.backtest.costs import compute_entry_cost_usd, compute_exit_cost_usd
from src.backtest.ledger import trades_to_dataframe
from src.backtest.models import BacktestResult, TradeRecord, TradeState
from src.backtest.risk import RiskState, entries_allowed, maybe_update_pause
from src.signals.confidence import compute_confidence_score
from src.signals.edge import compute_round_trip_cost_bps
from src.signals.signal_engine import is_funding_timestamp, should_enter, should_exit
from src.utils.logging_utils import get_logger
from src.utils.validation import require_columns


logger = get_logger(__name__)


class FundingHarvestBacktester:
    def __init__(
        self,
        df: pd.DataFrame,
        strategy_cfg: Any,
        cost_cfg: Any,
        *,
        risk_cfg: Any | None = None,
        initial_capital: float = 100000.0,
    ):
        self.df = df.copy()
        self.strategy_cfg = strategy_cfg
        self.cost_cfg = cost_cfg
        self.risk_cfg = risk_cfg
        self.initial_capital = float(initial_capital)

        require_columns(
            self.df,
            [
                "timestamp_utc",
                "spot_close",
                "perp_close",
                "basis_bps",
                "funding_rate_last",
                "funding_rate_next",
                "minutes_to_next_funding",
                "expected_edge_bps",
            ],
        )
        self.df["timestamp_utc"] = pd.to_datetime(self.df["timestamp_utc"], utc=True)
        self.df = self.df.sort_values("timestamp_utc").reset_index(drop=True)

        self.state: TradeState = TradeState.FLAT
        self._cash_equity: float = self.initial_capital
        self._trade: TradeRecord | None = None
        self._trade_id: int = 0

        self._equity_rows: list[dict] = []
        self._trades: list[TradeRecord] = []

        self.risk_state = RiskState(peak_equity=self.initial_capital, pause_until=None)

    def _risk_value(self, key: str, default):
        if self.risk_cfg is None:
            return default
        if isinstance(self.risk_cfg, dict):
            return self.risk_cfg.get(key, default)
        return getattr(self.risk_cfg, key, default)

    def run(self) -> BacktestResult:
        for row in self.df.itertuples(index=False):
            ts: pd.Timestamp = row.timestamp_utc

            if self.state == TradeState.OPEN and self._trade is not None:
                self._accrue_funding(row)
                should_exit_flag, reason = self._should_exit(row)
                if should_exit_flag:
                    self._exit_trade(row, exit_reason=reason or "exit")
                self._mark_equity(row)
            else:
                if entries_allowed(self.risk_state, ts) and self._should_enter(row):
                    self._enter_trade(row)
                self._mark_equity(row)

            # Update pause logic after marking equity for this minute.
            current_equity = self._equity_rows[-1]["equity_usd"] if self._equity_rows else self._cash_equity
            self.risk_state = maybe_update_pause(
                state=self.risk_state,
                now=ts,
                equity_usd=float(current_equity),
                max_drawdown_pause_pct=float(self._risk_value("max_drawdown_pause_pct", 10.0)),
                pause_minutes=int(self._risk_value("pause_minutes", 240)),
            )

        trades_df = trades_to_dataframe(self._trades)
        equity_curve = pd.DataFrame(self._equity_rows)
        return BacktestResult(trades=self._trades, trades_df=trades_df, equity_curve=equity_curve)

    def _should_enter(self, row) -> bool:
        # Compute expected edge using the cost config used for execution/backtest.
        round_trip_cost_bps = compute_round_trip_cost_bps(self.cost_cfg)
        fr_next = float(row.funding_rate_next) if row.funding_rate_next is not None else 0.0

        class _R:
            pass

        r = _R()
        r.timestamp_utc = row.timestamp_utc
        r.basis_bps = float(row.basis_bps)
        r.funding_rate_next = row.funding_rate_next
        r.minutes_to_next_funding = float(row.minutes_to_next_funding)
        r.expected_edge_bps = float((fr_next * 10000.0) + float(row.basis_bps) - round_trip_cost_bps)
        return should_enter(self.state.value, r, self.strategy_cfg)

    def _enter_trade(self, row) -> None:
        self._trade_id += 1
        spot_px = float(row.spot_close)
        perp_px = float(row.perp_close)

        # Confidence score for logging/auditing (edge vs threshold).
        round_trip_cost_bps = compute_round_trip_cost_bps(self.cost_cfg)
        fr_next = float(row.funding_rate_next) if row.funding_rate_next is not None else 0.0
        expected_edge = (fr_next * 10000.0) + float(row.basis_bps) - round_trip_cost_bps
        entry_conf = compute_confidence_score(
            expected_edge_bps=float(expected_edge),
            entry_edge_bps=float(self.strategy_cfg.entry_edge_bps),
            minutes_to_next_funding=float(row.minutes_to_next_funding),
            min_minutes_to_next_funding=float(self.strategy_cfg.min_minutes_to_next_funding),
            scale_bps=float(self._risk_value("confidence_scale_bps", 10.0)),
        )

        position_fraction = float(self._risk_value("position_fraction", 1.0))
        notional_usd = float(self._cash_equity) * max(0.0, min(1.0, position_fraction))
        qty_eth = notional_usd / spot_px if spot_px > 0 else 0.0

        entry_cost = compute_entry_cost_usd(notional_usd, self.cost_cfg)
        self._cash_equity -= entry_cost

        self._trade = TradeRecord(
            trade_id=self._trade_id,
            entry_time=row.timestamp_utc,
            exit_time=None,
            entry_spot=spot_px,
            entry_perp=perp_px,
            exit_spot=None,
            exit_perp=None,
            entry_basis_bps=float(row.basis_bps),
            exit_basis_bps=None,
            qty_eth=float(qty_eth),
            notional_usd=float(notional_usd),
            entry_confidence=float(entry_conf),
            funding_events=[],
            funding_pnl=0.0,
            basis_pnl=0.0,
            cost_pnl=float(entry_cost),
            net_pnl=0.0,
            hold_minutes=0,
            exit_reason=None,
        )
        self.state = TradeState.OPEN

    def _accrue_funding(self, row) -> None:
        assert self._trade is not None
        ts: pd.Timestamp = row.timestamp_utc

        if not is_funding_timestamp(ts):
            return
        if not (self._trade.entry_time < ts):
            return

        notional_usd = float(self._trade.qty_eth) * float(self._trade.entry_spot)
        rate = float(row.funding_rate_last) if row.funding_rate_last is not None else 0.0
        pnl = notional_usd * rate
        self._trade.funding_pnl += pnl
        self._cash_equity += pnl
        self._trade.funding_events.append({"time": ts, "rate": rate, "pnl": pnl})

    def _should_exit(self, row) -> Tuple[bool, str | None]:
        assert self._trade is not None
        round_trip_cost_bps = compute_round_trip_cost_bps(self.cost_cfg)
        fr_next = float(row.funding_rate_next) if row.funding_rate_next is not None else 0.0

        class _R:
            pass

        r = _R()
        r.timestamp_utc = row.timestamp_utc
        r.basis_bps = float(row.basis_bps)
        r.expected_edge_bps = float((fr_next * 10000.0) + float(row.basis_bps) - round_trip_cost_bps)
        return should_exit(self._trade, r, self.strategy_cfg)

    def _exit_trade(self, row, exit_reason: str) -> None:
        assert self._trade is not None

        notional_usd = float(self._trade.qty_eth) * float(self._trade.entry_spot)
        exit_cost = compute_exit_cost_usd(notional_usd, self.cost_cfg)

        exit_spot = float(row.spot_close)
        exit_perp = float(row.perp_close)
        qty = float(self._trade.qty_eth)

        spot_leg_pnl = qty * (exit_spot - float(self._trade.entry_spot))
        perp_leg_pnl = qty * (float(self._trade.entry_perp) - exit_perp)
        basis_pnl = spot_leg_pnl + perp_leg_pnl

        self._trade.exit_time = row.timestamp_utc
        self._trade.exit_spot = exit_spot
        self._trade.exit_perp = exit_perp
        self._trade.exit_basis_bps = float(row.basis_bps)
        self._trade.exit_reason = exit_reason

        self._trade.basis_pnl = float(basis_pnl)
        self._trade.cost_pnl += float(exit_cost)
        self._trade.net_pnl = float(self._trade.funding_pnl + self._trade.basis_pnl - self._trade.cost_pnl)

        self._trade.hold_minutes = int((self._trade.exit_time - self._trade.entry_time).total_seconds() / 60.0)

        self._cash_equity += basis_pnl
        self._cash_equity -= exit_cost

        self._trades.append(self._trade)
        self._trade = None
        self.state = TradeState.FLAT

    def _mark_equity(self, row) -> None:
        ts = row.timestamp_utc
        if self.state == TradeState.OPEN and self._trade is not None:
            qty = float(self._trade.qty_eth)
            spot_px = float(row.spot_close)
            perp_px = float(row.perp_close)
            unreal_basis = qty * (spot_px - float(self._trade.entry_spot)) + qty * (
                float(self._trade.entry_perp) - perp_px
            )

            notional_usd = qty * float(self._trade.entry_spot)
            est_exit_cost = compute_exit_cost_usd(notional_usd, self.cost_cfg)
            equity = float(self._cash_equity + unreal_basis - est_exit_cost)
            state = TradeState.OPEN.value
        else:
            equity = float(self._cash_equity)
            state = TradeState.FLAT.value

        self._equity_rows.append({"timestamp_utc": ts, "equity_usd": equity, "state": state})
