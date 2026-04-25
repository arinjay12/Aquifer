"""
Live signal runner.

Polls Binance once per minute (configurable), computes the same signal logic as the backtest,
logs every observation to CSV, and records simulated entry/exit events.

Restart recovery:
- Reads the existing live log (if present)
- Restores OPEN/FLAT state from the last recorded row
- Reconstructs entry time and funding events since entry from the log
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from collections import deque

import pandas as pd

from config.settings import AppConfig, get_config
from src.backtest.costs import compute_entry_cost_usd, compute_exit_cost_usd
from src.data.binance_client import BinanceClient
from src.data.live_fetch import fetch_live_snapshot
from src.live.logger import append_live_row, read_last_rows
from src.signals.basis import compute_basis_bps
from src.signals.confidence import compute_confidence_score
from src.signals.edge import compute_expected_edge_bps, compute_round_trip_cost_bps
from src.signals.funding import compute_minutes_to_next_funding
from src.signals.signal_engine import is_funding_timestamp, should_enter, should_exit
from src.utils.logging_utils import get_logger, setup_logging


logger = get_logger(__name__)


@dataclass
class LiveTrade:
    entry_time: pd.Timestamp
    entry_spot: float
    entry_perp: float
    entry_basis_bps: float
    qty_eth: float
    funding_payments_collected: int = 0
    funding_pnl: float = 0.0
    cost_pnl: float = 0.0


class LiveSignalRunner:
    def __init__(
        self,
        config: AppConfig | None = None,
        *,
        cost_scenario: str = "baseline",
        initial_capital: float = 100000.0,
    ):
        self.cfg = config or get_config()
        self.cost_cfg = getattr(self.cfg.costs, cost_scenario)
        self.initial_capital = float(initial_capital)

        self.client = BinanceClient()

        self.state: str = "FLAT"
        self.trade: LiveTrade | None = None
        self.cash_equity: float = self.initial_capital

        self.output_csv = Path(self.cfg.live.output_csv)
        self.round_trip_cost_bps = compute_round_trip_cost_bps(self.cost_cfg)

        self._last_ts: pd.Timestamp | None = None
        self._recent_edges = deque(maxlen=1440)  # ~24h at 1-minute cadence
        self._recover_state()

    def _recover_state(self) -> None:
        df = read_last_rows(self.output_csv, n=5000)
        if df.empty:
            logger.info("No existing live log found at %s. Starting fresh.", self.output_csv)
            return

        last = df.tail(1).iloc[0].to_dict()
        try:
            self._last_ts = pd.to_datetime(last.get("timestamp_utc"), utc=True, errors="coerce")
        except Exception:  # noqa: BLE001
            self._last_ts = None
        last_state = str(last.get("state", "FLAT"))
        self.state = "OPEN" if last_state == "OPEN" else "FLAT"

        if self.state == "OPEN":
            # Reconstruct entry as the most recent entry_flag==True row after the most recent exit_flag==True.
            df2 = df.copy()
            for c in ["entry_flag", "exit_flag"]:
                if c in df2.columns:
                    df2[c] = df2[c].astype(str).str.lower().isin(["true", "1", "yes"])

            last_exit_idx = df2.index[df2.get("exit_flag", False) == True].max() if "exit_flag" in df2.columns else None
            after = df2 if pd.isna(last_exit_idx) else df2.loc[last_exit_idx + 1 :]
            entries = after[after.get("entry_flag", False) == True] if "entry_flag" in after.columns else pd.DataFrame()
            if entries.empty:
                logger.warning("Last state OPEN but could not find entry row in log; starting FLAT.")
                self.state = "FLAT"
                self.trade = None
                return

            e = entries.tail(1).iloc[0]
            entry_time = pd.to_datetime(e["timestamp_utc"], utc=True, errors="coerce")
            entry_spot = float(e["spot_price"])
            entry_perp = float(e["perp_price"])
            entry_basis = float(e["basis_bps"])
            qty = self.initial_capital / entry_spot if entry_spot > 0 else 0.0
            entry_cost = compute_entry_cost_usd(self.initial_capital, self.cost_cfg)

            # Count funding accrual notes after entry
            post = after[after["timestamp_utc"] >= entry_time]
            funding_mask = post["note"].astype(str).str.contains("funding_accrued", na=False)
            funding_events = int(funding_mask.sum())
            notional = qty * entry_spot
            funding_pnl = float((pd.to_numeric(post.loc[funding_mask, "funding_rate_last"], errors="coerce").fillna(0.0) * notional).sum())
            self.trade = LiveTrade(
                entry_time=entry_time,
                entry_spot=entry_spot,
                entry_perp=entry_perp,
                entry_basis_bps=entry_basis,
                qty_eth=float(qty),
                funding_payments_collected=funding_events,
                funding_pnl=float(funding_pnl),
                cost_pnl=float(entry_cost),
            )
            self.cash_equity = self.initial_capital - entry_cost
            logger.info("Recovered OPEN state from log. Entry at %s.", entry_time)

    def _sleep_to_next_tick(self) -> None:
        poll_s = int(self.cfg.live.poll_seconds)
        now = time.time()
        next_tick = (int(now // poll_s) + 1) * poll_s
        sleep_s = max(0.1, next_tick - now)
        time.sleep(sleep_s)

    def _accrue_funding_if_due(self, ts: pd.Timestamp, funding_rate_last: float | None) -> str | None:
        if self.state != "OPEN" or self.trade is None:
            return None
        if funding_rate_last is None:
            return None

        ts_min = ts.floor("min")
        last_min = self._last_ts.floor("min") if self._last_ts is not None else None
        if last_min is not None and ts_min == last_min:
            return None
        if not is_funding_timestamp(ts_min):
            return None

        if not (self.trade.entry_time < ts_min):
            return None

        notional = self.trade.qty_eth * self.trade.entry_spot
        pnl = float(notional) * float(funding_rate_last)
        self.trade.funding_pnl += pnl
        self.cash_equity += pnl
        self.trade.funding_payments_collected += 1
        return "funding_accrued"

    def run(self, *, run_minutes: int | None = None) -> None:
        setup_logging()
        logger.info("LiveSignalRunner starting. Output CSV: %s", self.output_csv)

        start_ts = time.time()
        while True:
            snap = fetch_live_snapshot(self.client, self.cfg.symbol_spot, self.cfg.symbol_perp)
            ts = pd.Timestamp(snap.timestamp_utc)
            if ts.tzinfo is None:
                ts = ts.tz_localize("UTC")
            else:
                ts = ts.tz_convert("UTC")

            basis_bps = float(compute_basis_bps(snap.spot_price, snap.perp_price))
            minutes_to_next = (
                float(compute_minutes_to_next_funding(ts, snap.next_funding_time))
                if snap.next_funding_time is not None
                else float("nan")
            )
            expected_edge = float(
                compute_expected_edge_bps(
                    funding_rate_next=float(snap.funding_rate_next or 0.0),
                    basis_bps=basis_bps,
                    round_trip_cost_bps=self.round_trip_cost_bps,
                )
            )
            confidence = compute_confidence_score(
                expected_edge_bps=float(expected_edge),
                entry_edge_bps=float(self.cfg.strategy.entry_edge_bps),
                minutes_to_next_funding=float(minutes_to_next),
                min_minutes_to_next_funding=float(self.cfg.strategy.min_minutes_to_next_funding),
                scale_bps=float(getattr(self.cfg, "risk").confidence_scale_bps) if getattr(self.cfg, "risk", None) else 10.0,
            )

            self._recent_edges.append(expected_edge)
            if len(self._recent_edges) >= 600:
                try:
                    import numpy as np

                    p95 = float(np.nanpercentile(list(self._recent_edges), 95))
                    if p95 < float(self.cfg.strategy.entry_edge_bps):
                        note_parts.append("edge_regime_low")
                except Exception:
                    pass

            # Build a row-like object for shared rule functions.
            class _Row:
                pass

            r = _Row()
            r.timestamp_utc = ts
            r.spot_close = float(snap.spot_price)
            r.perp_close = float(snap.perp_price)
            r.basis_bps = float(basis_bps)
            r.funding_rate_last = snap.funding_rate_last
            r.funding_rate_next = snap.funding_rate_next
            r.minutes_to_next_funding = float(minutes_to_next)
            r.expected_edge_bps = float(expected_edge)

            note_parts: list[str] = []
            entry_flag = False
            exit_flag = False
            signal_flag = should_enter("FLAT", r, self.cfg.strategy)

            funding_note = self._accrue_funding_if_due(ts, snap.funding_rate_last)
            if funding_note:
                note_parts.append(funding_note)

            if self.state == "OPEN" and self.trade is not None:
                should_exit_flag, reason = should_exit(self.trade, r, self.cfg.strategy)
                if should_exit_flag:
                    # Realize basis pnl on exit and pay exit costs.
                    notional = self.trade.qty_eth * self.trade.entry_spot
                    exit_cost = compute_exit_cost_usd(notional, self.cost_cfg)
                    spot_leg = self.trade.qty_eth * (float(snap.spot_price) - self.trade.entry_spot)
                    perp_leg = self.trade.qty_eth * (self.trade.entry_perp - float(snap.perp_price))
                    basis_pnl = float(spot_leg + perp_leg)

                    self.cash_equity += basis_pnl
                    self.cash_equity -= exit_cost
                    self.trade.cost_pnl += float(exit_cost)

                    exit_flag = True
                    note_parts.append(f"exit:{reason}")
                    self.trade = None
                    self.state = "FLAT"

            if self.state == "FLAT":
                if should_enter("FLAT", r, self.cfg.strategy):
                    qty = self.cash_equity / float(snap.spot_price) if float(snap.spot_price) > 0 else 0.0
                    entry_cost = compute_entry_cost_usd(self.cash_equity, self.cost_cfg)
                    self.cash_equity -= entry_cost
                    self.trade = LiveTrade(
                        entry_time=ts,
                        entry_spot=float(snap.spot_price),
                        entry_perp=float(snap.perp_price),
                        entry_basis_bps=float(basis_bps),
                        qty_eth=float(qty),
                        funding_payments_collected=0,
                        funding_pnl=0.0,
                        cost_pnl=float(entry_cost),
                    )
                    self.state = "OPEN"
                    entry_flag = True
                    note_parts.append("entry")

            row = {
                "timestamp_utc": ts.isoformat(),
                "instrument": self.cfg.symbol_perp,
                "direction": "LONG_SPOT_SHORT_PERP",
                "spot_price": float(snap.spot_price),
                "perp_price": float(snap.perp_price),
                "basis_bps": float(basis_bps),
                "funding_rate_last": snap.funding_rate_last,
                "funding_rate_next": snap.funding_rate_next,
                "next_funding_time": snap.next_funding_time.isoformat() if snap.next_funding_time is not None else None,
                "minutes_to_next_funding": float(minutes_to_next),
                "expected_edge_bps": float(expected_edge),
                "confidence_score": float(confidence),
                "state": self.state,
                "signal_flag": bool(signal_flag),
                "entry_flag": bool(entry_flag),
                "exit_flag": bool(exit_flag),
                "note": ";".join(note_parts) if note_parts else "",
            }
            append_live_row(self.output_csv, row)
            self._last_ts = ts

            if run_minutes is not None and (time.time() - start_ts) >= (60.0 * float(run_minutes)):
                logger.info("Stopping after %s minutes.", run_minutes)
                break
            self._sleep_to_next_tick()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the live polling signal runner.")
    p.add_argument("--scenario", type=str, default="baseline", choices=["baseline", "optimistic"])
    p.add_argument("--capital", type=float, default=100000.0)
    p.add_argument("--run-minutes", type=int, default=None, help="If set, stop after this many minutes.")
    p.add_argument("--poll-seconds", type=int, default=None, help="Override live.poll_seconds from params.yaml.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = get_config()
    if args.poll_seconds is not None:
        cfg.live.poll_seconds = int(args.poll_seconds)  # type: ignore[misc]
    runner = LiveSignalRunner(cfg, cost_scenario=args.scenario, initial_capital=float(args.capital))
    runner.run(run_minutes=args.run_minutes)


if __name__ == "__main__":
    main()
