"""
Historical "live log" simulator.

Purpose: generate a 48-hour (or any duration) minute-by-minute CSV log in the same schema
as the live runner, without waiting in real time. This is useful for producing a "live artifact"
within a limited wall-clock window.

Usage:
    python -m src.live.historical_sim --dataset data/processed/master_1m.parquet --start 2026-04-18T00:00:00Z --hours 48
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from config.settings import get_config
from src.backtest.costs import compute_entry_cost_usd, compute_exit_cost_usd
from src.live.logger import append_live_row
from src.signals.confidence import compute_confidence_score
from src.signals.edge import compute_round_trip_cost_bps
from src.signals.signal_engine import should_enter, should_exit
from src.utils.io_utils import ensure_dir, read_parquet
from src.utils.logging_utils import get_logger, setup_logging


logger = get_logger(__name__)


@dataclass
class SimTrade:
    entry_time: pd.Timestamp
    entry_spot: float
    entry_perp: float
    entry_basis_bps: float
    qty_eth: float
    funding_payments_collected: int = 0
    funding_pnl: float = 0.0
    cost_pnl: float = 0.0


def simulate_live_log(
    *,
    df: pd.DataFrame,
    strategy_cfg: Any,
    cost_cfg: Any,
    initial_capital: float,
    out_csv: Path,
) -> dict[str, Any]:
    app_cfg = get_config()
    instrument = app_cfg.symbol_perp
    scale_bps = float(app_cfg.risk.confidence_scale_bps) if getattr(app_cfg, "risk", None) else 10.0

    state = "FLAT"
    trade: SimTrade | None = None
    cash_equity = float(initial_capital)
    round_trip_cost_bps = compute_round_trip_cost_bps(cost_cfg)

    entry_count = 0
    exit_count = 0

    for row in df.itertuples(index=False):
        fr_next = float(row.funding_rate_next) if row.funding_rate_next is not None else 0.0
        expected_edge_bps = (fr_next * 10000.0) + float(row.basis_bps) - float(round_trip_cost_bps)

        class _R:
            pass

        r = _R()
        r.timestamp_utc = row.timestamp_utc
        r.spot_close = float(row.spot_close)
        r.perp_close = float(row.perp_close)
        r.basis_bps = float(row.basis_bps)
        r.funding_rate_last = row.funding_rate_last
        r.funding_rate_next = row.funding_rate_next
        r.next_funding_time = row.next_funding_time
        r.minutes_to_next_funding = float(row.minutes_to_next_funding)
        r.expected_edge_bps = float(expected_edge_bps)

        note_parts: list[str] = []
        entry_flag = False
        exit_flag = False

        # Funding accrual: only at real funding timestamps, using funding_rate_last.
        ts = row.timestamp_utc
        if state == "OPEN" and trade is not None:
            # Funding timestamps were already aligned in dataset; treat any non-null funding_rate_last at 00/08/16:00 as payable.
            if (
                ts.minute == 0
                and ts.hour in (0, 8, 16)
                and float(getattr(row, "funding_rate_last", 0.0) or 0.0) != 0.0
                and trade.entry_time < ts
            ):
                notional = trade.qty_eth * trade.entry_spot
                pnl = float(notional) * float(row.funding_rate_last)
                trade.funding_pnl += pnl
                cash_equity += pnl
                trade.funding_payments_collected += 1
                note_parts.append("funding_accrued")

            should_exit_flag, reason = should_exit(trade, r, strategy_cfg)
            if should_exit_flag:
                notional = trade.qty_eth * trade.entry_spot
                exit_cost = compute_exit_cost_usd(notional, cost_cfg)
                spot_leg = trade.qty_eth * (float(row.spot_close) - trade.entry_spot)
                perp_leg = trade.qty_eth * (trade.entry_perp - float(row.perp_close))
                basis_pnl = float(spot_leg + perp_leg)
                cash_equity += basis_pnl
                cash_equity -= exit_cost
                trade.cost_pnl += float(exit_cost)
                state = "FLAT"
                trade = None
                exit_flag = True
                exit_count += 1
                note_parts.append(f"exit:{reason}")

        signal_flag = should_enter("FLAT", r, strategy_cfg)
        confidence = compute_confidence_score(
            expected_edge_bps=float(expected_edge_bps),
            entry_edge_bps=float(strategy_cfg.entry_edge_bps),
            minutes_to_next_funding=float(row.minutes_to_next_funding),
            min_minutes_to_next_funding=float(strategy_cfg.min_minutes_to_next_funding),
            scale_bps=float(scale_bps),
        )
        if state == "FLAT" and should_enter("FLAT", r, strategy_cfg):
            qty = cash_equity / float(row.spot_close) if float(row.spot_close) > 0 else 0.0
            entry_cost = compute_entry_cost_usd(cash_equity, cost_cfg)
            cash_equity -= entry_cost
            trade = SimTrade(
                entry_time=row.timestamp_utc,
                entry_spot=float(row.spot_close),
                entry_perp=float(row.perp_close),
                entry_basis_bps=float(row.basis_bps),
                qty_eth=float(qty),
                funding_payments_collected=0,
                funding_pnl=0.0,
                cost_pnl=float(entry_cost),
            )
            state = "OPEN"
            entry_flag = True
            entry_count += 1
            note_parts.append("entry")

        out_row = {
            "timestamp_utc": row.timestamp_utc.isoformat(),
            "instrument": instrument,
            "direction": "LONG_SPOT_SHORT_PERP",
            "spot_price": float(row.spot_close),
            "perp_price": float(row.perp_close),
            "basis_bps": float(row.basis_bps),
            "funding_rate_last": float(row.funding_rate_last) if row.funding_rate_last is not None else None,
            "funding_rate_next": float(row.funding_rate_next) if row.funding_rate_next is not None else None,
            "next_funding_time": row.next_funding_time.isoformat()
            if getattr(row, "next_funding_time", None) is not None
            else None,
            "minutes_to_next_funding": float(row.minutes_to_next_funding),
            "expected_edge_bps": float(expected_edge_bps),
            "confidence_score": float(confidence),
            "state": state,
            "signal_flag": bool(signal_flag),
            "entry_flag": bool(entry_flag),
            "exit_flag": bool(exit_flag),
            "note": ";".join(note_parts) if note_parts else "",
        }
        append_live_row(out_csv, out_row)

    return {"entries": entry_count, "exits": exit_count, "rows": int(len(df)), "out_csv": str(out_csv)}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Simulate a live log from historical master dataset.")
    p.add_argument("--dataset", type=str, default="data/processed/master_1m.parquet")
    p.add_argument("--start", type=str, required=True, help="UTC start timestamp (e.g. 2026-04-18T00:00:00Z)")
    p.add_argument("--hours", type=int, default=48)
    p.add_argument("--scenario", type=str, default="baseline", choices=["baseline", "optimistic"])
    p.add_argument("--out", type=str, default=None, help="Output CSV path (default under outputs/live_logs).")
    p.add_argument("--capital", type=float, default=100000.0)
    p.add_argument("--entry-edge-bps", type=float, default=None, help="Override strategy.entry_edge_bps")
    p.add_argument("--exit-edge-bps", type=float, default=None, help="Override strategy.exit_edge_bps")
    p.add_argument("--max-holding-hours", type=int, default=None, help="Override strategy.max_holding_hours")
    return p.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()
    cfg = get_config()

    dataset = read_parquet(args.dataset)
    dataset["timestamp_utc"] = pd.to_datetime(dataset["timestamp_utc"], utc=True)

    start = pd.Timestamp(args.start)
    if start.tzinfo is None:
        start = start.tz_localize("UTC")
    else:
        start = start.tz_convert("UTC")
    end = start + pd.Timedelta(hours=int(args.hours))

    window = dataset[(dataset["timestamp_utc"] >= start) & (dataset["timestamp_utc"] < end)].copy()
    if window.empty:
        raise SystemExit(f"No rows in dataset for window {start} -> {end}.")

    out = Path(args.out) if args.out else Path("outputs/live_logs") / f"live_log_sim_{start.strftime('%Y%m%dT%H%M')}_{int(args.hours)}h.csv"
    ensure_dir(out.parent)
    if out.exists():
        out.unlink()

    cost_cfg = getattr(cfg.costs, args.scenario)
    strategy_cfg = cfg.strategy
    overrides = {}
    if args.entry_edge_bps is not None:
        overrides["entry_edge_bps"] = float(args.entry_edge_bps)
    if args.exit_edge_bps is not None:
        overrides["exit_edge_bps"] = float(args.exit_edge_bps)
    if args.max_holding_hours is not None:
        overrides["max_holding_hours"] = int(args.max_holding_hours)
    if overrides and hasattr(strategy_cfg, "model_copy"):
        strategy_cfg = strategy_cfg.model_copy(update=overrides)

    summary = simulate_live_log(
        df=window,
        strategy_cfg=strategy_cfg,
        cost_cfg=cost_cfg,
        initial_capital=float(args.capital),
        out_csv=out,
    )
    logger.info("Simulated live log: %s", summary)


if __name__ == "__main__":
    main()
