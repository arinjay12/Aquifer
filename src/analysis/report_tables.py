"""
Report table builders.

Produces:
- summary metrics table
- quarterly metrics table
- exit reason breakdown
- funding vs basis vs costs contribution

Writes CSVs under `outputs/tables/` and (optionally) ledgers under `outputs/ledgers/`.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from config.settings import get_config
from src.backtest.engine import FundingHarvestBacktester
from src.backtest.ledger import save_trade_ledger
from src.backtest.metrics import compute_summary_metrics
from src.utils.io_utils import ensure_dir, read_parquet
from src.utils.logging_utils import get_logger, setup_logging


logger = get_logger(__name__)


def _quarterly_trade_table(trades_df: pd.DataFrame) -> pd.DataFrame:
    if trades_df.empty:
        return pd.DataFrame()
    df = trades_df.copy()
    df["exit_time"] = pd.to_datetime(df["exit_time"], utc=True, errors="coerce")
    df = df.dropna(subset=["exit_time"])
    df["quarter"] = df["exit_time"].dt.to_period("Q").astype(str)
    agg = df.groupby("quarter").agg(
        trade_count=("trade_id", "count"),
        net_pnl_usd=("net_pnl", "sum"),
        funding_pnl_usd=("funding_pnl", "sum"),
        basis_pnl_usd=("basis_pnl", "sum"),
        cost_pnl_usd=("cost_pnl", "sum"),
        win_rate=("net_pnl", lambda s: float((pd.to_numeric(s, errors="coerce") > 0).mean())),
        avg_hold_hours=("hold_minutes", lambda s: float(pd.to_numeric(s, errors="coerce").mean() / 60.0)),
    )
    return agg.reset_index()


def _exit_reason_table(trades_df: pd.DataFrame) -> pd.DataFrame:
    if trades_df.empty:
        return pd.DataFrame()
    df = trades_df.copy()
    df["exit_reason"] = df["exit_reason"].fillna("unknown")
    out = (
        df.groupby("exit_reason")
        .agg(trade_count=("trade_id", "count"), net_pnl_usd=("net_pnl", "sum"))
        .sort_values("trade_count", ascending=False)
        .reset_index()
    )
    return out


def _pnl_decomposition_table(trades_df: pd.DataFrame) -> pd.DataFrame:
    if trades_df.empty:
        return pd.DataFrame(
            [
                {"bucket": "funding_pnl_usd", "value_usd": 0.0},
                {"bucket": "basis_pnl_usd", "value_usd": 0.0},
                {"bucket": "cost_pnl_usd", "value_usd": 0.0},
                {"bucket": "net_pnl_usd", "value_usd": 0.0},
            ]
        )
    funding = float(pd.to_numeric(trades_df["funding_pnl"], errors="coerce").sum())
    basis = float(pd.to_numeric(trades_df["basis_pnl"], errors="coerce").sum())
    cost = float(pd.to_numeric(trades_df["cost_pnl"], errors="coerce").sum())
    net = float(pd.to_numeric(trades_df["net_pnl"], errors="coerce").sum())
    return pd.DataFrame(
        [
            {"bucket": "funding_pnl_usd", "value_usd": funding},
            {"bucket": "basis_pnl_usd", "value_usd": basis},
            {"bucket": "cost_pnl_usd", "value_usd": cost},
            {"bucket": "net_pnl_usd", "value_usd": net},
        ]
    )


def run_backtest(
    df: pd.DataFrame, strategy_cfg: Any, cost_cfg: Any, risk_cfg: Any, initial_capital: float
) -> tuple[dict, pd.DataFrame]:
    bt = FundingHarvestBacktester(
        df=df,
        strategy_cfg=strategy_cfg,
        cost_cfg=cost_cfg,
        risk_cfg=risk_cfg,
        initial_capital=initial_capital,
    )
    res = bt.run()
    metrics = compute_summary_metrics(res, capital=initial_capital)
    trades_df = res.trades_df
    return metrics, trades_df


def build_reports(
    *,
    dataset_path: Path,
    out_tables_dir: Path,
    out_ledgers_dir: Path,
    initial_capital: float = 100000.0,
) -> None:
    cfg = get_config()
    df = read_parquet(dataset_path)

    ensure_dir(out_tables_dir)
    ensure_dir(out_ledgers_dir)

    rows = []
    for scenario_name in ["baseline", "optimistic"]:
        cost_cfg = getattr(cfg.costs, scenario_name)
        metrics, trades_df = run_backtest(df, cfg.strategy, cost_cfg, cfg.risk, initial_capital=initial_capital)
        metrics["scenario"] = scenario_name
        rows.append(metrics)

        save_trade_ledger(trades_df, out_ledgers_dir / f"trades_{scenario_name}.csv")
        _quarterly_trade_table(trades_df).to_csv(out_tables_dir / f"quarterly_{scenario_name}.csv", index=False)
        _exit_reason_table(trades_df).to_csv(out_tables_dir / f"exit_reasons_{scenario_name}.csv", index=False)
        _pnl_decomposition_table(trades_df).to_csv(out_tables_dir / f"pnl_decomposition_{scenario_name}.csv", index=False)

    summary_df = pd.DataFrame(rows).set_index("scenario").reset_index()
    summary_df.to_csv(out_tables_dir / "summary_metrics.csv", index=False)
    logger.info("Wrote report tables to %s", out_tables_dir)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate metrics/report tables from a processed master dataset.")
    p.add_argument("--dataset", type=str, default="data/processed/master_1m.parquet")
    p.add_argument("--tables-dir", type=str, default="outputs/tables")
    p.add_argument("--ledgers-dir", type=str, default="outputs/ledgers")
    p.add_argument("--capital", type=float, default=100000.0)
    p.add_argument("--entry-edge-bps", type=float, default=None)
    p.add_argument("--exit-edge-bps", type=float, default=None)
    p.add_argument("--max-holding-hours", type=int, default=None)
    p.add_argument("--min-basis-bps", type=float, default=None)
    return p.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()
    cfg = get_config()
    overrides = {}
    if args.entry_edge_bps is not None:
        overrides["entry_edge_bps"] = float(args.entry_edge_bps)
    if args.exit_edge_bps is not None:
        overrides["exit_edge_bps"] = float(args.exit_edge_bps)
    if args.max_holding_hours is not None:
        overrides["max_holding_hours"] = int(args.max_holding_hours)
    if args.min_basis_bps is not None:
        overrides["min_basis_bps"] = float(args.min_basis_bps)
    if overrides and hasattr(cfg.strategy, "model_copy"):
        cfg.strategy = cfg.strategy.model_copy(update=overrides)  # type: ignore[misc]

    build_reports(
        dataset_path=Path(args.dataset),
        out_tables_dir=Path(args.tables_dir),
        out_ledgers_dir=Path(args.ledgers_dir),
        initial_capital=float(args.capital),
    )


if __name__ == "__main__":
    main()
