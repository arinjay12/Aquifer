"""
Plotting utilities.

Writes PNG plots to `outputs/figures/`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config.settings import get_config
from src.backtest.engine import FundingHarvestBacktester
from src.utils.io_utils import ensure_dir, read_parquet
from src.utils.logging_utils import get_logger, setup_logging


logger = get_logger(__name__)


def _save_fig(path: Path) -> None:
    ensure_dir(path.parent)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_funding_history(df: pd.DataFrame, out_path: Path) -> None:
    s = pd.to_numeric(df["funding_rate_next"], errors="coerce") * 10000.0
    plt.figure(figsize=(10, 4))
    plt.plot(df["timestamp_utc"], s, linewidth=0.8)
    plt.title("Funding Rate (next, bps)")
    plt.xlabel("Time (UTC)")
    plt.ylabel("Funding (bps)")
    _save_fig(out_path)


def plot_basis_history(df: pd.DataFrame, out_path: Path) -> None:
    s = pd.to_numeric(df["basis_bps"], errors="coerce")
    plt.figure(figsize=(10, 4))
    plt.plot(df["timestamp_utc"], s, linewidth=0.8)
    plt.title("Basis (perp - spot, bps)")
    plt.xlabel("Time (UTC)")
    plt.ylabel("Basis (bps)")
    _save_fig(out_path)


def plot_equity_curve(equity_df: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(10, 4))
    plt.plot(equity_df["timestamp_utc"], equity_df["equity_usd"], linewidth=1.0)
    plt.title("Equity Curve (USD)")
    plt.xlabel("Time (UTC)")
    plt.ylabel("Equity (USD)")
    _save_fig(out_path)


def plot_drawdown_curve(equity_df: pd.DataFrame, out_path: Path) -> None:
    eq = pd.to_numeric(equity_df["equity_usd"], errors="coerce")
    peak = eq.cummax()
    dd = (eq - peak) / peak
    plt.figure(figsize=(10, 3))
    plt.plot(equity_df["timestamp_utc"], dd * 100.0, linewidth=1.0)
    plt.title("Drawdown (%)")
    plt.xlabel("Time (UTC)")
    plt.ylabel("Drawdown (%)")
    _save_fig(out_path)


def plot_trade_pnl_hist(trades_df: pd.DataFrame, out_path: Path) -> None:
    if trades_df.empty:
        return
    pnl = pd.to_numeric(trades_df["net_pnl"], errors="coerce").dropna()
    plt.figure(figsize=(6, 4))
    plt.hist(pnl, bins=30)
    plt.title("Trade Net P&L Histogram (USD)")
    plt.xlabel("Net P&L (USD)")
    plt.ylabel("Count")
    _save_fig(out_path)


def plot_pnl_decomposition(trades_df: pd.DataFrame, out_path: Path) -> None:
    funding = float(pd.to_numeric(trades_df.get("funding_pnl", 0.0), errors="coerce").sum()) if not trades_df.empty else 0.0
    basis = float(pd.to_numeric(trades_df.get("basis_pnl", 0.0), errors="coerce").sum()) if not trades_df.empty else 0.0
    cost = float(pd.to_numeric(trades_df.get("cost_pnl", 0.0), errors="coerce").sum()) if not trades_df.empty else 0.0
    net = float(pd.to_numeric(trades_df.get("net_pnl", 0.0), errors="coerce").sum()) if not trades_df.empty else 0.0

    labels = ["Funding", "Basis", "Costs", "Net"]
    values = [funding, basis, -cost, net]
    colors = ["#4c78a8", "#f58518", "#e45756", "#54a24b"]
    plt.figure(figsize=(6, 4))
    plt.bar(labels, values, color=colors)
    plt.title("P&L Decomposition (USD)")
    plt.ylabel("USD")
    _save_fig(out_path)


def plot_threshold_vs_sharpe(sweep_df: pd.DataFrame, out_path: Path) -> None:
    if sweep_df.empty:
        return
    df = sweep_df.copy()
    df = df.sort_values(["entry_edge_bps", "exit_edge_bps", "max_holding_hours"])
    plt.figure(figsize=(7, 4))
    plt.plot(df["entry_edge_bps"], df["sharpe_ann"], marker="o", linestyle="-", linewidth=1.0)
    plt.title("Entry Threshold vs Sharpe (in-sample)")
    plt.xlabel("entry_edge_bps")
    plt.ylabel("Sharpe (ann.)")
    _save_fig(out_path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate standard figures from a processed master dataset.")
    p.add_argument("--dataset", type=str, default="data/processed/master_1m.parquet")
    p.add_argument("--figures-dir", type=str, default="outputs/figures")
    p.add_argument("--scenario", type=str, default="baseline", choices=["baseline", "optimistic"])
    p.add_argument("--sweep-csv", type=str, default="outputs/tables/threshold_sweep.csv")
    p.add_argument("--entry-edge-bps", type=float, default=None)
    p.add_argument("--exit-edge-bps", type=float, default=None)
    p.add_argument("--max-holding-hours", type=int, default=None)
    p.add_argument("--min-basis-bps", type=float, default=None)
    return p.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()
    cfg = get_config()

    dataset = read_parquet(args.dataset)
    figures_dir = Path(args.figures_dir)
    ensure_dir(figures_dir)

    plot_funding_history(dataset, figures_dir / "funding_history_next_bps.png")
    plot_basis_history(dataset, figures_dir / "basis_history_bps.png")

    overrides = {}
    if args.entry_edge_bps is not None:
        overrides["entry_edge_bps"] = float(args.entry_edge_bps)
    if args.exit_edge_bps is not None:
        overrides["exit_edge_bps"] = float(args.exit_edge_bps)
    if args.max_holding_hours is not None:
        overrides["max_holding_hours"] = int(args.max_holding_hours)
    if args.min_basis_bps is not None:
        overrides["min_basis_bps"] = float(args.min_basis_bps)
    strategy = cfg.strategy.model_copy(update=overrides) if overrides and hasattr(cfg.strategy, "model_copy") else cfg.strategy

    cost_cfg = getattr(cfg.costs, args.scenario)
    bt = FundingHarvestBacktester(dataset, strategy, cost_cfg, risk_cfg=cfg.risk, initial_capital=100000.0)
    res = bt.run()
    plot_equity_curve(res.equity_curve, figures_dir / f"equity_curve_{args.scenario}.png")
    plot_drawdown_curve(res.equity_curve, figures_dir / f"drawdown_{args.scenario}.png")
    plot_trade_pnl_hist(res.trades_df, figures_dir / f"trade_pnl_hist_{args.scenario}.png")
    plot_pnl_decomposition(res.trades_df, figures_dir / f"pnl_decomposition_{args.scenario}.png")

    sweep_path = Path(args.sweep_csv)
    if sweep_path.exists():
        sweep_df = pd.read_csv(sweep_path)
        plot_threshold_vs_sharpe(sweep_df, figures_dir / "threshold_vs_sharpe.png")

    logger.info("Wrote figures to %s", figures_dir)


if __name__ == "__main__":
    main()
