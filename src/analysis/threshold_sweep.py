"""
Threshold sweep utilities.

Sweeps:
- entry_edge_bps: [10, 15, 20, 25, 30, 40, 50]
- exit_edge_bps: [0, 2, 5]
- max_holding_hours: [12, 24, 48, 72]

Saves results to CSV and produces simple plots.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from config.settings import get_config
from src.backtest.engine import FundingHarvestBacktester
from src.backtest.metrics import compute_summary_metrics
from src.utils.io_utils import ensure_dir, read_parquet
from src.utils.logging_utils import get_logger, setup_logging


logger = get_logger(__name__)


DEFAULT_ENTRY_SWEEP = [10, 15, 20, 25, 30, 40, 50]
DEFAULT_EXIT_SWEEP = [0, 2, 5]
DEFAULT_HOLD_SWEEP = [12, 24, 48, 72]


def run_threshold_sweep(
    df: pd.DataFrame,
    strategy_cfg: Any,
    cost_cfg: Any,
    *,
    risk_cfg: Any | None = None,
    entry_edges: list[float] | None = None,
    exit_edges: list[float] | None = None,
    max_holds: list[int] | None = None,
    initial_capital: float = 100000.0,
) -> pd.DataFrame:
    rows: list[dict] = []
    entry_edges = entry_edges or [float(x) for x in DEFAULT_ENTRY_SWEEP]
    exit_edges = exit_edges or [float(x) for x in DEFAULT_EXIT_SWEEP]
    max_holds = max_holds or [int(x) for x in DEFAULT_HOLD_SWEEP]

    for entry in entry_edges:
        for exit_edge in exit_edges:
            for hold in max_holds:
                # Clone strategy config by constructing a new model with updated fields (Pydantic BaseModel supports copy).
                if hasattr(strategy_cfg, "model_copy"):
                    s = strategy_cfg.model_copy(update={"entry_edge_bps": float(entry), "exit_edge_bps": float(exit_edge), "max_holding_hours": int(hold)})
                else:
                    # fallback for duck-typed configs
                    class _S:  # noqa: D401
                        pass

                    s = _S()
                    for k, v in strategy_cfg.__dict__.items():
                        setattr(s, k, v)
                    s.entry_edge_bps = float(entry)
                    s.exit_edge_bps = float(exit_edge)
                    s.max_holding_hours = int(hold)

                bt = FundingHarvestBacktester(
                    df=df,
                    strategy_cfg=s,
                    cost_cfg=cost_cfg,
                    risk_cfg=risk_cfg,
                    initial_capital=initial_capital,
                )
                res = bt.run()
                m = compute_summary_metrics(res, capital=initial_capital)
                rows.append(
                    {
                        "entry_edge_bps": float(entry),
                        "exit_edge_bps": float(exit_edge),
                        "max_holding_hours": int(hold),
                        "trade_count": int(m["trade_count"]),
                        "sharpe_ann": float(m["sharpe_ann"]) if m["sharpe_ann"] == m["sharpe_ann"] else None,
                        "max_drawdown_pct": float(m["max_drawdown_pct"]) if m["max_drawdown_pct"] == m["max_drawdown_pct"] else None,
                        "total_net_pnl_usd": float(m["total_net_pnl_usd"]),
                        "total_return_pct": float(m["total_return_pct"]),
                    }
                )
    return pd.DataFrame(rows)


def select_stable_region(results_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Simple heuristic selection: best Sharpe among configs with at least 5 trades.
    """
    if results_df.empty:
        return {}
    df = results_df.copy()
    df = df.dropna(subset=["sharpe_ann"])
    df = df[df["trade_count"] >= 5]
    if df.empty:
        return {}
    best = df.sort_values("sharpe_ann", ascending=False).iloc[0].to_dict()
    return best


def _plot_entry_vs_sharpe(results_df: pd.DataFrame, out_path: Path) -> None:
    if results_df.empty:
        return
    # Aggregate over exit/hold by taking max Sharpe for each entry threshold.
    df = results_df.dropna(subset=["sharpe_ann"]).groupby("entry_edge_bps")["sharpe_ann"].max().reset_index()
    if df.empty:
        return
    plt.figure(figsize=(7, 4))
    plt.plot(df["entry_edge_bps"], df["sharpe_ann"], marker="o")
    plt.title("Entry Threshold vs Best Sharpe")
    plt.xlabel("entry_edge_bps")
    plt.ylabel("Sharpe (ann.)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _plot_entry_vs_trade_count(results_df: pd.DataFrame, out_path: Path) -> None:
    if results_df.empty:
        return
    df = results_df.groupby("entry_edge_bps")["trade_count"].max().reset_index()
    plt.figure(figsize=(7, 4))
    plt.plot(df["entry_edge_bps"], df["trade_count"], marker="o")
    plt.title("Entry Threshold vs Trade Count (max over grid)")
    plt.xlabel("entry_edge_bps")
    plt.ylabel("Trades")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run an in-sample threshold sweep and save results.")
    p.add_argument("--dataset", type=str, default="data/processed/master_1m.parquet")
    p.add_argument("--out-csv", type=str, default="outputs/tables/threshold_sweep.csv")
    p.add_argument("--figures-dir", type=str, default="outputs/figures")
    p.add_argument("--scenario", type=str, default="baseline", choices=["baseline", "optimistic"])
    p.add_argument("--capital", type=float, default=100000.0)
    p.add_argument("--entry-edges", type=str, default=None, help="Comma list override, e.g. 5,10,15")
    p.add_argument("--exit-edges", type=str, default=None, help="Comma list override, e.g. -50,-20,0,2")
    p.add_argument("--max-holds", type=str, default=None, help="Comma list override, e.g. 12,24,48,72")
    return p.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()
    cfg = get_config()

    df = read_parquet(args.dataset)
    cost_cfg = getattr(cfg.costs, args.scenario)
    entry_edges = [float(x) for x in args.entry_edges.split(",")] if args.entry_edges else None
    exit_edges = [float(x) for x in args.exit_edges.split(",")] if args.exit_edges else None
    max_holds = [int(x) for x in args.max_holds.split(",")] if args.max_holds else None

    results = run_threshold_sweep(
        df,
        cfg.strategy,
        cost_cfg,
        risk_cfg=cfg.risk,
        entry_edges=entry_edges,
        exit_edges=exit_edges,
        max_holds=max_holds,
        initial_capital=float(args.capital),
    )

    out_csv = Path(args.out_csv)
    ensure_dir(out_csv.parent)
    results.to_csv(out_csv, index=False)

    figures_dir = Path(args.figures_dir)
    ensure_dir(figures_dir)
    _plot_entry_vs_sharpe(results, figures_dir / "threshold_vs_sharpe.png")
    _plot_entry_vs_trade_count(results, figures_dir / "trade_count_vs_entry_threshold.png")

    best = select_stable_region(results)
    if best:
        logger.info("Best (heuristic) region: %s", best)
    logger.info("Saved sweep: %s rows -> %s", len(results), out_csv)


if __name__ == "__main__":
    main()
