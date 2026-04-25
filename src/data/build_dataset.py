"""
Build the canonical minute-level dataset.

Creates the v1 canonical dataset where each row is one UTC minute:

[
    "timestamp_utc",
    "spot_close",
    "perp_close",
    "mark_price",
    "index_price",
    "basis_bps",
    "funding_rate_last",
    "funding_rate_next",
    "next_funding_time",
    "minutes_to_next_funding",
    "expected_edge_bps",
]

Notes:
- Historical mark/index prices are not available from the kline endpoints; v1 sets them to NaN.
- Funding "next" is derived by looking forward to the next funding event in the funding history table.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd

from config.settings import get_config
from src.backtest.costs import get_round_trip_cost_bps
from src.utils.io_utils import read_parquet, write_parquet
from src.utils.logging_utils import get_logger, setup_logging


logger = get_logger(__name__)


CANON_COLS = [
    "timestamp_utc",
    "spot_close",
    "perp_close",
    "mark_price",
    "index_price",
    "basis_bps",
    "funding_rate_last",
    "funding_rate_next",
    "next_funding_time",
    "minutes_to_next_funding",
    "expected_edge_bps",
]


def load_raw_spot(raw_dir: Path, symbol: str, interval: str) -> pd.DataFrame:
    files = sorted((raw_dir / "spot").glob(f"{symbol}_{interval}_*.parquet"))
    if not files:
        # Backward compatibility for single-file downloads.
        single = raw_dir / "spot" / f"{symbol}_{interval}.parquet"
        files = [single] if single.exists() else []
    if not files:
        return pd.DataFrame(columns=["timestamp_utc", "spot_close"])

    parts = []
    for p in files:
        df = read_parquet(p)
        if df.empty:
            continue
        df = df[["timestamp_utc", "close"]].copy()
        parts.append(df)
    if not parts:
        return pd.DataFrame(columns=["timestamp_utc", "spot_close"])
    df = pd.concat(parts, ignore_index=True)
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.rename(columns={"close": "spot_close"})
    return df.sort_values("timestamp_utc").drop_duplicates("timestamp_utc", keep="last").reset_index(drop=True)


def load_raw_perp(raw_dir: Path, symbol: str, interval: str) -> pd.DataFrame:
    files = sorted((raw_dir / "perp").glob(f"{symbol}_{interval}_*.parquet"))
    if not files:
        single = raw_dir / "perp" / f"{symbol}_{interval}.parquet"
        files = [single] if single.exists() else []
    if not files:
        return pd.DataFrame(columns=["timestamp_utc", "perp_close"])

    parts = []
    for p in files:
        df = read_parquet(p)
        if df.empty:
            continue
        df = df[["timestamp_utc", "close"]].copy()
        parts.append(df)
    if not parts:
        return pd.DataFrame(columns=["timestamp_utc", "perp_close"])
    df = pd.concat(parts, ignore_index=True)
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.rename(columns={"close": "perp_close"})
    return df.sort_values("timestamp_utc").drop_duplicates("timestamp_utc", keep="last").reset_index(drop=True)


def load_raw_funding(raw_dir: Path, symbol: str) -> pd.DataFrame:
    path = raw_dir / "funding" / f"{symbol}_funding.parquet"
    df = read_parquet(path)
    if df.empty:
        return df
    # From BinanceClient.get_funding_rate_history
    cols = [c for c in ["funding_time_utc", "fundingRate"] if c in df.columns]
    df = df[cols].copy()
    df["funding_time_utc"] = pd.to_datetime(df["funding_time_utc"], utc=True)
    df["fundingRate"] = pd.to_numeric(df["fundingRate"], errors="coerce")
    df = df.rename(columns={"fundingRate": "funding_rate"})
    return df.sort_values("funding_time_utc").drop_duplicates("funding_time_utc", keep="last").reset_index(drop=True)


def align_to_minute_grid(
    spot_df: pd.DataFrame, perp_df: pd.DataFrame, start: pd.Timestamp | None = None, end: pd.Timestamp | None = None
) -> pd.DataFrame:
    if spot_df.empty or perp_df.empty:
        raise ValueError("Spot and perp data must be non-empty to build the dataset.")

    start_ts = max(spot_df["timestamp_utc"].min(), perp_df["timestamp_utc"].min())
    end_ts = min(spot_df["timestamp_utc"].max(), perp_df["timestamp_utc"].max())

    if start is not None:
        start_ts = max(start_ts, start)
    if end is not None:
        end_ts = min(end_ts, end - pd.Timedelta(minutes=1))

    start_ts = pd.Timestamp(start_ts).floor("min")
    end_ts = pd.Timestamp(end_ts).floor("min")
    if end_ts < start_ts:
        raise ValueError(f"Invalid aligned range: start={start_ts}, end={end_ts}")

    grid = pd.DataFrame({"timestamp_utc": pd.date_range(start_ts, end_ts, freq="1min", tz="UTC")})

    out = grid.merge(spot_df, on="timestamp_utc", how="left").merge(perp_df, on="timestamp_utc", how="left")
    # Do not fabricate prices across large gaps; keep only minutes where both legs exist.
    out = out.dropna(subset=["spot_close", "perp_close"]).reset_index(drop=True)
    return out


def attach_last_funding(master: pd.DataFrame, funding_df: pd.DataFrame) -> pd.DataFrame:
    if funding_df.empty:
        master["funding_rate_last"] = np.nan
        return master
    right = funding_df.rename(columns={"funding_time_utc": "funding_time_last", "funding_rate": "funding_rate_last"})
    left = master.sort_values("timestamp_utc").copy()
    right = right.sort_values("funding_time_last").copy()
    left["_asof_key_ns"] = (
        pd.to_datetime(left["timestamp_utc"], utc=True).dt.tz_localize(None).astype("datetime64[ns]").astype("int64")
    )
    right["_asof_key_ns"] = (
        pd.to_datetime(right["funding_time_last"], utc=True)
        .dt.tz_localize(None)
        .astype("datetime64[ns]")
        .astype("int64")
    )
    master = pd.merge_asof(left, right, on="_asof_key_ns", direction="backward")
    master = master.drop(columns=["_asof_key_ns", "funding_time_last"])
    return master


def attach_next_funding(master: pd.DataFrame, funding_df: pd.DataFrame) -> pd.DataFrame:
    if funding_df.empty:
        master["funding_rate_next"] = np.nan
        master["next_funding_time"] = pd.NaT
        master["minutes_to_next_funding"] = np.nan
        return master

    right = funding_df.rename(columns={"funding_time_utc": "next_funding_time", "funding_rate": "funding_rate_next"})
    left = master.sort_values("timestamp_utc").copy()
    right = right.sort_values("next_funding_time").copy()
    left["_asof_key_ns"] = (
        pd.to_datetime(left["timestamp_utc"], utc=True).dt.tz_localize(None).astype("datetime64[ns]").astype("int64")
    )
    right["_asof_key_ns"] = (
        pd.to_datetime(right["next_funding_time"], utc=True)
        .dt.tz_localize(None)
        .astype("datetime64[ns]")
        .astype("int64")
    )
    master = pd.merge_asof(left, right, on="_asof_key_ns", direction="forward", allow_exact_matches=False)
    master = master.drop(columns=["_asof_key_ns"])
    master["minutes_to_next_funding"] = (
        (master["next_funding_time"] - master["timestamp_utc"]).dt.total_seconds() / 60.0
    )
    return master


def compute_basis(master: pd.DataFrame) -> pd.DataFrame:
    master["basis_bps"] = (master["perp_close"] - master["spot_close"]) / master["spot_close"] * 10000.0
    return master


def compute_expected_edge(master: pd.DataFrame, round_trip_cost_bps: float) -> pd.DataFrame:
    master["expected_edge_bps"] = (master["funding_rate_next"] * 10000.0) + master["basis_bps"] - float(
        round_trip_cost_bps
    )
    return master


def build_master_dataset(
    *,
    raw_dir: Path,
    symbol_spot: str,
    symbol_perp: str,
    interval: str,
    funding_symbol: str,
    start: pd.Timestamp | None,
    end: pd.Timestamp | None,
    round_trip_cost_bps: float,
) -> pd.DataFrame:
    spot = load_raw_spot(raw_dir, symbol_spot, interval)
    perp = load_raw_perp(raw_dir, symbol_perp, interval)
    funding = load_raw_funding(raw_dir, funding_symbol)

    # Bound the dataset so that "next funding" is defined for every row.
    effective_end = end
    if not funding.empty:
        funding_max = pd.to_datetime(funding["funding_time_utc"].max(), utc=True)
        if effective_end is None:
            effective_end = funding_max
        else:
            effective_end = min(effective_end, funding_max)

    master = align_to_minute_grid(spot, perp, start=start, end=effective_end)
    master["mark_price"] = np.nan
    master["index_price"] = np.nan

    master = attach_last_funding(master, funding)
    master = attach_next_funding(master, funding)
    before = len(master)
    master = master.dropna(subset=["funding_rate_next", "next_funding_time"]).reset_index(drop=True)
    dropped = before - len(master)
    if dropped:
        logger.info("Dropped %s minutes with unknown next funding.", dropped)
    master = compute_basis(master)
    master = compute_expected_edge(master, round_trip_cost_bps=round_trip_cost_bps)

    master = master[CANON_COLS].sort_values("timestamp_utc").reset_index(drop=True)
    return master


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build the canonical 1-minute master dataset from raw parquet files.")
    p.add_argument("--raw-dir", type=str, default="data/raw", help="Raw data directory (default: data/raw).")
    p.add_argument(
        "--out",
        type=str,
        default="data/processed/master_1m.parquet",
        help="Output parquet path (default: data/processed/master_1m.parquet).",
    )
    p.add_argument(
        "--scenario",
        type=str,
        default="baseline",
        choices=["baseline", "optimistic"],
        help="Cost scenario used for expected edge computation.",
    )
    return p.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()
    cfg = get_config()

    raw_dir = Path(args.raw_dir)
    out_path = Path(args.out)

    start = pd.Timestamp(cfg.data.start_date, tz="UTC")
    end = pd.Timestamp(cfg.data.end_date, tz="UTC") + pd.Timedelta(days=1) if cfg.data.end_date else None

    cost_scenario = getattr(cfg.costs, args.scenario)
    round_trip_cost_bps = get_round_trip_cost_bps(cost_scenario)
    logger.info("Using cost scenario '%s' -> round_trip_cost_bps=%.4f", args.scenario, round_trip_cost_bps)

    master = build_master_dataset(
        raw_dir=raw_dir,
        symbol_spot=cfg.symbol_spot,
        symbol_perp=cfg.symbol_perp,
        interval=cfg.data.interval,
        funding_symbol=cfg.symbol_perp,
        start=start,
        end=end,
        round_trip_cost_bps=round_trip_cost_bps,
    )

    write_parquet(master, out_path, index=False)
    logger.info("Saved master dataset: %s rows -> %s", len(master), out_path)


if __name__ == "__main__":
    main()
