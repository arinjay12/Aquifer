"""
Historical data downloader CLI.

CLI:
    python -m src.data.download_historical

Responsibilities:
- download 1m spot klines
- download 1m perp klines
- download funding history
- save raw parquet files under `data/raw/...`
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd

from config.settings import get_config
from src.data.binance_client import BinanceClient
from src.utils.io_utils import write_parquet
from src.utils.logging_utils import get_logger, setup_logging


logger = get_logger(__name__)


def _parse_date_utc(s: str) -> pd.Timestamp:
    ts = pd.Timestamp(s)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _parse_end_exclusive_utc(s: str) -> pd.Timestamp:
    """
    Interpret an end date like 'YYYY-MM-DD' as exclusive end (start of next day, UTC).
    If a full timestamp is provided, use it as-is.
    """
    if len(s) == 10 and s[4] == "-" and s[7] == "-":
        return _parse_date_utc(s) + pd.Timedelta(days=1)
    return _parse_date_utc(s)


def _iter_chunks(start: pd.Timestamp, end_exclusive: pd.Timestamp, chunk_days: int) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    chunks: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    cur = start
    step = pd.Timedelta(days=int(chunk_days))
    while cur < end_exclusive:
        nxt = min(cur + step, end_exclusive)
        chunks.append((cur, nxt))
        cur = nxt
    return chunks


def _chunk_path(out_dir: Path, kind: str, symbol: str, interval: str | None, start: pd.Timestamp, end: pd.Timestamp) -> Path:
    s = start.strftime("%Y%m%d")
    e = (end - pd.Timedelta(seconds=1)).strftime("%Y%m%d")
    if interval:
        return out_dir / kind / f"{symbol}_{interval}_{s}_{e}.parquet"
    return out_dir / kind / f"{symbol}_{s}_{e}.parquet"


def download_all(
    *,
    symbol_spot: str,
    symbol_perp: str,
    interval: str,
    start_time: pd.Timestamp,
    end_time: pd.Timestamp | None,
    out_dir: Path,
    chunk_days: int = 30,
) -> None:
    client = BinanceClient()

    if end_time is None:
        end_time = pd.Timestamp.utcnow().tz_localize("UTC")

    chunks = _iter_chunks(start_time, end_time, chunk_days=int(chunk_days))
    logger.info("Downloading spot klines in %s chunks of ~%s days", len(chunks), chunk_days)
    for a, b in chunks:
        spot_path = _chunk_path(out_dir, "spot", symbol_spot, interval, a, b)
        if spot_path.exists():
            continue
        spot_df = client.get_spot_klines(symbol=symbol_spot, interval=interval, start_time=a, end_time=b)
        write_parquet(spot_df, spot_path, index=False)
        logger.info("Saved spot: %s rows -> %s", len(spot_df), spot_path)

    logger.info("Downloading perp klines in %s chunks of ~%s days", len(chunks), chunk_days)
    for a, b in chunks:
        perp_path = _chunk_path(out_dir, "perp", symbol_perp, interval, a, b)
        if perp_path.exists():
            continue
        perp_df = client.get_perp_klines(symbol=symbol_perp, interval=interval, start_time=a, end_time=b)
        write_parquet(perp_df, perp_path, index=False)
        logger.info("Saved perp: %s rows -> %s", len(perp_df), perp_path)

    logger.info("Downloading funding rate history: %s", symbol_perp)
    funding_end = end_time + pd.Timedelta(hours=8)
    funding_df = client.get_funding_rate_history(symbol=symbol_perp, start_time=start_time, end_time=funding_end)
    funding_path = out_dir / "funding" / f"{symbol_perp}_funding.parquet"
    write_parquet(funding_df, funding_path, index=False)
    logger.info("Saved funding history: %s rows -> %s", len(funding_df), funding_path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download ETHUSDT spot/perp 1m klines + funding history from Binance.")
    p.add_argument("--start-date", type=str, default=None, help="Override config start_date (YYYY-MM-DD).")
    p.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="Override config end_date (YYYY-MM-DD, inclusive day; or full timestamp).",
    )
    p.add_argument("--interval", type=str, default=None, help="Override interval (default from config).")
    p.add_argument("--raw-dir", type=str, default="data/raw", help="Output raw directory (default: data/raw).")
    p.add_argument("--chunk-days", type=int, default=30, help="Chunk size in days for kline downloads (default: 30).")
    return p.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()
    cfg = get_config()

    interval = args.interval or cfg.data.interval
    start_date = args.start_date or cfg.data.start_date
    end_date = args.end_date if args.end_date is not None else cfg.data.end_date

    start_time = _parse_date_utc(start_date)
    end_time: Optional[pd.Timestamp] = _parse_end_exclusive_utc(end_date) if end_date else None

    out_dir = Path(args.raw_dir)
    download_all(
        symbol_spot=cfg.symbol_spot,
        symbol_perp=cfg.symbol_perp,
        interval=interval,
        start_time=start_time,
        end_time=end_time,
        out_dir=out_dir,
        chunk_days=int(args.chunk_days),
    )


if __name__ == "__main__":
    main()
