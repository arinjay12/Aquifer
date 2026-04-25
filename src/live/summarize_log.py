"""
Summarize a live (or simulated-live) CSV log.

Usage:
    python -m src.live.summarize_log --csv outputs/live_logs/live_signal_log_48h_trade_window.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.utils.io_utils import ensure_dir
from src.utils.logging_utils import get_logger, setup_logging


logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarize a live signal log CSV.")
    p.add_argument("--csv", type=str, required=True)
    p.add_argument("--out", type=str, default=None, help="Output JSON path (default next to CSV).")
    p.add_argument("--sample-rows", type=int, default=10)
    return p.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()
    path = Path(args.csv)
    df = pd.read_csv(path)

    for c in ["entry_flag", "exit_flag", "signal_flag"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.lower().isin(["true", "1", "yes"])

    summary = {
        "csv": str(path),
        "rows": int(len(df)),
        "entries": int(df["entry_flag"].sum()) if "entry_flag" in df.columns else 0,
        "exits": int(df["exit_flag"].sum()) if "exit_flag" in df.columns else 0,
        "signals": int(df["signal_flag"].sum()) if "signal_flag" in df.columns else 0,
        "start": str(df["timestamp_utc"].iloc[0]) if len(df) else None,
        "end": str(df["timestamp_utc"].iloc[-1]) if len(df) else None,
    }

    out = Path(args.out) if args.out else path.with_suffix(".summary.json")
    ensure_dir(out.parent)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Wrote summary: %s", out)

    n = int(args.sample_rows)
    if n > 0 and len(df):
        sample = pd.concat([df.head(n // 2), df.tail(n - n // 2)], ignore_index=True)
        sample_path = path.with_suffix(".sample.csv")
        sample.to_csv(sample_path, index=False)
        logger.info("Wrote sample rows: %s", sample_path)


if __name__ == "__main__":
    main()

