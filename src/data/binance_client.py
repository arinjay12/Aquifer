"""
Binance REST client wrapper.

Implements official REST endpoints with:
- retries + exponential backoff
- simple rate-limit handling
- pagination via startTime/endTime + limit
- UTC timestamp normalization

Endpoints:
- spot klines
- futures klines
- funding rate history
- mark price / premium index snapshot
- open interest history (optional; not used in v1 core)
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Optional

import pandas as pd
import requests

from src.utils.logging_utils import get_logger


logger = get_logger(__name__)


def _to_millis_utc(value: Any) -> int:
    """
    Convert input into UTC epoch milliseconds.

    Accepts:
    - int/float (assumed already ms)
    - datetime-like (naive assumed UTC)
    - pandas Timestamp (naive assumed UTC)
    - ISO string
    """
    if value is None:
        raise ValueError("time value cannot be None")
    if isinstance(value, (int, float)):
        return int(value)
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    ts = ts.tz_convert("UTC")
    return int(ts.value // 1_000_000)  # ns -> ms


def _ms_to_timestamp_utc(ms: int) -> pd.Timestamp:
    return pd.Timestamp(ms, unit="ms", tz="UTC")


@dataclass(frozen=True)
class BinanceEndpoints:
    spot_base: str = "https://api.binance.com"
    futures_base: str = "https://fapi.binance.com"


class BinanceClient:
    def __init__(
        self,
        api_key: str | None = None,
        api_secret: str | None = None,
        timeout_s: int = 30,
        max_retries: int = 6,
        backoff_base_s: float = 0.5,
    ):
        self.api_key = api_key
        self.api_secret = api_secret
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.backoff_base_s = backoff_base_s

        self.endpoints = BinanceEndpoints()
        self._session = requests.Session()

    def _request(self, method: str, url: str, params: Optional[dict[str, Any]] = None) -> Any:
        headers: dict[str, str] = {}
        if self.api_key:
            headers["X-MBX-APIKEY"] = self.api_key

        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                resp = self._session.request(
                    method=method,
                    url=url,
                    params=params,
                    headers=headers,
                    timeout=self.timeout_s,
                )

                if resp.status_code in (418, 429):
                    retry_after = resp.headers.get("Retry-After")
                    sleep_s = float(retry_after) if retry_after else (self.backoff_base_s * (2**attempt))
                    logger.warning("Rate limited (%s). Sleeping %.2fs then retrying.", resp.status_code, sleep_s)
                    time.sleep(sleep_s)
                    continue

                resp.raise_for_status()
                return resp.json()
            except Exception as e:  # noqa: BLE001
                last_err = e
                time.sleep(self.backoff_base_s * (2**attempt))
                continue

        raise RuntimeError(f"Binance request failed after {self.max_retries} attempts: {url}") from last_err

    def get_spot_klines(
        self,
        symbol: str,
        interval: str,
        start_time: Any,
        end_time: Any | None,
        limit: int = 1000,
    ) -> pd.DataFrame:
        url = f"{self.endpoints.spot_base}/api/v3/klines"
        return self._get_klines(url=url, symbol=symbol, interval=interval, start_time=start_time, end_time=end_time, limit=limit)

    def get_perp_klines(
        self,
        symbol: str,
        interval: str,
        start_time: Any,
        end_time: Any | None,
        limit: int = 1000,
    ) -> pd.DataFrame:
        url = f"{self.endpoints.futures_base}/fapi/v1/klines"
        return self._get_klines(url=url, symbol=symbol, interval=interval, start_time=start_time, end_time=end_time, limit=limit)

    def _get_klines(
        self,
        *,
        url: str,
        symbol: str,
        interval: str,
        start_time: Any,
        end_time: Any | None,
        limit: int,
    ) -> pd.DataFrame:
        start_ms = _to_millis_utc(start_time)
        end_ms = _to_millis_utc(end_time) if end_time is not None else None
        query_end_ms = (end_ms - 1) if end_ms is not None else None

        all_rows: list[list[Any]] = []
        current_start = start_ms

        while True:
            params: dict[str, Any] = {
                "symbol": symbol,
                "interval": interval,
                "startTime": current_start,
                "limit": limit,
            }
            if query_end_ms is not None:
                params["endTime"] = query_end_ms

            data = self._request("GET", url, params=params)
            if not data:
                break
            if not isinstance(data, list):
                raise ValueError(f"Unexpected klines response type: {type(data)}")

            all_rows.extend(data)

            last_open_time = int(data[-1][0])
            next_start = last_open_time + 1
            if next_start <= current_start:
                break
            current_start = next_start

            if end_ms is not None and current_start >= end_ms:
                break

            time.sleep(0.05)

        if not all_rows:
            return pd.DataFrame()

        cols = [
            "open_time_ms",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time_ms",
            "quote_asset_volume",
            "num_trades",
            "taker_buy_base_asset_volume",
            "taker_buy_quote_asset_volume",
            "ignore",
        ]
        df = pd.DataFrame(all_rows, columns=cols)
        df["timestamp_utc"] = df["open_time_ms"].astype("int64").map(_ms_to_timestamp_utc)

        float_cols = [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "quote_asset_volume",
            "taker_buy_base_asset_volume",
            "taker_buy_quote_asset_volume",
        ]
        for c in float_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df["num_trades"] = pd.to_numeric(df["num_trades"], errors="coerce").astype("Int64")

        df = df.sort_values("timestamp_utc").reset_index(drop=True)
        return df

    def get_funding_rate_history(
        self,
        symbol: str,
        start_time: Any,
        end_time: Any | None,
        limit: int = 1000,
    ) -> pd.DataFrame:
        url = f"{self.endpoints.futures_base}/fapi/v1/fundingRate"
        start_ms = _to_millis_utc(start_time)
        end_ms = _to_millis_utc(end_time) if end_time is not None else None
        query_end_ms = (end_ms - 1) if end_ms is not None else None

        rows: list[dict[str, Any]] = []
        current_start = start_ms

        while True:
            params: dict[str, Any] = {"symbol": symbol, "startTime": current_start, "limit": limit}
            if query_end_ms is not None:
                params["endTime"] = query_end_ms

            data = self._request("GET", url, params=params)
            if not data:
                break
            if not isinstance(data, list):
                raise ValueError(f"Unexpected fundingRate response type: {type(data)}")

            rows.extend(data)
            last_time = int(data[-1]["fundingTime"])
            next_start = last_time + 1
            if next_start <= current_start:
                break
            current_start = next_start

            if end_ms is not None and current_start >= end_ms:
                break

            time.sleep(0.05)

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        df["fundingTime_ms"] = pd.to_numeric(df["fundingTime"], errors="coerce").astype("int64")
        df["fundingRate"] = pd.to_numeric(df["fundingRate"], errors="coerce")
        df["funding_time_utc"] = df["fundingTime_ms"].map(_ms_to_timestamp_utc)
        df = df.sort_values("funding_time_utc").reset_index(drop=True)
        return df

    def get_mark_price_snapshot(self, symbol: str) -> dict[str, Any]:
        """
        Futures premium index snapshot endpoint.
        """
        url = f"{self.endpoints.futures_base}/fapi/v1/premiumIndex"
        data = self._request("GET", url, params={"symbol": symbol})
        if not isinstance(data, dict):
            raise ValueError(f"Unexpected premiumIndex response type: {type(data)}")

        out = dict(data)
        for key in ["markPrice", "indexPrice", "lastFundingRate", "interestRate"]:
            if key in out:
                try:
                    out[key] = float(out[key])
                except Exception:  # noqa: BLE001
                    pass
        if "nextFundingTime" in out:
            out["nextFundingTime_utc"] = _ms_to_timestamp_utc(int(out["nextFundingTime"]))
        if "time" in out:
            out["time_utc"] = _ms_to_timestamp_utc(int(out["time"]))
        return out

    def get_open_interest_hist(self, symbol: str, period: str, limit: int = 500) -> pd.DataFrame:
        url = f"{self.endpoints.futures_base}/futures/data/openInterestHist"
        data = self._request("GET", url, params={"symbol": symbol, "period": period, "limit": limit})
        if not isinstance(data, list):
            raise ValueError(f"Unexpected openInterestHist response type: {type(data)}")
        if not data:
            return pd.DataFrame()

        df = pd.DataFrame(data)
        if "timestamp" in df.columns:
            df["timestamp_utc"] = (
                pd.to_numeric(df["timestamp"], errors="coerce").astype("int64").map(_ms_to_timestamp_utc)
            )
        for c in ["sumOpenInterest", "sumOpenInterestValue"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        return df.reset_index(drop=True)

    def get_spot_ticker_price(self, symbol: str) -> float:
        url = f"{self.endpoints.spot_base}/api/v3/ticker/price"
        data = self._request("GET", url, params={"symbol": symbol})
        if not isinstance(data, dict) or "price" not in data:
            raise ValueError(f"Unexpected spot ticker response: {data}")
        return float(data["price"])

    def get_perp_ticker_price(self, symbol: str) -> float:
        url = f"{self.endpoints.futures_base}/fapi/v1/ticker/price"
        data = self._request("GET", url, params={"symbol": symbol})
        if not isinstance(data, dict) or "price" not in data:
            raise ValueError(f"Unexpected futures ticker response: {data}")
        return float(data["price"])
