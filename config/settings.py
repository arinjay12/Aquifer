"""
Typed configuration loading.

Loads `config/params.yaml` once and exposes Pydantic-based config objects:

- DataConfig
- StrategyConfig
- CostScenarioConfig / CostsConfig
- LiveConfig
- AppConfig

This module intentionally contains no strategy or trading logic.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field


class DataConfig(BaseModel):
    start_date: str = Field(..., description="YYYY-MM-DD (UTC)")
    end_date: Optional[str] = Field(default=None, description="YYYY-MM-DD (UTC) or null for latest")
    interval: str = Field(default="1m", description="Binance kline interval, e.g. 1m")


class StrategyConfig(BaseModel):
    entry_edge_bps: float = 15.0
    min_basis_bps: float = 5.0
    exit_edge_bps: float = 2.0
    exit_basis_bps: float = 1.0
    hard_stop_basis_widen_bps: float = 25.0
    max_holding_hours: int = 48
    min_minutes_to_next_funding: int = 10
    positive_funding_only: bool = True


class CostScenarioConfig(BaseModel):
    spot_fee_bps_per_side: float
    perp_fee_bps_per_side: float
    spot_slippage_bps_per_side: float
    perp_slippage_bps_per_side: float
    capital_mode: str = "isolated"


class CostsConfig(BaseModel):
    baseline: CostScenarioConfig
    optimistic: CostScenarioConfig


class LiveConfig(BaseModel):
    poll_seconds: int = 60
    output_csv: str = "outputs/live_logs/live_signal_log.csv"


class RiskConfig(BaseModel):
    position_fraction: float = 1.0
    max_drawdown_pause_pct: float = 10.0
    pause_minutes: int = 240
    confidence_scale_bps: float = 10.0


class AppConfig(BaseModel):
    symbol_spot: str = "ETHUSDT"
    symbol_perp: str = "ETHUSDT"
    venue: str = "binance"
    data: DataConfig
    strategy: StrategyConfig
    costs: CostsConfig
    live: LiveConfig
    risk: RiskConfig = RiskConfig()


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping at root of {path}, got {type(data)}")
    return data


def load_config(params_path: str | Path | None = None) -> AppConfig:
    """
    Load AppConfig from YAML, optionally overridden by env var `FUNDING_HARVEST_PARAMS_PATH`.
    """
    load_dotenv(override=False)

    if params_path is None:
        params_path = os.getenv("FUNDING_HARVEST_PARAMS_PATH")
        if params_path is None:
            params_path = Path.cwd() / "config" / "params.yaml"

    params_path = Path(params_path)
    raw = _read_yaml(params_path)
    return AppConfig.model_validate(raw)


_CACHED_CONFIG: AppConfig | None = None


def get_config(params_path: str | Path | None = None, *, force_reload: bool = False) -> AppConfig:
    """
    Get a cached AppConfig instance (load once per process by default).
    """
    global _CACHED_CONFIG
    if force_reload or _CACHED_CONFIG is None:
        _CACHED_CONFIG = load_config(params_path=params_path)
    return _CACHED_CONFIG
