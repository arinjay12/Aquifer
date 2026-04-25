"""
Expected edge computation.

Formula (v1):
    expected_edge_bps = funding_rate_next_bps + basis_bps - round_trip_cost_bps
"""

from __future__ import annotations

from typing import Any, Union

import numpy as np
import pandas as pd


NumberLike = Union[float, int, np.number, pd.Series]


def _get_cost_value(cost_cfg: Any, key: str) -> float:
    if isinstance(cost_cfg, dict):
        return float(cost_cfg[key])
    return float(getattr(cost_cfg, key))


def compute_round_trip_cost_bps(cost_cfg: Any) -> float:
    """
    Compute round-trip trading costs in bps.

    round_trip_cost_bps = (
        2 * spot_fee_bps_per_side
        + 2 * perp_fee_bps_per_side
        + 2 * spot_slippage_bps_per_side
        + 2 * perp_slippage_bps_per_side
    )
    """
    spot_fee = _get_cost_value(cost_cfg, "spot_fee_bps_per_side")
    perp_fee = _get_cost_value(cost_cfg, "perp_fee_bps_per_side")
    spot_slip = _get_cost_value(cost_cfg, "spot_slippage_bps_per_side")
    perp_slip = _get_cost_value(cost_cfg, "perp_slippage_bps_per_side")
    return 2.0 * (spot_fee + perp_fee + spot_slip + perp_slip)


def compute_expected_edge_bps(
    *,
    funding_rate_next: NumberLike,
    basis_bps: NumberLike,
    round_trip_cost_bps: float,
) -> NumberLike:
    """
    Compute expected edge in bps.

    funding_rate_next is in decimal terms (e.g., 0.0001 = 1 bp).
    """
    funding_rate_next_bps = funding_rate_next * 10000.0
    return funding_rate_next_bps + basis_bps - float(round_trip_cost_bps)
