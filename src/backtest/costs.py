"""
Cost model utilities.

Only `get_round_trip_cost_bps` is needed for dataset building (Prompt 3).
Entry/exit cost USD calculations are implemented in later steps.
"""

from __future__ import annotations

from typing import Any


def get_round_trip_cost_bps(cost_config: Any) -> float:
    """
    Round-trip costs in bps for long spot + short perp.

    round_trip_cost_bps = (
        2 * spot_fee_bps_per_side
        + 2 * perp_fee_bps_per_side
        + 2 * spot_slippage_bps_per_side
        + 2 * perp_slippage_bps_per_side
    )
    """
    spot_fee = float(getattr(cost_config, "spot_fee_bps_per_side"))
    perp_fee = float(getattr(cost_config, "perp_fee_bps_per_side"))
    spot_slip = float(getattr(cost_config, "spot_slippage_bps_per_side"))
    perp_slip = float(getattr(cost_config, "perp_slippage_bps_per_side"))
    return 2.0 * (spot_fee + perp_fee + spot_slip + perp_slip)


def _get_cost_attr(cost_config: Any, name: str) -> float:
    if isinstance(cost_config, dict):
        return float(cost_config[name])
    return float(getattr(cost_config, name))


def compute_entry_cost_usd(notional_usd: float, cost_config: Any) -> float:
    """
    Entry cost for opening long spot + short perp (one side each).
    """
    bps = (
        _get_cost_attr(cost_config, "spot_fee_bps_per_side")
        + _get_cost_attr(cost_config, "perp_fee_bps_per_side")
        + _get_cost_attr(cost_config, "spot_slippage_bps_per_side")
        + _get_cost_attr(cost_config, "perp_slippage_bps_per_side")
    )
    return float(notional_usd) * float(bps) / 10000.0


def compute_exit_cost_usd(notional_usd: float, cost_config: Any) -> float:
    """
    Exit cost for closing long spot + short perp (one side each).
    """
    return compute_entry_cost_usd(notional_usd, cost_config)
