"""
Basis computation.

Formula:
    basis_bps = (perp_close - spot_close) / spot_close * 10000
"""

from __future__ import annotations

from typing import Union

import numpy as np
import pandas as pd


NumberLike = Union[float, int, np.number, pd.Series]


def compute_basis_bps(spot_close: NumberLike, perp_close: NumberLike) -> NumberLike:
    """
    Compute basis in basis-points (bps).

    Works with scalars or pandas Series.
    """
    if isinstance(spot_close, pd.Series) or isinstance(perp_close, pd.Series):
        with np.errstate(divide="ignore", invalid="ignore"):
            return (perp_close - spot_close) / spot_close * 10000.0

    spot = np.asarray(spot_close, dtype="float64")
    perp = np.asarray(perp_close, dtype="float64")
    with np.errstate(divide="ignore", invalid="ignore"):
        out = (perp - spot) / spot * 10000.0
    if out.shape == ():
        return float(out)  # type: ignore[return-value]
    return out  # type: ignore[return-value]
