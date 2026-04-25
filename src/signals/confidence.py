"""
Confidence scoring for signals.

The goal is a simple, interpretable confidence score in [0, 1] for live logging:
- higher when expected edge is far above the entry threshold
- modestly higher when there is more time to next funding (less "rush" risk)
"""

from __future__ import annotations

import math


def compute_confidence_score(
    *,
    expected_edge_bps: float,
    entry_edge_bps: float,
    minutes_to_next_funding: float,
    min_minutes_to_next_funding: float,
    scale_bps: float = 10.0,
) -> float:
    """
    Map edge and timing into a [0, 1] confidence score.
    """
    scale = max(1e-6, float(scale_bps))
    z = (float(expected_edge_bps) - float(entry_edge_bps)) / scale
    edge_score = 1.0 / (1.0 + math.exp(-z))

    if minutes_to_next_funding != minutes_to_next_funding:
        time_score = 0.0
    else:
        denom = max(1.0, float(min_minutes_to_next_funding))
        time_score = min(1.0, max(0.0, float(minutes_to_next_funding) / denom))

    # Weighted blend: edge dominates, time is a small modifier.
    score = 0.85 * edge_score + 0.15 * time_score
    return float(min(1.0, max(0.0, score)))

