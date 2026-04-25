"""
Unit tests for cost/edge helper computations.
"""

import unittest

from src.signals.edge import compute_expected_edge_bps, compute_round_trip_cost_bps


class TestCostsAndEdge(unittest.TestCase):
    def test_compute_round_trip_cost_bps(self) -> None:
        cfg = {
            "spot_fee_bps_per_side": 10.0,
            "perp_fee_bps_per_side": 5.0,
            "spot_slippage_bps_per_side": 1.0,
            "perp_slippage_bps_per_side": 1.0,
        }
        self.assertEqual(compute_round_trip_cost_bps(cfg), 34.0)

    def test_compute_expected_edge_bps(self) -> None:
        # funding 0.0003 => 3 bps
        out = compute_expected_edge_bps(funding_rate_next=0.0003, basis_bps=10.0, round_trip_cost_bps=5.0)
        self.assertAlmostEqual(out, 3.0 + 10.0 - 5.0, places=12)


if __name__ == "__main__":
    unittest.main()
