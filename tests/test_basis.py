"""
Unit tests for basis computations.
"""

import unittest

import numpy as np

from src.signals.basis import compute_basis_bps


class TestBasis(unittest.TestCase):
    def test_compute_basis_bps_scalar(self) -> None:
        spot = 2000.0
        perp = 2001.0
        self.assertAlmostEqual(compute_basis_bps(spot, perp), 5.0, places=10)

    def test_compute_basis_bps_zero_spot(self) -> None:
        out = compute_basis_bps(0.0, 1.0)
        self.assertTrue(np.isnan(out) or np.isinf(out))


if __name__ == "__main__":
    unittest.main()
