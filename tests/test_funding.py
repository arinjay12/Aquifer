"""
Unit tests for funding computations.
"""

import unittest

from src.signals.funding import compute_minutes_to_next_funding


class TestFunding(unittest.TestCase):
    def test_minutes_to_next_funding_exact(self) -> None:
        ts = "2024-01-01T00:00:00Z"
        nft = "2024-01-01T08:00:00Z"
        self.assertEqual(compute_minutes_to_next_funding(ts, nft), 480.0)


if __name__ == "__main__":
    unittest.main()
