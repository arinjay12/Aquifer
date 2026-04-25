"""
Unit tests for the minute-level backtest engine.
"""

import unittest

import pandas as pd

from src.backtest.engine import FundingHarvestBacktester


class TestBacktestEngine(unittest.TestCase):
    def test_single_trade_collects_funding_and_exits_on_convergence(self) -> None:
        ts0 = pd.Timestamp("2024-01-01T07:50:00Z")
        times = pd.date_range(ts0, periods=30, freq="1min", tz="UTC")

        # Construct a simple path:
        # - enter at 07:50 (10 min to 08:00 funding)
        # - collect funding at 08:00
        # - basis converges below exit_basis after 08:00 -> exit
        spot = []
        perp = []
        basis_bps = []
        expected_edge = []
        mins_to_next = []
        funding_last = []
        funding_next = []
        next_funding_time = []

        for t in times:
            # Baseline prices
            s = 2000.0
            p = 2004.0
            b = (p - s) / s * 10000.0  # 20 bps
            # Convergence after 08:05
            if t >= pd.Timestamp("2024-01-01T08:05:00Z"):
                p = 2000.1
                b = (p - s) / s * 10000.0

            # Next funding always 08:00 for pre-08:00; then 16:00 afterwards (not needed here)
            nft = pd.Timestamp("2024-01-01T08:00:00Z") if t < pd.Timestamp("2024-01-01T08:00:00Z") else pd.Timestamp("2024-01-01T16:00:00Z")
            mtnf = float((nft - t).total_seconds() / 60.0)

            # Use a positive next funding rate to enable entry
            fr_next = 0.0003

            # funding_rate_last only matters at 08:00 for accrual; set that event to 0.0005
            fr_last = 0.0005 if t == pd.Timestamp("2024-01-01T08:00:00Z") else 0.0001

            # Expected edge just ensure entry, keep it above entry threshold until after convergence
            ee = 20.0
            if t >= pd.Timestamp("2024-01-01T08:05:00Z"):
                ee = 20.0

            spot.append(s)
            perp.append(p)
            basis_bps.append(b)
            expected_edge.append(ee)
            mins_to_next.append(mtnf)
            funding_last.append(fr_last)
            funding_next.append(fr_next)
            next_funding_time.append(nft)

        df = pd.DataFrame(
            {
                "timestamp_utc": times,
                "spot_close": spot,
                "perp_close": perp,
                "mark_price": [None] * len(times),
                "index_price": [None] * len(times),
                "basis_bps": basis_bps,
                "funding_rate_last": funding_last,
                "funding_rate_next": funding_next,
                "next_funding_time": next_funding_time,
                "minutes_to_next_funding": mins_to_next,
                "expected_edge_bps": expected_edge,
            }
        )

        class StrategyCfg:
            entry_edge_bps = 15.0
            min_basis_bps = 5.0
            exit_edge_bps = 2.0
            exit_basis_bps = 1.0
            hard_stop_basis_widen_bps = 25.0
            max_holding_hours = 48
            min_minutes_to_next_funding = 10

        class CostCfg:
            spot_fee_bps_per_side = 0.0
            perp_fee_bps_per_side = 0.0
            spot_slippage_bps_per_side = 0.0
            perp_slippage_bps_per_side = 0.0

        bt = FundingHarvestBacktester(df=df, strategy_cfg=StrategyCfg, cost_cfg=CostCfg, initial_capital=100000.0)
        res = bt.run()

        self.assertEqual(len(res.trades), 1)
        trade = res.trades[0]
        self.assertEqual(trade.exit_reason, "profit_convergence")
        self.assertEqual(trade.funding_payments_collected, 1)

        # Funding pnl should be notional * 0.0005 at 08:00.
        # notional ~= initial_capital (no costs) and qty based on entry spot 2000.
        notional = trade.qty_eth * trade.entry_spot
        self.assertAlmostEqual(trade.funding_pnl, notional * 0.0005, places=6)
        self.assertAlmostEqual(trade.net_pnl, trade.funding_pnl + trade.basis_pnl - trade.cost_pnl, places=8)


if __name__ == "__main__":
    unittest.main()
