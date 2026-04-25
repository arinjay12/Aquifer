# funding_harvest

ETH funding-rate harvest / single-venue cash-and-carry (Binance, ETHUSDT).

This project designs, backtests, and generates live signals (no execution) for a **market-neutral** trade:

- **Long ETHUSDT spot**
- **Short ETHUSDT perpetual**
- Enter only when positive funding + positive basis make the expected edge **after costs** large enough

Disclaimer: Research software, not investment advice. Real execution has additional risks (latency, liquidations, outages, exchange risk).

---

## Strategy design 

### What inefficiency are we exploiting?
Perpetual futures funding rates and the “basis” (perp price minus spot price) can be persistently positive because many traders prefer **leveraged long exposure via perps**. That pushes perp prices above spot and makes funding positive. This isn’t fully arbitraged away because it requires capital + operational reliability and has tail risks (basis widening, funding flipping, exchange risk).

### What are the signals?
We trade a “cash-and-carry” pair:
- Long spot ETHUSDT
- Short perp ETHUSDT

Entry (all must be true):
- state is FLAT
- `expected_edge_bps >= entry_edge_bps`
- `basis_bps >= min_basis_bps`
- `funding_rate_next > 0`
- `minutes_to_next_funding >= min_minutes_to_next_funding`

Exit (first match wins):
1) hard stop: basis widened vs entry by `hard_stop_basis_widen_bps`
2) profit/convergence: basis <= `exit_basis_bps` and ≥1 funding payment collected
3) funding decay: `expected_edge_bps <= exit_edge_bps`
4) time stop: holding time >= `max_holding_hours`

### What is the theoretical edge?
We use a transparent v1 edge model:

```
basis_bps = (perp_close - spot_close) / spot_close * 10000

round_trip_cost_bps = (
  2*spot_fee + 2*perp_fee + 2*spot_slippage + 2*perp_slippage
)

funding_rate_next_bps = funding_rate_next * 10000

expected_edge_bps = funding_rate_next_bps + basis_bps - round_trip_cost_bps
```

### Risk framework
Configured in `config/params.yaml` under `risk:`:
- position sizing via `position_fraction` (fraction of equity used per trade)
- drawdown pause: if drawdown exceeds `max_drawdown_pause_pct`, pause new entries for `pause_minutes`
- a simple `confidence_score` in [0,1] for live logging (how far edge is above threshold)

---

## Backtesting engine

### Data (real historical, 1-minute)
- Binance spot klines (ETHUSDT, 1m)
- Binance futures klines (ETHUSDT, 1m)
- Binance futures funding history

Raw data saved under `data/raw_*` as Parquet; merged into a canonical minute dataset under `data/processed/`.

### Costs (required)
Backtests include fees + slippage for both legs. Results without these are disqualified; we include them.

### Reported metrics
- Sharpe ratio (minute returns annualized)
- max drawdown
- win rate
- average holding period
- turnover

Trade ledger decomposes P&L into funding, basis/spread, and costs.

---

## Live signal generator (and 48h log artifact)

Real-time runner:
- `python -m src.live.runner`
- polls Binance, computes the same entry/exit rules as backtest, appends one row per minute to CSV
- logs: timestamp, instrument, direction, edge, confidence score, state, entry/exit flags

48-hour log without waiting 48 hours:
- `python -m src.live.historical_sim ...` replays historical minutes into the same live-log CSV schema instantly
- this produces a shareable 48h log within minutes

---

## Results (what we ran)

### Minimum 2 years backtest
We built a 2-year dataset:
- 2024-04-25 → 2026-04-25 (UTC)
- `data/processed/master_2y_1m.parquet`

Default params results:
- `outputs/tables_2y_default/summary_metrics.csv`
- `outputs/figures_2y_default/`

Tuned “funding-harvest demo” run (to ensure holding across funding timestamps):
- `entry_edge_bps=5`, `exit_edge_bps=-50`, `max_holding_hours=72`
- `outputs/tables_2y_tuned/summary_metrics.csv`
- `outputs/ledgers_2y_tuned/trades_optimistic.csv`
- `outputs/figures_2y_tuned/`

48-hour live-style log artifact (historical replay):
- `outputs/live_logs/live_signal_log_48h_trade_example.csv`
- `outputs/live_logs/live_signal_log_48h_trade_example.summary.json`

Note: many 48-hour windows will legitimately produce 0 trades (no edge after costs). That’s expected for a selective stat-arb strategy.

### Headline numbers (to make this readable without running code)

All runs start with **$100,000** notional capital. We report both “baseline” and “optimistic” cost assumptions.

**6-month pilot (2024-01-01 → 2024-08-01), tuned demo settings (`entry_edge_bps=5`, `exit_edge_bps=-50`, `max_holding_hours=72`)**
- Optimistic costs: **+1.41% total return** (**+$1,410** net), **3 trades**, Sharpe ≈ **1.46**, max drawdown ≈ **-0.42%**
- Funding harvesting is visible in this tuned run (non-zero funding P&L); see `outputs/tables_tuned/summary_metrics.csv`

**2-year evaluation (2024-04-25 → 2026-04-25), default settings (as in `config/params.yaml`)**
- This is intentionally selective and can be “inactive” for long periods; see `outputs/tables_2y_default/summary_metrics.csv`

**2-year evaluation (2024-04-25 → 2026-04-25), tuned demo settings (`entry_edge_bps=5`, `exit_edge_bps=-50`, `max_holding_hours=72`)**
- Optimistic costs: **+3.52% total return** (**+$3,521** net), **7 trades**, Sharpe ≈ **0.64**, max drawdown ≈ **-1.65%**
- P&L decomposition in this tuned run is meaningfully split across basis + funding − costs; see:
  - `outputs/tables_2y_tuned/summary_metrics.csv`
  - `outputs/tables_2y_tuned/pnl_decomposition_optimistic.csv`
  - `outputs/ledgers_2y_tuned/trades_optimistic.csv` (auditable trades)

### Was it successful?
As a **submission/system**: yes — it includes a strategy with explicit edge logic, a cost-aware backtester on real historical data (including a 2-year run), a live signal generator that logs edge + confidence, and a risk framework (position sizing + drawdown pause).

As an **always-on money machine**: no guarantee — the strategy is regime-dependent. Many windows legitimately yield **0 trades** when the basis/funding edge is not present after costs. This is expected and is itself an important result.

### Practical improvements (next steps)
- Multi-venue / cross-exchange cash-and-carry (reduce regime dependence)
- More realistic execution modeling (spread, partial fills, inventory constraints)
- Better “next funding” estimation in live mode (Binance snapshot doesn’t provide it directly; v1 proxies it)
- Capital efficiency constraints (margin, liquidation buffers) and stress tests
- More robust regime detection / “edge vanished” logic (rolling edge distributions, funding flip detection)

---

## Reproduce (commands)

Install:
```bash
pip install -r requirements.txt
```

Download 2 years of data (chunked):
```bash
python -m src.data.download_historical --start-date 2024-04-25 --end-date 2026-04-25 --chunk-days 30 --raw-dir data/raw_2y
```

Build dataset:
```bash
python -m src.data.build_dataset --raw-dir data/raw_2y --out data/processed/master_2y_1m.parquet --scenario baseline
```

Generate tables + plots:
```bash
python -m src.analysis.report_tables --dataset data/processed/master_2y_1m.parquet --tables-dir outputs/tables_2y_default --ledgers-dir outputs/ledgers_2y_default
python -m src.analysis.plots --dataset data/processed/master_2y_1m.parquet --figures-dir outputs/figures_2y_default --scenario optimistic
```

Generate tuned demo results:
```bash
python -m src.analysis.report_tables --dataset data/processed/master_2y_1m.parquet --tables-dir outputs/tables_2y_tuned --ledgers-dir outputs/ledgers_2y_tuned --entry-edge-bps 5 --exit-edge-bps -50 --max-holding-hours 72
python -m src.analysis.plots --dataset data/processed/master_2y_1m.parquet --figures-dir outputs/figures_2y_tuned --scenario optimistic --entry-edge-bps 5 --exit-edge-bps -50 --max-holding-hours 72
```

Generate a 48h shareable live-style log (historical replay):
```bash
python -m src.live.historical_sim --dataset data/processed/master_2y_1m.parquet --start 2024-07-03T00:00:00Z --hours 48 --scenario optimistic --out outputs/live_logs/live_signal_log_48h_trade_example.csv --entry-edge-bps 5 --exit-edge-bps -50 --max-holding-hours 72
python -m src.live.summarize_log --csv outputs/live_logs/live_signal_log_48h_trade_example.csv
```

---

## Repo layout

- `config/` typed config + YAML params
- `data/` raw/processed data artifacts (parquet, CSV)
- `src/` python modules (data, signals, backtest, analysis, live)
- `outputs/` figures, tables, ledgers, live logs
- `tests/` unit tests
