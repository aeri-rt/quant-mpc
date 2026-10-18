# Backtest System Implementation Prompt

## Role and Context

You are implementing a backtesting system for an MPC-based gamma scalping trading system at `/Users/rtrinh/Quant/quant-mpc`. The system trades SPY options (long gamma via ATM straddles) and delta-hedges with SPY equity using a Model Predictive Controller. The live/paper trading system is complete and working. You are now building the offline backtesting engine.

## Objective

Build a backtesting system that replays historical intraday SPY price data through the **exact same** `GammaScalpingStrategy` and `DeltaHedgeMPC` used in live trading, with VIX-calibrated synthetic options, proper P&L accounting, and performance reporting. The backtest must run fully offline (no TWS/IBKR connection needed at runtime).

## Architecture

The system has three phases: **data preparation** (fetch + cache), **simulation** (engine loop), and **reporting** (metrics + plots).

```
Data Layer:
  fetch_backtest_data.py  -->  data/cache/*.parquet  (one-time, needs TWS or yfinance)

Engine:
  cached bars + VIX  -->  BacktestEngine  -->  BacktestResult
                              |
                  +-----------+-----------+
                  |           |           |
            VolModel   GammaScalping   PnLEngine
          (VIX-calibrated) (existing)  (new, critical)

Reporting:
  BacktestResult  -->  BacktestReport  -->  plots + console summary
```

## Critical Design Constraints

### 1. The P&L Engine is the centerpiece
The live system delegates P&L to IBKR (`MetricsTracker` explicitly says "P&L tracked by IBKR"). A backtest has no IBKR. You MUST build a standalone P&L engine that tracks:
- **Equity hedge P&L**: Cost basis tracking for SPY shares. When we buy 100 shares at $690 and sell at $692, realized P&L = $200.
- **Options mark-to-market**: Reprice the straddle via Black-Scholes at each bar using current spot, time-to-expiry (which shrinks bar by bar), and current IV. Unrealized P&L = current_value - entry_value.
- **Theta decay**: Computed naturally by repricing with smaller T each bar.
- **Gamma P&L**: The profit from delta-hedging around spot moves. This is realized P&L from equity hedges minus theta decay.
- **Transaction costs**: Commissions + slippage on both equity and option trades.

### 2. Options are stateful positions, not regenerated each bar
When the engine opens an ATM straddle (e.g., $690 strike, 30 DTE, IV=0.18), that position persists across bars. Each bar:
- `T` decreases (time passes)
- `S` changes (spot moves)
- `sigma` may change (if vol model updates)
- The option is repriced with `BlackScholesNumeric.call_price(S, K, T, r, sigma, q)` using the NEW values
- Greeks are recomputed with `BlackScholesNumeric.call_greeks(S, K, T, r, sigma, q)`
Do NOT regenerate new synthetic options every bar. The position object persists; only its market values update.

### 3. `OptionPosition.dte` and `time_to_expiry` use `datetime.now()`
These properties in `src/models/greeks.py` call `datetime.now()` to compute DTE:
```python
@property
def dte(self) -> int:
    now = datetime.now()
    delta = self.expiry - now
    return max(int(delta.total_seconds() / 86400), 0)
```
In a backtest, `datetime.now()` returns wall time, not simulation time. You MUST work around this. Options:
- (Preferred) Compute T directly in the engine: `T = (expiry - sim_time).total_seconds() / (365.25 * 86400)` and pass it to BS functions, bypassing the property.
- Or monkey-patch, but that's fragile.

### 4. `SimpleKappaEstimator` needs warmup
The strategy's kappa estimator uses a rolling window of 60 prices. The first ~60 bars will produce unreliable MPC solutions. Feed a warmup period of bars through the strategy WITHOUT recording metrics. Document this in config as `warmup_bars`.

### 5. Do NOT reuse `MetricsTracker` for the backtest
`MetricsTracker` in `src/utils/metrics.py` hardcodes `datetime.now()` in `record_trade()`, `record_greeks()`, `record_pnl()`, and `get_session_summary()`. Build a new `BacktestMetrics` class that accepts simulation timestamps explicitly. It should compute backtest-specific analytics: Sharpe ratio, annualized return, max drawdown, gamma P&L decomposition.

### 6. Fill simulation must distinguish equity vs options
- **Equity hedge fills** (SPY shares): Fill at mid + configurable slippage (default 1 bps). Commission $0.005/share.
- **Option fills** (straddle entry/exit/roll): Fill at mid + half the bid-ask spread. For synthetic options, model the spread as a function of moneyness and DTE (ATM ~$0.05-$0.10 for SPY). Commission $0.65/contract.

### 7. VIX-to-IV calibration
- VIX represents ~30-day ATM SPX implied vol. Use it as the base IV.
- Apply square-root-of-time term structure scaling: `sigma(T) = VIX * sqrt(30/365) / sqrt(T)` simplified to `sigma = VIX_daily / 100 * adjustment_factor`. Keep this simple but configurable via `vix_scale_factor` in config.
- Intraday: VIX is constant for the day (daily close). This is a known limitation -- document it.

## Existing Components to Reuse (DO NOT MODIFY these files)

### `src/strategy/gamma_scalping.py` -- GammaScalpingStrategy
- `__init__(config: GammaScalpingConfig, mpc_config: Optional[MPCConfig])` 
- `set_option_positions(positions: List[OptionPosition])` -- sets option positions
- `set_equity_position(symbol: str, quantity: int, price: float)` -- sets equity hedge
- `update(spot_price: float, ivs: Optional[dict] = None) -> HedgeRecommendation` -- **main entry point, synchronous, call once per bar**
- `execute_hedge(quantity: float)` -- records that a hedge was executed (updates internal equity position)
- Returns `HedgeRecommendation` with: `trade_quantity`, `trade_direction`, `spot_price`, `portfolio_delta`, `equity_position`, `net_delta`, `portfolio_gamma`, `mpc_status`, `reason`

### `src/control/mpc_controller.py` -- DeltaHedgeMPC
- Called internally by the strategy. Do not call directly.
- `MPCConfig` loaded from `config/mpc_params.yaml`
- `SimpleKappaEstimator` maintains state (price history, EMA). Needs warmup.

### `src/models/black_scholes.py` -- BlackScholesNumeric (static methods)
- `call_price(S, K, T, r, sigma, q=0.0) -> float`
- `put_price(S, K, T, r, sigma, q=0.0) -> float`
- `call_greeks(S, K, T, r, sigma, q) -> OptionGreeks` (returns delta, gamma, vega, theta, rho)
- `put_greeks(S, K, T, r, sigma, q) -> OptionGreeks`
- `gamma(S, K, T, r, sigma, q) -> float`
- All take: S=spot, K=strike, T=time_to_expiry_in_years, r=risk_free_rate, sigma=volatility, q=dividend_yield

### `src/models/greeks.py` -- OptionPosition, EquityPosition, PortfolioGreeks
- `OptionPosition(symbol, position_type, strike, expiry, quantity, multiplier=100, current_price, underlying_price, iv)`
- `EquityPosition(symbol, quantity, current_price)`
- `PositionType.CALL`, `PositionType.PUT`
- Note: `OptionPosition.dte` and `time_to_expiry` use `datetime.now()` -- see constraint #3.

### `src/strategy/position_manager.py` -- PositionManager
- `check_positions(positions: List[OptionPosition]) -> List[PositionSignal]`
- `should_roll(position) -> (bool, reason)` -- checks DTE threshold
- `should_exit(position) -> (bool, action, reason)` -- checks P&L thresholds
- `set_entry_price(symbol, price)`, `clear_all_positions()`
- `PositionManagerConfig(roll_dte=7, roll_to_dte=30, take_profit_pct=0.50, stop_loss_pct=-0.30)`
- Note: `should_roll` uses `position.dte` which has the datetime.now() issue.

### `src/core/types.py` -- Quote, OptionQuote, OptionType
- `Quote(bid, ask, bid_size, ask_size, last, last_size, timestamp)` -- frozen dataclass
- `OptionQuote(symbol, underlying, option_type, strike, expiry, bid, ask, last, volume, open_interest, timestamp, delta, gamma, vega, theta, iv)` -- frozen dataclass

### Config files
- `config/mpc_params.yaml` -- MPC weights, constraints, horizon
- `config/strategy_params.yaml` -- Strategy params, risk limits, position sizing
- `config/settings.yaml` -- Broker, instrument, risk-free rate, dividend yield

### Dependencies already installed
`numpy`, `scipy`, `pandas`, `matplotlib` (dev), `casadi`, `do-mpc`, `loguru`
**Add**: `yfinance` for VIX data download.

## Files to Create

| File | Purpose |
|------|---------|
| `backtest/__init__.py` | Export BacktestEngine, BacktestConfig, BacktestResult |
| `backtest/config.py` | `BacktestConfig` dataclass with all parameters |
| `backtest/data_fetcher.py` | Download SPY bars (IBKR) + VIX daily (yfinance), cache to parquet |
| `backtest/vol_model.py` | VIX-calibrated IV model with term structure scaling |
| `backtest/synthetic_options.py` | Create and maintain stateful synthetic OptionPosition objects |
| `backtest/fill_simulator.py` | Simulate fills with slippage + spread + commissions (equity vs options) |
| `backtest/pnl_engine.py` | **Core**: Mark-to-market, realized P&L, cost basis, P&L decomposition |
| `backtest/engine.py` | Main simulation loop orchestrating all components |
| `backtest/metrics.py` | `BacktestMetrics` -- sim-clock-aware, Sharpe, drawdown, decomposition |
| `backtest/report.py` | Generate matplotlib plots + console summary |
| `scripts/fetch_backtest_data.py` | CLI to download and cache data (needs TWS for intraday) |
| `scripts/run_backtest.py` | Replace existing stub with working runner |
| `config/backtest.yaml` | Default backtest parameters |
| `tests/test_backtest.py` | Unit tests for P&L engine, fill sim, synthetic options, engine |

## Files to Edit

| File | Change |
|------|--------|
| `scripts/run_backtest.py` | Replace placeholder with working backtest runner |
| `backtest/__init__.py` | Add exports |

## Implementation Order

1. **`backtest/config.py`** -- Define `BacktestConfig` first so everything else can reference it.
2. **`backtest/vol_model.py`** -- Simple VIX-to-IV mapping. Needed by synthetic options.
3. **`backtest/synthetic_options.py`** -- Stateful option positions. Depends on vol model + BS.
4. **`backtest/fill_simulator.py`** -- Fill logic. Independent module.
5. **`backtest/pnl_engine.py`** -- The critical P&L tracker. Depends on fill simulator + BS for mark-to-market.
6. **`backtest/metrics.py`** -- Aggregates P&L engine output into analytics.
7. **`backtest/engine.py`** -- Main loop wiring everything together with existing strategy.
8. **`backtest/report.py`** -- Visualization. Last because it only reads results.
9. **`backtest/data_fetcher.py`** -- Data download utility.
10. **`scripts/fetch_backtest_data.py`** and **`scripts/run_backtest.py`** -- CLI scripts.
11. **`config/backtest.yaml`** -- Default config file.
12. **`tests/test_backtest.py`** -- Tests for all new components.

## BacktestConfig Specification

```python
@dataclass
class BacktestConfig:
    # Time range
    start_date: str = "2025-12-01"
    end_date: str = "2026-01-31"
    bar_size_minutes: int = 5

    # Warmup
    warmup_bars: int = 60  # Bars to feed before recording metrics

    # Capital and sizing
    initial_capital: float = 100_000.0
    position_size: int = 10       # Contracts per straddle leg
    target_dte: int = 30          # Target DTE for new straddles
    
    # Position management
    roll_dte: int = 7             # Roll when DTE falls below
    roll_to_dte: int = 30         # Roll into this DTE
    take_profit_pct: float = 0.50
    stop_loss_pct: float = -0.30

    # Volatility model
    vix_scale_factor: float = 1.0   # Multiplier on VIX -> BS sigma
    
    # Market parameters
    risk_free_rate: float = 0.045
    dividend_yield: float = 0.0106

    # Transaction costs
    equity_slippage_bps: float = 1.0
    equity_commission_per_share: float = 0.005
    option_spread_dollars: float = 0.05   # Half-spread for ATM SPY options
    option_commission_per_contract: float = 0.65

    # MPC config (reuse from yaml or override)
    mpc_horizon: int = 20
    mpc_q_max: float = 1000.0
    mpc_u_max: float = 200.0
    mpc_w_delta: float = 10.0

    # Data paths
    cache_dir: str = "data/cache"
    output_dir: str = "data/backtest_results"
```

## Engine Main Loop Pseudocode

```python
def run(self, bars: pd.DataFrame, vix: pd.DataFrame) -> BacktestResult:
    # Initialize strategy (reuses existing GammaScalpingStrategy)
    strategy = GammaScalpingStrategy(gamma_config, mpc_config)
    position_mgr = PositionManager(pos_config)
    pnl = PnLEngine(config)
    metrics = BacktestMetrics()
    
    # State
    option_positions = []
    equity_qty = 0
    sim_time = bars.index[0]
    
    for i, (timestamp, bar) in enumerate(bars.iterrows()):
        spot = bar['close']
        sim_time = timestamp
        
        # Get today's VIX -> IV
        iv = vol_model.get_iv(sim_time, config.target_dte)
        
        # --- WARMUP: feed price, skip everything else ---
        if i < config.warmup_bars:
            strategy.update(spot_price=spot)
            continue
        
        # --- POSITION LIFECYCLE ---
        # If no positions, open initial straddle
        if not option_positions:
            option_positions = open_straddle(spot, iv, sim_time, config.target_dte)
            entry_cost = fill_sim.option_fill(option_positions, spot, iv)
            pnl.record_option_entry(option_positions, entry_cost, sim_time)
            strategy.set_option_positions(option_positions)
            strategy.set_equity_position("SPY", equity_qty, spot)
            position_mgr.set_entry_price(...)
            continue
        
        # Update option values (reprice with new S, T, sigma)
        T = compute_T(option_positions[0].expiry, sim_time)
        for pos in option_positions:
            pos.current_price = bs_price(spot, pos.strike, T, r, iv, q)
            pos.underlying_price = spot
            pos.iv = iv
        
        # Check for roll/exit signals
        # (Cannot use position_mgr.should_roll directly due to datetime.now issue)
        dte = compute_dte(option_positions[0].expiry, sim_time)
        if dte <= config.roll_dte:
            # Close old straddle, compute realized P&L
            close_cost = fill_sim.option_fill(...)
            pnl.record_option_exit(option_positions, close_cost, sim_time)
            # Open new straddle
            option_positions = open_straddle(spot, iv, sim_time, config.roll_to_dte)
            entry_cost = fill_sim.option_fill(...)
            pnl.record_option_entry(option_positions, entry_cost, sim_time)
            strategy.set_option_positions(option_positions)
        
        # --- HEDGING ---
        strategy.set_option_positions(option_positions)
        strategy.set_equity_position("SPY", equity_qty, spot)
        rec = strategy.update(spot_price=spot)
        
        if rec.trade_quantity != 0:
            fill_price = fill_sim.equity_fill(spot, rec.trade_quantity)
            commission = fill_sim.equity_commission(rec.trade_quantity)
            pnl.record_equity_trade(rec.trade_quantity, fill_price, commission, sim_time)
            equity_qty += int(rec.trade_quantity)
            strategy.execute_hedge(rec.trade_quantity)
        
        # --- MARK TO MARKET ---
        option_mtm = sum(pos.current_price * pos.quantity * pos.multiplier for pos in option_positions)
        equity_mtm = equity_qty * spot
        pnl.mark_to_market(option_mtm, equity_mtm, sim_time)
        
        # --- RECORD METRICS ---
        metrics.record(sim_time, spot, rec, pnl.snapshot(), option_positions)
    
    return BacktestResult(metrics=metrics, pnl=pnl, config=config)
```

## PnL Engine Specification

The P&L engine must track:
```
Total P&L = Equity Realized P&L + Option Realized P&L + Option Unrealized P&L - Total Costs

Where:
- Equity Realized P&L: Sum of (sell_price - buy_price) * shares for closed equity trades
- Option Realized P&L: Sum of (exit_value - entry_value) * contracts * multiplier for closed straddles
- Option Unrealized P&L: current_mtm - entry_value for open straddle
- Total Costs: All commissions + slippage
```

It should expose:
- `snapshot() -> PnLSnapshot` (current state at any point)
- `record_equity_trade(qty, price, commission, timestamp)`
- `record_option_entry(positions, cost, timestamp)`
- `record_option_exit(positions, proceeds, timestamp)`
- `mark_to_market(option_value, equity_value, timestamp)`
- `get_equity_curve() -> pd.Series` (timestamped total P&L)
- `get_decomposition() -> dict` with keys: `equity_pnl`, `option_pnl`, `theta_pnl`, `gamma_pnl`, `costs`

## Report Output

The report should generate these plots (saved to `data/backtest_results/`):
1. **Equity curve**: Cumulative P&L over time
2. **P&L decomposition**: Stacked area of gamma P&L, theta decay, transaction costs
3. **Delta exposure**: Time series of net portfolio delta (should stay near zero)
4. **Spot price with hedge trades**: SPY price with buy/sell markers
5. **Drawdown chart**: Running max drawdown over time

Console summary:
```
Backtest Results: 2025-12-01 to 2026-01-31
Duration: 42 trading days | Bars: 3,276
Total P&L: $X,XXX | Sharpe: X.XX | Max DD: $X,XXX (X.X%)
Equity hedge P&L: $X,XXX | Option P&L: $X,XXX | Costs: $X,XXX
Trades: XXX | Avg size: XX shares | Rolls: X
Avg |delta|: X.X | Max |delta|: XX.X | % time hedged (<50 delta): XX%
```

## Testing Requirements

`tests/test_backtest.py` must cover:
1. **PnL engine**: Buy 100 shares at $690, sell at $692 -> realized P&L = $200 minus commissions
2. **PnL engine**: Open straddle, reprice after spot move, verify unrealized P&L matches BS
3. **Fill simulator**: Equity fill at mid + slippage, option fill at mid + half-spread
4. **Synthetic options**: Create ATM straddle, verify call+put deltas sum near zero
5. **Vol model**: VIX=20 -> IV ~0.20 (with configurable scaling)
6. **Engine integration**: Run 100 bars of synthetic data, verify no crashes, P&L is computed

## Important Notes

- Do NOT modify any existing files in `src/`. The backtest system is entirely additive.
- The only existing file to edit is `scripts/run_backtest.py` (replace the stub).
- Add `yfinance` to `pyproject.toml` dependencies.
- All new code goes in `backtest/` and `scripts/`.
- Use `loguru` for logging (import from `src.utils.logging import get_logger`).
- Use `pandas` DataFrames for time series data.
- Use `matplotlib` for plotting (it's in dev dependencies).
- Run `./venv/bin/python -m pytest tests/test_backtest.py -v` to verify tests pass.
