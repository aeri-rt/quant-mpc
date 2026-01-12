# ADR 002: Greeks Engine and MPC Controller Integration

## Status
Accepted

## Date
2026-01-04

## Context

We have developed two core components independently:

1. **Greeks Engine** (`src/models/greeks.py`)
   - `PortfolioGreeksCalculator`: Aggregates Greeks across option and equity positions
   - Uses `BlackScholesNumeric` for per-option calculations
   - Outputs `PortfolioGreeks` with delta, gamma, vega, theta

2. **MPC Controller** (`src/control/mpc_controller.py`)
   - `DeltaHedgeMPC`: Optimizes hedge trajectory over a rolling horizon
   - Expects `MPCState` with current delta_opt, gamma_opt, and equity position
   - Outputs `MPCSolution` with recommended trade

These components need to be integrated into a cohesive strategy that:
- Takes market data (spot price, positions) as input
- Computes portfolio Greeks in real-time
- Feeds Greeks to MPC controller
- Produces actionable hedge recommendations

## Decision

### Integration Architecture

Create a `GammaScalpingStrategy` class in `src/strategy/gamma_scalping.py` that:

1. **Orchestrates the workflow:**
   ```
   MarketState → GreeksCalculator → MPCState → MPC → HedgeRecommendation
   ```

2. **Maintains state:**
   - Current option positions (with IV, expiry, strike)
   - Current equity hedge position (q)
   - EMA of spot price (for S_bar)
   - Rolling volatility estimate

3. **Provides a clean interface:**
   ```python
   strategy = GammaScalpingStrategy(config)
   strategy.update_positions(options, equity)
   recommendation = strategy.get_hedge_recommendation(spot_price)
   ```

### Data Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    GammaScalpingStrategy                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  MarketData ──→ PortfolioGreeksCalculator                   │
│  (spot, IV)            │                                    │
│                        ▼                                    │
│                 PortfolioGreeks                             │
│                 (delta, gamma)                              │
│                        │                                    │
│                        ▼                                    │
│  EquityPosition ──→ MPCState ──→ DeltaHedgeMPC             │
│  (q, S_bar, σ)         │                 │                  │
│                        │                 ▼                  │
│                        │         MPCSolution                │
│                        │         (u_optimal)                │
│                        │                 │                  │
│                        ▼                 ▼                  │
│                 HedgeRecommendation                         │
│                 (trade_quantity, reason)                    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Key Design Choices

1. **Separation of Options Greeks vs Equity**
   - `delta_opt` from options only (not including equity hedge)
   - `q` tracks equity position separately
   - `delta_portfolio = delta_opt + q` computed within MPC

2. **Real-time Gamma Updates**
   - Gamma changes as spot moves
   - Strategy recalculates Greeks on each update
   - Passes fresh `gamma_opt` to MPC at each solve

3. **EMA for Reference Price**
   - `S_bar` computed as EMA of spot prices
   - Configurable span (default 20)
   - Updated incrementally on each new price

4. **Volatility Estimation**
   - Simple rolling realized volatility for v1
   - Can be upgraded to MHE in v2

## Consequences

### Positive
- Clean separation between Greeks calculation and hedging optimization
- Each component can be tested independently
- Strategy layer is easy to extend with additional logic
- Logging captures full state at each step

### Negative
- Additional abstraction layer adds some overhead
- Need to ensure Greeks calculator and MPC use consistent parameters

### Risks
- Greeks calculation assumes Black-Scholes (European options)
- MPC may recommend trades that are impractical to execute
- IV from market data may be stale or unavailable

## Implementation Notes

- Strategy lives in `src/strategy/gamma_scalping.py`
- Uses existing `PortfolioGreeksCalculator` from `src/models/greeks.py`
- Uses existing `DeltaHedgeMPC` from `src/control/mpc_controller.py`
- Logging via `src/utils/logging.py` for full audit trail

