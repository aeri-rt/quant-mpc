# Design Document: System Architecture Overview

## Overview

This document describes the high-level architecture of the MPC-based quantitative trading system for SPY options and equities.

## Goals

- Create a modular, maintainable trading system
- Apply Model Predictive Control (MPC) to optimize delta hedging decisions
- Use Moving Horizon Estimation (MHE) for real-time parameter estimation
- Support backtesting and live trading with minimal code changes
- Enable future strategy extensions (Vol Arb, Regime-Adaptive)

## Non-Goals

- High-frequency trading (sub-second decisions)
- Multi-asset portfolio optimization (initially)
- Market making
- Complex option spreads (initially)

## Background

The system treats trading as an optimal control problem rather than a traditional "predict → trade" pipeline. This allows unified optimization of position management and execution while handling constraints naturally.

### Key Concepts

**Model Predictive Control (MPC):**
- Solves an optimization problem over a finite horizon
- Implements only the first control action
- Re-solves at the next timestep with updated state
- Naturally handles constraints

**Moving Horizon Estimation (MHE):**
- Estimates hidden states and parameters from noisy observations
- Like a Kalman filter but handles nonlinearity and constraints
- Used here for: realized volatility, mean-reversion parameter (κ)

## Design

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           SYSTEM ARCHITECTURE                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │                          DATA LAYER                                   │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  │ │
│  │  │   Market    │  │   Options   │  │   Market    │  │  Database   │  │ │
│  │  │   Data      │  │   Chain     │  │   State     │  │  (SQLite)   │  │ │
│  │  │  Provider   │  │  Manager    │  │  Container  │  │             │  │ │
│  │  │ (IBKR/Poly) │  │             │  │             │  │             │  │ │
│  │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  │ │
│  └─────────┼────────────────┼────────────────┼────────────────┼─────────┘ │
│            │                │                │                │           │
│            └────────────────┴────────┬───────┴────────────────┘           │
│                                      │                                     │
│                                      ▼                                     │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │                         MODELS LAYER                                  │ │
│  │  ┌─────────────────────┐  ┌─────────────────────┐                    │ │
│  │  │   Greeks Engine     │  │   Volatility        │                    │ │
│  │  │   (CasADi symbolic) │  │   Estimator         │                    │ │
│  │  │                     │  │                     │                    │ │
│  │  │  • Black-Scholes    │  │  • Realized vol     │                    │ │
│  │  │  • Portfolio Greeks │  │  • IV extraction    │                    │ │
│  │  │  • Sensitivities    │  │  • Vol forecasting  │                    │ │
│  │  └──────────┬──────────┘  └──────────┬──────────┘                    │ │
│  └─────────────┼────────────────────────┼───────────────────────────────┘ │
│                │                        │                                  │
│                └───────────┬────────────┘                                  │
│                            │                                               │
│                            ▼                                               │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │                        CONTROL LAYER                                  │ │
│  │  ┌─────────────────────────────────────────────────────────────────┐ │ │
│  │  │                    MHE (State Estimator)                        │ │ │
│  │  │  • Realized volatility estimation                               │ │ │
│  │  │  • Mean-reversion parameter κ (from Clinet paper)               │ │ │
│  │  │  • Regime detection (future enhancement)                        │ │ │
│  │  └──────────────────────────┬──────────────────────────────────────┘ │ │
│  │                             │                                        │ │
│  │                             ▼                                        │ │
│  │  ┌─────────────────────────────────────────────────────────────────┐ │ │
│  │  │                    MPC (Controller)                             │ │ │
│  │  │                                                                 │ │ │
│  │  │  State: [price, portfolio_delta, portfolio_gamma, ...]          │ │ │
│  │  │  Control: [delta_hedge_quantity]                                │ │ │
│  │  │                                                                 │ │ │
│  │  │  Objective:                                                     │ │ │
│  │  │    min Σ [ execution_cost + risk_penalty + tracking_error ]     │ │ │
│  │  │                                                                 │ │ │
│  │  │  Constraints:                                                   │ │ │
│  │  │    • Position limits                                            │ │ │
│  │  │    • Delta bounds                                               │ │ │
│  │  │    • Transaction rate limits                                    │ │ │
│  │  │                                                                 │ │ │
│  │  │  Features (from Clinet paper):                                  │ │ │
│  │  │    • "Aggressive in the money" behavior                         │ │ │
│  │  │    • Mean-reversion aware hedging                               │ │ │
│  │  │    • Positivity constraints via MPC (not hard floor)            │ │ │
│  │  └──────────────────────────┬──────────────────────────────────────┘ │ │
│  └─────────────────────────────┼────────────────────────────────────────┘ │
│                                │                                           │
│                                ▼                                           │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │                       STRATEGY LAYER                                  │ │
│  │  ┌─────────────────────┐  ┌─────────────────────┐                    │ │
│  │  │   Instrument        │  │   Gamma Scalping    │                    │ │
│  │  │   Selector          │  │   Strategy          │                    │ │
│  │  │   (Config-driven)   │  │                     │                    │ │
│  │  │                     │  │  • Entry logic      │                    │ │
│  │  │  • ATM straddle     │  │  • Roll logic       │                    │ │
│  │  │  • 30 DTE default   │  │  • Exit logic       │                    │ │
│  │  │  • Roll at 7 DTE    │  │  • MPC integration  │                    │ │
│  │  └──────────┬──────────┘  └──────────┬──────────┘                    │ │
│  └─────────────┼────────────────────────┼───────────────────────────────┘ │
│                │                        │                                  │
│                └───────────┬────────────┘                                  │
│                            │                                               │
│                            ▼                                               │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │                      EXECUTION LAYER                                  │ │
│  │  ┌─────────────────────┐  ┌─────────────────────┐                    │ │
│  │  │   Order Manager     │  │   Broker Adapter    │                    │ │
│  │  │                     │  │   (ib_insync)       │                    │ │
│  │  │  • Order creation   │  │                     │                    │ │
│  │  │  • Fill tracking    │  │  • Connection mgmt  │                    │ │
│  │  │  • Position sync    │  │  • Order submission │                    │ │
│  │  │                     │  │  • Market data      │                    │ │
│  │  └─────────────────────┘  └─────────────────────┘                    │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                        CROSS-CUTTING CONCERNS                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │    Risk      │  │   Logging    │  │    Config    │  │   Events     │   │
│  │   Manager    │  │   (loguru)   │  │   Manager    │  │   System     │   │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Components

#### Data Layer

**MarketDataProvider (Abstract)**
- Interface for market data sources
- Implementations: IBKRDataProvider, PolygonDataProvider
- Responsibilities: Real-time quotes, historical data

**OptionsChainManager**
- Maintains current options chain for SPY
- Calculates Greeks from quotes
- Identifies ATM options, specific strikes

**MarketState**
- Centralized state container
- Current SPY price, option prices, portfolio positions
- Updated by data providers, read by all layers

**Database (SQLite)**
- Trade history, position history
- Performance metrics
- Configuration snapshots

#### Models Layer

**GreeksEngine**
- CasADi symbolic implementation of Black-Scholes
- Computes delta, gamma, vega, theta
- Provides Jacobians/Hessians for optimization

**VolatilityEstimator**
- Realized volatility from price history
- IV extraction from option prices
- Simple vol forecasting (EWMA, GARCH-like)

#### Control Layer

**MHEEstimator (do-mpc)**
- Estimates: realized vol (σ), mean-reversion parameter (κ)
- Horizon: ~50-100 past observations
- Updates: Every N seconds

**MPCController (do-mpc)**
- Horizon: 10-20 steps ahead
- State: [price_deviation, portfolio_delta, portfolio_gamma, ...]
- Control: [hedge_quantity]
- Objective: Minimize execution cost + risk + tracking error
- Constraints: Position limits, delta bounds

#### Strategy Layer

**InstrumentSelector**
- Config-driven rules for option selection
- Selects ATM straddle at target DTE
- Determines roll timing

**GammaScalpingStrategy**
- Orchestrates the gamma scalping logic
- Entry: When vol premium is attractive
- Management: Calls MPC for hedging decisions
- Exit: Roll before expiry, close on vol collapse

#### Execution Layer

**OrderManager**
- Translates strategy decisions to orders
- Tracks order states (pending, filled, cancelled)
- Reconciles positions

**BrokerAdapter (Abstract)**
- Interface for broker operations
- Implementation: IBKRBroker (via ib_insync)
- Handles connection, orders, positions, market data

### Data Flow

```
1. Market Data Arrives (IBKR WebSocket)
   │
   ▼
2. MarketState Updated
   │
   ├──────────────────────────────────┐
   │                                  │
   ▼                                  ▼
3. GreeksEngine Recalculates      4. VolatilityEstimator Updates
   │                                  │
   └──────────────┬───────────────────┘
                  │
                  ▼
5. MHE Updates Parameter Estimates (σ, κ)
   │
   ▼
6. MPC Solves for Optimal Hedge
   │
   ▼
7. Strategy Validates & Confirms
   │
   ▼
8. Risk Manager Approves
   │
   ▼
9. OrderManager Executes
   │
   ▼
10. Broker Submits to Market
```

### Implemented: Greeks-MPC Integration Flow

This is the currently implemented data flow in `GammaScalpingStrategy`:

```
┌─────────────────────────────────────────────────────────────┐
│              GammaScalpingStrategy.update()                 │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Input: spot_price, option_positions, ivs                   │
│            │                                                │
│            ▼                                                │
│  ┌─────────────────────────────────────┐                   │
│  │  PortfolioGreeksCalculator          │                   │
│  │  • Updates underlying prices        │                   │
│  │  • Recalculates per-option Greeks   │                   │
│  │  • Aggregates to portfolio level    │                   │
│  └───────────────┬─────────────────────┘                   │
│                  │                                          │
│                  ▼                                          │
│  ┌─────────────────────────────────────┐                   │
│  │  SimpleKappaEstimator               │                   │
│  │  • Updates EMA of prices            │                   │
│  │  • Estimates S_bar (reference)      │                   │
│  │  • Estimates σ (rolling vol)        │                   │
│  └───────────────┬─────────────────────┘                   │
│                  │                                          │
│                  ▼                                          │
│  ┌─────────────────────────────────────┐                   │
│  │  MPCState Construction              │                   │
│  │  • S = spot_price                   │                   │
│  │  • delta_opt = portfolio delta      │                   │
│  │  • q = current equity hedge         │                   │
│  │  • gamma_opt = portfolio gamma      │                   │
│  │  • sigma, S_bar from estimator      │                   │
│  └───────────────┬─────────────────────┘                   │
│                  │                                          │
│                  ▼                                          │
│  ┌─────────────────────────────────────┐                   │
│  │  DeltaHedgeMPC.solve()              │                   │
│  │  • Sets direction regime bounds     │                   │
│  │  • Solves NLP via IPOPT             │                   │
│  │  • Returns optimal u (trade)        │                   │
│  └───────────────┬─────────────────────┘                   │
│                  │                                          │
│                  ▼                                          │
│  Output: HedgeRecommendation                                │
│  • trade_quantity: shares to trade                          │
│  • trade_direction: BUY/SELL/HOLD                           │
│  • mpc_status: solver status                                │
│  • reason: human-readable explanation                       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Interfaces

#### Key Data Structures

```python
@dataclass
class MarketState:
    timestamp: datetime
    spy_price: float
    spy_bid: float
    spy_ask: float
    options: Dict[str, OptionQuote]  # keyed by option symbol
    positions: Dict[str, Position]
    portfolio_greeks: PortfolioGreeks

@dataclass
class PortfolioGreeks:
    delta: float
    gamma: float
    vega: float
    theta: float
    delta_dollars: float  # delta * spy_price * 100

@dataclass
class MPCOutput:
    hedge_quantity: float  # shares of SPY to trade
    optimal_trajectory: np.ndarray  # future control plan
    solve_time: float
    status: str  # 'optimal', 'infeasible', etc.

@dataclass  
class TradeSignal:
    instrument: str  # 'SPY' or option symbol
    quantity: int
    side: str  # 'BUY' or 'SELL'
    order_type: str  # 'MARKET', 'LIMIT'
    limit_price: Optional[float]
```

## Implementation Plan

### Phase 1: Foundation ✅
- [x] Project structure and documentation
- [x] Core type definitions (`src/core/types.py`)
- [x] CasADi Black-Scholes Greeks (`src/models/black_scholes.py`)
- [x] Portfolio Greeks aggregation (`src/models/greeks.py`)
- [x] Basic MarketState container
- [x] Configuration system (YAML configs)
- [x] Structured logging system (`src/utils/logging.py`)

### Phase 2: MPC Core ✅
- [x] MPC for delta hedging with extended state (`src/control/mpc_controller.py`)
  - State: `[S, Δ_opt, q]` (spot, options delta, equity position)
  - Control: `u = Δq` (shares to trade)
  - Smooth |u| via auxiliary variable z
  - Direction regime constraints
- [x] Simple κ estimator (EMA-based, v1 placeholder)
- [x] Validation tests (synthetic OU, trending, cost sensitivity, constraints)
- [x] Greeks-MPC integration layer (`src/strategy/gamma_scalping.py`)

### Phase 3: Data & Strategy (Current)
- [x] GammaScalpingStrategy implementation
- [ ] **IBKR data connection** ← Current focus
- [ ] Options chain manager
- [ ] Full pipeline integration
- [ ] Paper trading validation

### Phase 4: Hardening
- [ ] Risk management (limits, kill switch)
- [ ] Monitoring and alerting
- [ ] Error handling and recovery
- [ ] Backtest harness
- [ ] Small-size live testing

## Testing Strategy

- **Unit tests:** Greeks calculations, utility functions
- **Integration tests:** MPC solver, broker connection
- **Backtest validation:** Compare to known strategies
- **Paper trading:** Full system on simulated account
- **Shadow mode:** Run live but don't execute (log only)

## Monitoring & Observability

- **Logging:** Structured logs via loguru
- **Metrics:** P&L, Greeks exposure, MPC solve times
- **Alerts:** Position limits, unusual activity, errors
- **Dashboard:** (Future) Real-time visualization

## Open Questions

1. What MPC horizon length is optimal for SPY hedging?
2. How often should MHE update parameter estimates?
3. What's the right balance between hedging cost and delta exposure?
4. How to handle overnight positions (market closed)?

## References

- ADR-001: Project Inception
- Clinet et al. (2021): Optimal trading via MPC
- do-mpc documentation
- CasADi documentation

