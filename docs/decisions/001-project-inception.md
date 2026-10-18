# ADR-001: Project Inception and Strategic Direction

## Status
Accepted (historical). Superseded by [ADR-004](004-v2-thesis-and-hedger.md) (2026-10-17): the
do-mpc/CasADi/IPOPT controller, MHE and `ib_insync` choices below were replaced by a
Whalley–Wilmott band, a closed-form QP with no solver, and `ib_async`. The body is kept as written.

## Date
2025-12-31

## Context

This document captures the foundational decisions made during the initial brainstorming and planning phase for a quantitative trading system. The goal is to create a systematic, algorithmic trading system that generates alpha in the options and equities markets, specifically targeting SPY (S&P 500 ETF).

### The Retail Quant Challenge

Retail traders face structural disadvantages:
- No co-located servers or ultra-low latency infrastructure
- No prime brokerage access
- Limited capital (capacity-constrained strategies become accessible)
- No proprietary order flow data

However, retail can compete in:
- Capacity-constrained strategies (sub-$10M AUM)
- Longer holding periods (daily/weekly signals)
- Behavioral/sentiment anomalies
- Factor timing and momentum strategies
- Volatility risk premium harvesting

### Why Control Theory?

Most quantitative trading approaches follow a "Signal → Position → Execute" paradigm with each stage optimized separately. This project takes a different approach: treating trading as an **optimal control problem**.

Model Predictive Control (MPC) and Moving Horizon Estimation (MHE) offer:
- Unified optimization of signal exploitation, position management, and execution
- Built-in constraint handling (margin, Greeks limits, position limits)
- Receding horizon optimization (continuously re-optimizing with new information)
- Robustness to model uncertainty
- State estimation when direct observation isn't possible

This approach is inspired by the paper "Optimal trading: a model predictive control approach" (Clinet, Perreton, Reydellet, 2021), which applies MPC to optimal execution with mean-reversion signals.

## Decision

### 1. Target Asset: SPY Only (Initially)

**Rationale:**
- Most liquid options market globally
- Tight bid-ask spreads ($0.01-0.02)
- Full expiration spectrum (0DTE to LEAPS)
- Well-studied dynamics with academic benchmarks
- VIX provides direct regime signal

**Future Extension:** Other liquid ETFs (QQQ, IWM, TLT) once foundation is proven.

### 2. Primary Strategy: Gamma Scalping with Predictive Rebalancing (Concept B)

**What it is:**
- Buy options (long gamma exposure)
- Profit when realized volatility exceeds implied volatility
- Delta hedge dynamically to capture gamma P&L

**Why this strategy:**
- Clear feedback loop and well-documented academically
- MPC value-add is obvious (optimizing *when* and *how much* to hedge)
- The paper's "aggressive in the money" concept directly applies
- Can evolve into more sophisticated strategies

**MPC Application:**
- Traditional gamma scalping hedges at fixed intervals or delta thresholds
- MPC optimizes hedging based on estimated mean-reversion parameter (κ)
- When mean-reversion is strong: hedge less frequently (price reverts)
- When momentum is strong: hedge more aggressively

### 3. Instrument Selection: Simple Rules, Not Optimization

For Concept B (Gamma Scalping), instrument selection is straightforward:
- ATM straddle/strangle
- ~30 DTE expiry (balancing theta decay vs gamma)
- Roll around 7-14 DTE before gamma collapses

**Decision:** Use config-driven rules for instrument selection. MPC focuses entirely on the hedging problem. Sophisticated instrument selection will be added for future strategy extensions (Vol Arb, Regime-Adaptive).

### 4. Technology Stack

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Language | Python 3.11+ | All core libraries are Python |
| MPC/MHE Framework | do-mpc | Purpose-built, well-documented |
| Symbolic Math | CasADi | Auto-differentiation, NLP solvers |
| NLP Solver | IPOPT | Industry standard for nonlinear |
| Broker | Interactive Brokers (via ib_insync) | Gold standard, already have account |
| Data (Live) | IBKR market data | Included with account |
| Data (Historical) | Polygon.io ($199/mo) | Comprehensive options data |
| Database | SQLite initially | Simple, scales to PostgreSQL if needed |
| Configuration | YAML + Pydantic | Type-safe, human-readable |

### 5. Architecture: Layered, Modular Design

```
Data Layer → Models Layer → Control Layer → Strategy Layer → Execution Layer
```

Key design principles:
- Separation of concerns (each layer has single responsibility)
- Broker abstraction (can swap IBKR for another)
- Data provider abstraction (can swap Polygon for another)
- Config-driven (parameters in YAML, not hardcoded)

### 6. Risk Tolerance

High risk tolerance accepted, with appropriate safeguards:
- Position limits in config
- Greeks limits (max delta, gamma, vega exposure)
- Kill switch capability
- Margin monitoring

### 7. Timeline

- **Functional system:** ~2 months
- **Active trading:** 3-4 months
- Approach: Quality over speed, no rushing

## Consequences

### Positive
- Control-theoretic approach is differentiated from typical quant methods
- SPY focus keeps complexity manageable
- Gamma scalping is well-understood with clear benchmarks
- do-mpc/CasADi provide solid mathematical foundation
- Modular design allows incremental enhancement

### Negative
- Control theory has a learning curve
- MPC computational cost may limit update frequency
- Single-asset (SPY) limits diversification initially

### Risks
- Model risk: MPC is only as good as the dynamics model
- Overfitting: Many parameters to tune
- Execution slippage may erode gamma profits

## Alternatives Considered

### Alternative 1: Traditional ML-Based Approach
**Description:** Use ML for signal generation, simple rules for execution

**Why not chosen:** Wanted a more principled, unified framework. ML can be added later as a signal input to MPC.

### Alternative 2: Multiple Assets from Start
**Description:** Trade SPY, QQQ, IWM simultaneously

**Why not chosen:** Increases complexity. Better to prove approach on one asset first.

### Alternative 3: Start with Vol Selling
**Description:** Begin with short volatility strategies (premium harvesting)

**Why not chosen:** Higher tail risk. Gamma scalping is more controlled for learning the system.

### Alternative 4: Pure Alpaca for Simplicity
**Description:** Use Alpaca instead of IBKR for simpler API

**Why not chosen:** Already have IBKR account. ib_insync makes IBKR manageable. No migration later.

## References

- Clinet, S., Perreton, J.F., Reydellet, S. (2021). "Optimal trading: a model predictive control approach." arXiv:2110.11008v2
- do-mpc documentation: https://www.do-mpc.com/en/latest/
- CasADi documentation: https://web.casadi.org/
- ib_insync documentation: https://ib-insync.readthedocs.io/

