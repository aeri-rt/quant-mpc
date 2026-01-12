# Design Document: MPC Controller for Delta Hedging

## Overview

This document describes the design of a Model Predictive Control (MPC) system for optimal delta hedging in a gamma scalping options strategy. The goal is to optimize **when** and **how much** to hedge, rather than using simple threshold-based rules.

### Context

- **Strategy:** Gamma scalping on SPY options (long straddles)
- **Problem:** Delta changes as underlying moves; we must hedge to remain neutral
- **Traditional approach:** Hedge immediately when delta exceeds threshold
- **MPC approach:** Optimize hedge trajectory considering market dynamics, costs, and risk

### References

- Clinet, Perreton, Reydellet (2021): "Optimal trading: a model predictive control approach" (arXiv:2110.11008v2)
- do-mpc documentation: https://www.do-mpc.com/en/latest/
- CasADi documentation: https://web.casadi.org/

---

## Problem Statement

### The Core Question

Given a long gamma position (straddle), the underlying price moves, and our portfolio delta deviates from zero. We need to decide:

1. **When** to hedge (immediately? wait for mean-reversion?)
2. **How much** to hedge (full delta? partial?)
3. **How fast** to hedge (all at once? spread over time?)

### Traditional vs MPC Approach

| Aspect | Traditional | MPC |
|--------|-------------|-----|
| Decision rule | Fixed threshold or interval | Optimized trajectory |
| Market dynamics | Ignored | Modeled (mean-reversion, vol) |
| Transaction costs | Not optimized | Explicitly minimized |
| Look-ahead | None (reactive) | N-step horizon (predictive) |
| Adaptability | Static rules | Re-optimizes each step |

---

## Mathematical Formulation

### State Vector

**Option 1: Simple (Recommended for v1)**
```
x = [S, Δ]
```
- `S`: Spot price of underlying (SPY)
- `Δ`: Portfolio delta (in share equivalents)

**Option 2: Extended**
```
x = [S, Δ_options, q_equity]
```
- `S`: Spot price
- `Δ_options`: Delta from option positions only
- `q_equity`: Current equity hedge position (shares)

Note: In Option 1, `Δ` = `Δ_options + q_equity` (combined).

### Control Vector

```
u = [Δq]
```
- `Δq`: Change in equity position (shares of SPY to trade)
- Positive = buy shares, Negative = sell shares

### Dynamics Model

#### Price Dynamics (Ornstein-Uhlenbeck / Mean-Reversion)

From Clinet et al., price follows mean-reverting dynamics:

```
S_{t+1} = S_t + κ(S̄ - S_t)Δt + σ√Δt · ε_t
```

Where:
- `κ`: Mean-reversion speed (higher = faster reversion)
- `S̄`: Reference price (equilibrium level, e.g., EMA)
- `σ`: Volatility (annualized)
- `Δt`: Time step (e.g., 1/252/6.5/60 for 1-minute)
- `ε_t`: Standard normal random variable (treated deterministically in MPC)

For deterministic MPC, we use the expected dynamics:
```
S_{t+1} = S_t + κ(S̄ - S_t)Δt
```

#### Delta Dynamics

Portfolio delta changes due to:
1. **Gamma effect:** Price movement changes option delta
2. **Hedge trades:** Our control action

```
Δ_{t+1} = Δ_t + Γ · (S_{t+1} - S_t) + u_t
```

Where:
- `Γ`: Portfolio gamma (assumed constant over horizon, updated each MPC solve)
- `u_t`: Hedge trade (positive = buy shares = increases portfolio delta)

Note: Buying shares when we're long delta increases total delta; buying when short delta reduces the magnitude. The sign convention assumes `u_t` directly adds to portfolio delta.

### Objective Function

The MPC minimizes a cost function over the prediction horizon N:

```
J = Σ_{t=0}^{N-1} [ L_stage(x_t, u_t) ] + L_terminal(x_N)
```

#### Stage Cost Components

**1. Execution Cost (Transaction Costs + Market Impact)**
```
L_exec = c_linear · |u_t| + c_quad · u_t²
```
- `c_linear`: Linear cost (commissions, half-spread)
- `c_quad`: Quadratic cost (market impact)

**2. Risk Penalty (Delta Exposure)**
```
L_risk = w_Δ · Δ_t²
```
- `w_Δ`: Weight on delta exposure (higher = more aggressive hedging)

**3. Trading Rate Penalty (Smoothness)**
```
L_smooth = w_rate · (u_t - u_{t-1})²
```
- `w_rate`: Weight on trading smoothness

#### Terminal Cost
```
L_terminal = w_T · Δ_N²
```
- `w_T`: Terminal penalty (want delta near zero at horizon end)

#### Total Stage Cost
```
L_stage = w_exec · (c_lin·|u| + c_quad·u²) + w_Δ · Δ² + w_rate · (u - u_prev)²
```

### Constraints

#### Hard Constraints

**Position Limits:**
```
|q_equity| ≤ q_max
```
Maximum equity position (e.g., 1000 shares)

**Trade Size Limits:**
```
|u_t| ≤ u_max
```
Maximum trade per step (e.g., 200 shares)

**Delta Bounds:**
```
|Δ_portfolio| ≤ Δ_max
```
Maximum allowed delta exposure (e.g., 500 shares equivalent)

#### Soft Constraints (via Penalties)
- Target delta = 0 (encoded in objective)
- Smooth trading (encoded in objective)

### Positivity Constraints (Optional)

From Clinet et al., pure LQR can produce negative participation (round-trip trades). MPC handles this via:

```
u_t ≥ 0  (if currently selling)
u_t ≤ 0  (if currently buying)
```

This prevents the optimizer from suggesting "buy then immediately sell" within a horizon.

---

## "Aggressive in the Money" Behavior

A key insight from the Clinet paper: when mean-reversion is present, the optimal strategy naturally becomes **aggressive in the money**.

### What This Means

**When price is favorable for our hedge:**
- Need to BUY to hedge AND price is LOW (below S̄): Trade more aggressively
- Need to SELL to hedge AND price is HIGH (above S̄): Trade more aggressively

**When price is unfavorable:**
- Wait, because mean-reversion suggests price will revert to S̄

### Mathematical Intuition

The optimal control from LQR has the form:
```
u* = k_p · (S - S̄) + k_Δ · Δ + f
```

Where:
- `k_p`: Coefficient on price deviation (negative for aggressive-in-the-money)
- `k_Δ`: Coefficient on delta (drives toward zero)
- `f`: Baseline trading rate

When κ (mean-reversion) is high, `k_p` becomes more negative, causing more aggressive trading when price is favorable.

### Practical Implication

| Scenario | Price vs S̄ | Delta | Optimal Action |
|----------|------------|-------|----------------|
| Long delta, need to sell | S > S̄ | + | Sell aggressively (favorable) |
| Long delta, need to sell | S < S̄ | + | Sell slowly (wait for reversion) |
| Short delta, need to buy | S < S̄ | - | Buy aggressively (favorable) |
| Short delta, need to buy | S > S̄ | - | Buy slowly (wait for reversion) |

---

## Parameter Estimation (MHE)

### Parameters to Estimate

Moving Horizon Estimation (MHE) provides real-time estimates of:

1. **Mean-reversion speed (κ):**
   - High κ: Strong mean-reversion, be more patient
   - Low κ: Trending market, hedge quickly

2. **Realized volatility (σ):**
   - Affects expected delta change magnitude
   - Affects uncertainty in predictions

3. **Reference price (S̄):**
   - Could be EMA, VWAP, or estimated equilibrium
   - Determines what "favorable" price means

### MHE Formulation

MHE solves:
```
min_{κ, σ, S̄} Σ_{t=T-M}^{T} [ (S_observed - S_predicted)² / R + (params - params_prior)² / Q ]
```

Where:
- M: Estimation horizon (past observations)
- R: Measurement noise covariance
- Q: Process noise / prior covariance

### Simplified Approach (v1)

For initial implementation, use fixed/simple estimates:
- `κ = 0.2` (moderate mean-reversion, tunable)
- `σ` = rolling realized vol (e.g., 20-period)
- `S̄` = EMA(20) of recent prices

Graduate to MHE in v2 once basic MPC works.

---

## Timing and Discretization

### Time Step (Δt)

| Option | Δt | Notes |
|--------|-----|-------|
| 1 minute | 1/(252×6.5×60) ≈ 1e-5 years | High frequency, more compute |
| 5 minutes | 5/(252×6.5×60) ≈ 5e-5 years | Balanced |
| 15 minutes | 15/(252×6.5×60) ≈ 1.5e-4 years | Lower frequency |

**Recommendation:** Start with 1-minute steps for responsiveness.

### Prediction Horizon (N)

| Option | Horizon | Look-ahead |
|--------|---------|------------|
| N = 10 | 10 minutes | Quick reactions |
| N = 20 | 20 minutes | Balanced |
| N = 30 | 30 minutes | Smoother, more optimal |

**Recommendation:** Start with N = 20 (20 minutes look-ahead).

### MHE Horizon (M)

| Option | History | Notes |
|--------|---------|-------|
| M = 50 | 50 minutes | Shorter memory |
| M = 100 | 100 minutes | Longer memory |

**Recommendation:** M = 60 (1 hour of history).

### Execution Frequency

How often to re-solve MPC and potentially execute:
- Every time step (1 minute): Most responsive
- On price triggers: More efficient compute
- Hybrid: Re-solve every step, execute if |u*| > threshold

---

## Example Scenario

### Setup
- Position: 10 ATM straddles (long gamma)
- SPY: $500
- Portfolio gamma: 37 shares/$
- Portfolio delta: 0 (fully hedged)
- EMA(S̄): $500

### Price Moves to $502

**Delta change:**
```
ΔΔ = Γ × ΔS = 37 × $2 = 74 shares
```
New portfolio delta: +74 shares (long)

**Traditional hedge:** Sell 74 shares immediately.

**MPC analysis:**
- MHE estimates κ = 0.3 (moderate mean-reversion)
- Price $502 > S̄ = $500 → Favorable for selling
- MPC optimizes over 20 steps
- Result: Sell 50 shares now (aggressive in the money)
- Remaining 24 delta: Spread over next few steps or wait for reversion

**If price reverts to $501:**
- Delta drops by 37 shares
- New delta: 74 - 37 - 50 = -13 shares (slight short)
- MPC now suggests: Buy small amount or wait

**Result:** MPC traded less than traditional, captured gamma P&L, avoided round-trip if price reverted.

---

## Implementation Architecture

### Component Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    MPC CONTROLLER FLOW                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. MARKET DATA                                             │
│     └─► Current S, Δ_portfolio, Γ_portfolio                │
│                                                             │
│  2. MHE (Parameter Estimation)                              │
│     └─► Estimate κ, σ, S̄ from recent price history         │
│                                                             │
│  3. MPC (Optimal Control)                                   │
│     ├─► Build dynamics model with current Γ, estimated κ   │
│     ├─► Set initial state x₀ = [S, Δ]                      │
│     ├─► Solve: min J subject to dynamics & constraints     │
│     └─► Extract u₀* (optimal first action)                 │
│                                                             │
│  4. EXECUTION                                               │
│     ├─► If |u₀*| > min_trade_size: Execute trade           │
│     └─► Log decision for analysis                          │
│                                                             │
│  5. REPEAT (every Δt)                                       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### do-mpc Integration

```python
# Pseudo-code structure

import do_mpc

# 1. Create model
model = do_mpc.model.Model('continuous')  # or 'discrete'

# Define states
S = model.set_variable('_x', 'S')
delta = model.set_variable('_x', 'delta')

# Define control
u = model.set_variable('_u', 'u')

# Define parameters (from MHE or fixed)
kappa = model.set_variable('_p', 'kappa')
S_bar = model.set_variable('_p', 'S_bar')
gamma = model.set_variable('_p', 'gamma')

# Set dynamics
model.set_rhs('S', kappa * (S_bar - S))  # Mean-reversion
model.set_rhs('delta', gamma * kappa * (S_bar - S) + u)  # Delta evolution

model.setup()

# 2. Create MPC controller
mpc = do_mpc.controller.MPC(model)

# Set objective
mterm = w_T * delta**2  # Terminal cost
lterm = w_delta * delta**2 + w_exec * u**2  # Stage cost

mpc.set_objective(mterm=mterm, lterm=lterm)

# Set constraints
mpc.bounds['lower', '_u', 'u'] = -u_max
mpc.bounds['upper', '_u', 'u'] = u_max

mpc.setup()

# 3. Run MPC
x0 = np.array([S_current, delta_current])
u_optimal = mpc.make_step(x0)
```

---

## Open Design Questions

### Q1: State Representation

**Option A:** Simple `[S, Δ]`
- Pro: Simpler, fewer states
- Con: Loses information about hedge position vs option delta

**Option B:** Extended `[S, Δ_options, q_equity]`
- Pro: Full information
- Con: More complex, redundant (Δ_total = Δ_options + q_equity)

**Recommendation:** Option A for v1.

### Q2: Objective Weights

How to balance competing objectives?

| Weight | High Value Effect | Low Value Effect |
|--------|-------------------|------------------|
| `w_Δ` | Aggressive hedging | Tolerate delta exposure |
| `w_exec` | Minimize trading | Ignore costs |
| `w_rate` | Smooth trading | Allow jerky trades |
| `w_T` | End near delta-neutral | Flexible endpoint |

**Recommendation for gamma scalping:**
- `w_Δ = 10`: Moderate delta penalty (we want some exposure for gamma P&L)
- `w_exec = 1`: Baseline execution cost
- `w_rate = 0.1`: Small smoothness penalty
- `w_T = 100`: Strong terminal penalty

### Q3: Mean-Reversion Estimation

**Option A:** Fixed κ
- Simple, no MHE needed
- Tune based on historical analysis

**Option B:** Rolling estimate
- Simple online estimation
- κ_t = f(recent price autocorrelation)

**Option C:** Full MHE
- Optimal estimation
- More complex

**Recommendation:** Option A for v1, then Option C for v2.

### Q4: Reference Price S̄

**Options:**
- Arrival price (price when position was opened)
- EMA (exponential moving average)
- VWAP (volume-weighted average)
- Estimated equilibrium (from MHE)

**Recommendation:** EMA(20) for simplicity.

### Q5: Gamma Updates

Should gamma be updated during the horizon?

**Option A:** Fixed gamma over horizon
- Simpler
- Valid if horizon is short (~20 minutes)

**Option B:** Gamma as function of S
- `Γ(S) = Γ_0 + Γ' × (S - S_0)` (first-order approximation)
- More accurate but adds nonlinearity

**Recommendation:** Option A for v1.

---

## Proposed v1 Configuration

### State and Control
```
x = [S, Δ]
u = [Δq]  (shares to trade)
```

### Dynamics
```
S' = S + κ(S̄ - S)Δt
Δ' = Δ + Γ·κ(S̄ - S)Δt + u
```

### Objective
```
L = w_exec · u² + w_Δ · Δ² + w_rate · (u - u_prev)²
L_terminal = w_T · Δ²
```

### Parameters (Initial Values)
```
κ = 0.2 (tunable)
Γ = current portfolio gamma (updated each solve)
S̄ = EMA(20)
Δt = 1 minute = 1/(252×6.5×60) years
N = 20 steps (20 minute horizon)
```

### Constraints
```
|u| ≤ 200 shares per step
|Δ| ≤ 500 shares (soft, via high penalty)
```

### Weights
```
w_exec = 1.0
w_Δ = 10.0
w_rate = 0.1
w_T = 100.0
```

---

## Success Criteria

1. **Functional:** MPC solves in < 1 second
2. **Correctness:** Produces reasonable hedge recommendations
3. **Behavior:** Exhibits "aggressive in the money" when κ > 0
4. **Performance:** In backtest, outperforms fixed-threshold hedging on:
   - Total P&L
   - Transaction costs
   - Sharpe ratio of gamma scalping returns

---

## Next Steps

1. Review this design document
2. Agree on v1 configuration choices
3. Implement MPC controller using do-mpc
4. Test on synthetic price paths
5. Validate "aggressive in the money" behavior
6. Backtest on historical SPY data
7. Add MHE for parameter estimation (v2)

