# Paper Summary: Optimal Trading via Model Predictive Control

**Paper:** Clinet, S., Perreton, J.F., Reydellet, S. (2021). "Optimal trading: a model predictive control approach." arXiv:2110.11008v2

**Relevance:** This is the foundational paper for our MPC approach to trading.

---

## Key Contributions

1. **LQR Framework for Execution:** Formulates optimal execution as a Linear Quadratic Regulator problem
2. **Mean-Reversion Signal:** Incorporates price mean-reversion into the optimization
3. **Aggressive in the Money:** Strategy accelerates when price is favorable
4. **MPC for Constraints:** Uses Model Predictive Control to handle positivity constraints on participation rate

---

## Problem Setup

### State Variables
- $p_t = (S_t - \mathfrak{S}_t) / S_0$: Price slippage relative to benchmark $\mathfrak{S}_t$
- $q_t \in [0, 1]$: Proportion of shares remaining to execute

### Control Variable
- $u_t$: Participation rate (fraction of shares to trade at time $t$)

### Price Dynamics
The key innovation is including mean-reversion in price:

$$dS_t = \kappa(\bar{S} - S_t)dt + \sigma dW_t$$

Where:
- $\kappa$: Mean-reversion speed
- $\bar{S}$: Reference/equilibrium price
- $\sigma$: Volatility

### Market Impact
Linear market impact model:
$$\text{Impact} = \eta \cdot u_t$$

---

## LQR Formulation

### Objective Function
Minimize expected cost:

$$J = \mathbb{E}\left[\sum_{t=0}^{T} \left( \text{execution cost} + \text{risk penalty} \right) + \text{terminal cost}\right]$$

### Key Result: Optimal Control is Affine

The optimal participation rate is:

$$u_t^* = k_t^T x_t + f_t$$

Where:
- $x_t = (p_t, q_t)^T$ is the state
- $k_t$ are time-varying gains (computed iteratively backwards)
- $f_t$ is a time-varying offset

**This is computable before execution begins!**

---

## "Aggressive in the Money" Behavior

The strategy naturally accelerates when the price is favorable:

- **When buying:** Increase participation when price is LOW (below reference)
- **When selling:** Increase participation when price is HIGH (above reference)

This emerges from the mean-reversion term in the dynamics:
- If price is below $\bar{S}$, it's expected to rise → buy faster
- If price is above $\bar{S}$, it's expected to fall → sell faster

The **intensity** of this behavior is controlled by $\kappa$ (mean-reversion speed).

---

## MPC Extension

### Problem with Pure LQR
LQR can produce negative participation rates (round-trip trades), which are often undesirable.

### MPC Solution
Use Model Predictive Control with constraints:

1. Solve LQR over finite horizon
2. Add constraint: $u_t \geq 0$ (no round trips)
3. Implement only first control action
4. Re-solve at next timestep

### Advantages of MPC over Hard Constraints
- Smoother than putting a hard floor on participation
- More consistent with completion constraint
- Natural handling of changing market conditions

---

## Key Parameters

| Parameter | Symbol | Description | Effect |
|-----------|--------|-------------|--------|
| Mean-reversion speed | $\kappa$ | How fast price reverts | Higher → more aggressive ITM |
| Volatility | $\sigma$ | Price uncertainty | Higher → more cautious |
| Risk aversion | $\lambda$ | Penalizes execution risk | Higher → faster execution |
| Market impact | $\eta$ | Temporary price impact | Higher → slower execution |
| Horizon | $T$ | Time to complete | Shorter → more urgent |

---

## Application to Our System

### How We'll Use This

1. **For Delta Hedging:**
   - State: $(p_t, \Delta_t)$ where $\Delta_t$ is portfolio delta
   - Control: hedge quantity
   - Objective: minimize hedging cost + delta exposure

2. **Mean-Reversion Signal:**
   - Use MHE to estimate $\kappa$ in real-time
   - When $\kappa$ is high (strong mean-reversion): hedge less frequently
   - When $\kappa$ is low (momentum): hedge more aggressively

3. **Aggressive in the Money:**
   - When SPY price is favorable for hedging (low when we need to buy), increase hedge size
   - This naturally integrates market timing into hedging

### Modifications for Our Context

| Paper | Our System |
|-------|------------|
| Execution of fixed quantity | Continuous delta hedging |
| Complete by time $T$ | No hard completion (ongoing) |
| Single instrument | Multiple (SPY + options) |
| Price mean-reversion | Extend to vol mean-reversion |

---

## Mathematical Details

### Riccati Equation for Gains

The gain matrices $P_t$ are computed backwards:

$$P_t = Q + A^T P_{t+1} A - A^T P_{t+1} B (R + B^T P_{t+1} B)^{-1} B^T P_{t+1} A$$

Where:
- $A$: State transition matrix
- $B$: Control input matrix
- $Q$: State cost matrix
- $R$: Control cost matrix

### Continuous-Time Limit

In continuous time with further assumptions, closed-form solutions exist:

$$u^*(t, p, q) = \alpha(t) + \beta(t) p + \gamma(t) q$$

Where $\alpha, \beta, \gamma$ are functions of time and parameters.

---

## Implementation Notes

1. **do-mpc handles the MPC formulation** - we define dynamics, costs, constraints
2. **CasADi provides automatic differentiation** - for the nonlinear extensions
3. **IPOPT solves the resulting NLP** - or qpOASES for the QP case

### Horizon Selection
- Paper suggests horizon should cover expected duration of price excursion
- For delta hedging: 10-20 steps at 1-minute frequency = 10-20 minutes
- Can be tuned based on estimated $\kappa$

---

## References from Paper

- Almgren & Chriss (2001) - Original optimal execution framework
- Gatheral & Schied (2011) - Aggressive in the money property
- Bertsimas & Lo (1998) - Early optimal execution work
- Bertsekas (1995) - LQR theory

---

## Questions for Implementation

1. How to estimate $\kappa$ in real-time? → MHE
2. What reference price $\bar{S}$ to use? → EMA or VWAP
3. How to extend to multi-dimensional (hedging multiple Greeks)?
4. How to handle overnight positions?

