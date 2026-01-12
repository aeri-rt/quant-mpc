# Design Document: MPC Controller for Delta Hedging (v1 Revised)

## Revision History

| Version | Date | Changes |
|---------|------|---------|
| v1.0 | Initial | Original design |
| v1.1 | Revised | Incorporated review feedback (7 critical fixes) |

---

## Summary of Revisions

Based on implementation review, the following critical issues were addressed:

1. ✓ **State extended to include hedge position** for constraint enforcement
2. ✓ **Delta dynamics clarified** - Δ_opt vs Δ_portfolio distinction
3. ✓ **|u| cost made NLP-friendly** via auxiliary variable formulation
4. ✓ **Direction constraint fixed** - correct sign convention + regime-based approach
5. ✓ **MHE scope reduced** - S̄ external, simpler κ estimation for v1
6. ✓ **Tightened v1 specification** documented
7. ✓ **Validation tests** defined

---

## Revised Mathematical Formulation

### State Vector (Extended)

```
x = [S, Δ_opt, q]
```

Where:
- `S`: Spot price of underlying (SPY)
- `Δ_opt`: Delta from **option positions only** (in share equivalents)
- `q`: Current equity hedge position (shares of SPY held)

**Portfolio delta is derived:**
```
Δ_portfolio = Δ_opt + q
```

**Rationale:** This allows the MPC to enforce position limits on `q` and understand inventory state. Without tracking `q` explicitly, the controller cannot know if it's near position limits.

### Control Vector

```
u = Δq (shares to trade)
```

**Sign convention:**
- `u > 0`: BUY shares (increases q, increases Δ_portfolio)
- `u < 0`: SELL shares (decreases q, decreases Δ_portfolio)

### Dynamics Model

#### 1. Price Dynamics (Expected Ornstein-Uhlenbeck)

```
S_{t+1} = S_t + κ(S̄ - S_t)Δt
```

Where:
- `κ`: Mean-reversion speed (estimated or fixed)
- `S̄`: Reference price (external signal: EMA or VWAP)
- `Δt`: Time step

**Note:** This is deterministic expected dynamics. The actual price has noise σ√Δt·ε, but for MPC planning we use the expected path.

#### 2. Equity Position Dynamics

```
q_{t+1} = q_t + u_t
```

Simple integration of trades.

#### 3. Options Delta Dynamics

```
Δ_opt_{t+1} = Δ_opt_t + Γ_opt × (S_{t+1} - S_t)
```

Where:
- `Γ_opt`: Portfolio gamma from **options only** (equity has γ=0)

**Note:** This is a first-order approximation. For v1, we assume:
- Γ_opt is constant over the horizon (updated at each MPC solve)
- Ignore time decay of delta (charm) and vanna effects

**Future enhancement:** Add second-order terms:
```
Δ_opt_{t+1} = Δ_opt_t + Γ_opt×ΔS + (1/2)×speed×ΔS² + charm×Δt
```

#### 4. Derived Portfolio Delta

```
Δ_portfolio_t = Δ_opt_t + q_t
```

This is what we penalize in the objective.

---

## Objective Function (NLP-Safe)

### Stage Cost

```
L_stage = w_Δ × Δ_portfolio² + w_exec × (c_quad×u² + c_lin×z) + w_rate × (u - u_prev)²
```

Where `z` is an auxiliary variable for the absolute value (see below).

### Terminal Cost

```
L_terminal = w_T × Δ_portfolio_N²
```

### Total Objective

```
J = Σ_{t=0}^{N-1} L_stage(x_t, u_t, z_t) + L_terminal(x_N)
```

### Handling |u| Safely for IPOPT

**Problem:** `|u|` is non-smooth, causing issues with gradient-based NLP solvers.

**Solution:** Introduce auxiliary variable `z_t ≥ 0` with constraints:

```
z_t ≥ u_t
z_t ≥ -u_t
```

Then replace `|u_t|` with `z_t` in the objective. The optimizer will naturally set `z_t = |u_t|` at optimum.

**Implementation in CasADi:**
```python
z = model.set_variable('_z', 'z_abs_u')  # Auxiliary variable
# Add constraints:
# z >= u  →  z - u >= 0
# z >= -u →  z + u >= 0
```

### Volatility-Scaled Risk Penalty (Robustness Enhancement)

**Issue:** Deterministic OU dynamics may cause MPC to "wait for reversion" when the market is actually trending or volatile.

**v1 Enhancement:** Scale the delta penalty by realized volatility:

```
L_risk = w_Δ × (1 + α×σ/σ_base) × Δ_portfolio²
```

Where:
- `σ`: Current estimated volatility
- `σ_base`: Baseline volatility (e.g., 15%)
- `α`: Scaling factor (e.g., 1.0)

**Effect:** Higher volatility → higher penalty for delta exposure → more aggressive hedging when uncertain.

---

## Constraints

### Position Limits (on q)

```
-q_max ≤ q_t ≤ q_max
```

Example: `q_max = 1000` shares

### Trade Size Limits (on u)

```
-u_max ≤ u_t ≤ u_max
```

Example: `u_max = 200` shares per step

### Auxiliary Variable Constraints (for |u|)

```
z_t ≥ u_t
z_t ≥ -u_t
z_t ≥ 0
```

### Direction Regime Constraint (No Round-Trips)

**Issue:** Without constraints, MPC might suggest buying then immediately selling within a horizon.

**v1 Solution:** At each MPC solve, determine direction regime based on current portfolio delta sign:

```python
# At MPC solve time:
if Δ_portfolio > threshold:
    # Need to reduce delta → sell only
    u_lower = -u_max
    u_upper = 0
elif Δ_portfolio < -threshold:
    # Need to increase delta → buy only
    u_lower = 0
    u_upper = u_max
else:
    # Near neutral → allow both but prefer no action
    u_lower = -u_max
    u_upper = u_max
```

Where `threshold` is a small deadband (e.g., 10 shares) to avoid chatter.

---

## Parameter Specification (v1)

### Fixed Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| `Δt` | 1 min = 1/(252×6.5×60) years | Time step |
| `N` | 20 | Prediction horizon (20 minutes) |
| `κ` | 0.2 (tunable) | Mean-reversion speed |
| `Γ_opt` | Updated each solve | Current portfolio gamma |
| `σ` | Rolling 20-period realized vol | For risk scaling |

### External Signals

| Signal | Computation | Notes |
|--------|-------------|-------|
| `S̄` | EMA(S, span=20) | Reference price for mean-reversion |

### Constraint Bounds

| Constraint | Value | Notes |
|------------|-------|-------|
| `q_max` | 1000 shares | Maximum equity position |
| `u_max` | 200 shares | Maximum trade per step |
| `threshold` | 10 shares | Direction regime deadband |

### Objective Weights

| Weight | Value | Notes |
|--------|-------|-------|
| `w_Δ` | 10.0 | Delta exposure penalty |
| `w_exec` | 1.0 | Execution cost weight |
| `c_quad` | 0.001 | Quadratic cost per share² |
| `c_lin` | 0.01 | Linear cost per share (spread + commission) |
| `w_rate` | 0.1 | Trade smoothness penalty |
| `w_T` | 100.0 | Terminal penalty |

---

## State Transition Summary

```
Given: x_t = [S_t, Δ_opt_t, q_t], u_t, parameters (κ, S̄, Γ_opt)

1. Price update:
   S_{t+1} = S_t + κ(S̄ - S_t)Δt

2. Equity position update:
   q_{t+1} = q_t + u_t

3. Options delta update:
   Δ_opt_{t+1} = Δ_opt_t + Γ_opt × (S_{t+1} - S_t)

4. Portfolio delta (for objective):
   Δ_portfolio_{t+1} = Δ_opt_{t+1} + q_{t+1}

Output: x_{t+1} = [S_{t+1}, Δ_opt_{t+1}, q_{t+1}]
```

---

## CasADi/do-mpc Implementation

### Model Definition

```python
import do_mpc
import casadi as ca
import numpy as np

def create_mpc_model():
    """Create the do-mpc model for delta hedging."""
    
    model_type = 'discrete'  # Discrete time
    model = do_mpc.model.Model(model_type)
    
    # === States ===
    S = model.set_variable(var_type='_x', var_name='S', shape=(1, 1))
    delta_opt = model.set_variable(var_type='_x', var_name='delta_opt', shape=(1, 1))
    q = model.set_variable(var_type='_x', var_name='q', shape=(1, 1))
    
    # === Controls ===
    u = model.set_variable(var_type='_u', var_name='u', shape=(1, 1))
    
    # === Algebraic (auxiliary for |u|) ===
    z = model.set_variable(var_type='_z', var_name='z_abs', shape=(1, 1))
    
    # === Time-varying parameters (updated each solve) ===
    kappa = model.set_variable(var_type='_tvp', var_name='kappa')
    S_bar = model.set_variable(var_type='_tvp', var_name='S_bar')
    gamma_opt = model.set_variable(var_type='_tvp', var_name='gamma_opt')
    sigma_scale = model.set_variable(var_type='_tvp', var_name='sigma_scale')  # For risk scaling
    
    # === Derived quantities ===
    delta_portfolio = delta_opt + q
    
    # === Time step ===
    dt = 1.0 / (252 * 6.5 * 60)  # 1 minute in years
    
    # === State transitions ===
    
    # Price dynamics (discrete-time OU)
    S_next = S + kappa * (S_bar - S) * dt
    
    # Equity position dynamics
    q_next = q + u
    
    # Options delta dynamics (first-order gamma approximation)
    dS = S_next - S
    delta_opt_next = delta_opt + gamma_opt * dS
    
    # === Set right-hand side ===
    model.set_rhs('S', S_next)
    model.set_rhs('delta_opt', delta_opt_next)
    model.set_rhs('q', q_next)
    
    # === Algebraic equation for z (auxiliary for |u|) ===
    # z represents |u|, enforced via constraints z >= u, z >= -u
    model.set_alg('z_abs', z - ca.fmax(u, -u))  # Soft definition
    
    # === Setup model ===
    model.setup()
    
    return model


def create_mpc_controller(model, config):
    """Create and configure the MPC controller."""
    
    mpc = do_mpc.controller.MPC(model)
    
    # === Timing ===
    setup_mpc = {
        'n_horizon': config['N'],
        't_step': config['dt'],
        'n_robust': 0,
        'store_full_solution': True,
    }
    mpc.set_param(**setup_mpc)
    
    # === Objective function ===
    
    # Get variables
    S = model.x['S']
    delta_opt = model.x['delta_opt']
    q = model.x['q']
    u = model.u['u']
    z = model.z['z_abs']
    sigma_scale = model.tvp['sigma_scale']
    
    # Derived
    delta_portfolio = delta_opt + q
    
    # Weights
    w_delta = config['w_delta']
    w_exec = config['w_exec']
    c_quad = config['c_quad']
    c_lin = config['c_lin']
    w_rate = config['w_rate']
    w_T = config['w_T']
    
    # Terminal cost
    mterm = w_T * delta_portfolio**2
    
    # Stage cost
    # Risk penalty (scaled by volatility)
    L_risk = w_delta * sigma_scale * delta_portfolio**2
    
    # Execution cost (quadratic + linear via auxiliary z)
    L_exec = w_exec * (c_quad * u**2 + c_lin * z)
    
    # Note: Smoothness term (u - u_prev)² is harder in do-mpc
    # Alternative: penalize u² more heavily, or use rterm
    
    lterm = L_risk + L_exec
    
    mpc.set_objective(mterm=mterm, lterm=lterm)
    
    # Rate penalty (do-mpc built-in)
    mpc.set_rterm(u=config['w_rate'])
    
    # === Constraints ===
    
    # Control bounds
    mpc.bounds['lower', '_u', 'u'] = -config['u_max']
    mpc.bounds['upper', '_u', 'u'] = config['u_max']
    
    # Position bounds on q
    mpc.bounds['lower', '_x', 'q'] = -config['q_max']
    mpc.bounds['upper', '_x', 'q'] = config['q_max']
    
    # Auxiliary variable z >= 0
    mpc.bounds['lower', '_z', 'z_abs'] = 0
    
    # Constraints for z >= |u|
    # z >= u  →  z - u >= 0
    # z >= -u →  z + u >= 0
    mpc.set_nl_cons('z_geq_u', z - u, ub=np.inf, soft_constraint=False)
    mpc.set_nl_cons('z_geq_neg_u', z + u, ub=np.inf, soft_constraint=False)
    
    # === TVP function (time-varying parameters) ===
    tvp_template = mpc.get_tvp_template()
    
    def tvp_fun(t_now):
        # These will be updated at each MPC solve
        for k in range(config['N'] + 1):
            tvp_template['_tvp', k, 'kappa'] = config['kappa']
            tvp_template['_tvp', k, 'S_bar'] = config['S_bar']
            tvp_template['_tvp', k, 'gamma_opt'] = config['gamma_opt']
            tvp_template['_tvp', k, 'sigma_scale'] = config['sigma_scale']
        return tvp_template
    
    mpc.set_tvp_fun(tvp_fun)
    
    # === Setup ===
    mpc.setup()
    
    return mpc


def apply_direction_regime(mpc, delta_portfolio, config):
    """
    Apply direction regime constraint based on current portfolio delta.
    
    Call this before each MPC solve to set appropriate u bounds.
    """
    threshold = config.get('direction_threshold', 10)
    u_max = config['u_max']
    
    if delta_portfolio > threshold:
        # Need to reduce delta → sell only (u <= 0)
        mpc.bounds['lower', '_u', 'u'] = -u_max
        mpc.bounds['upper', '_u', 'u'] = 0
    elif delta_portfolio < -threshold:
        # Need to increase delta → buy only (u >= 0)
        mpc.bounds['lower', '_u', 'u'] = 0
        mpc.bounds['upper', '_u', 'u'] = u_max
    else:
        # Near neutral → allow both
        mpc.bounds['lower', '_u', 'u'] = -u_max
        mpc.bounds['upper', '_u', 'u'] = u_max
```

### Configuration

```python
DEFAULT_CONFIG = {
    # Timing
    'dt': 1.0 / (252 * 6.5 * 60),  # 1 minute
    'N': 20,  # 20-step horizon
    
    # Constraints
    'q_max': 1000,  # Max equity position (shares)
    'u_max': 200,   # Max trade per step (shares)
    'direction_threshold': 10,  # Deadband for direction regime
    
    # Objective weights
    'w_delta': 10.0,
    'w_exec': 1.0,
    'c_quad': 0.001,
    'c_lin': 0.01,
    'w_rate': 0.1,
    'w_T': 100.0,
    
    # Parameters (updated at runtime)
    'kappa': 0.2,
    'S_bar': 500.0,  # Will be updated with EMA
    'gamma_opt': 37.0,  # Will be updated with current gamma
    'sigma_scale': 1.0,  # Will be updated based on volatility
}
```

### MPC Solve Loop

```python
def run_mpc_step(mpc, market_state, config):
    """
    Run one MPC iteration.
    
    Args:
        mpc: Configured MPC controller
        market_state: Dict with current S, delta_opt, q, gamma_opt, sigma, S_bar
        config: Configuration dict
        
    Returns:
        u_optimal: Recommended trade (shares)
    """
    # Update time-varying parameters
    config['S_bar'] = market_state['S_bar']
    config['gamma_opt'] = market_state['gamma_opt']
    config['sigma_scale'] = 1.0 + market_state['sigma'] / 0.15  # Scale by vol
    
    # Apply direction regime constraint
    delta_portfolio = market_state['delta_opt'] + market_state['q']
    apply_direction_regime(mpc, delta_portfolio, config)
    
    # Set initial state
    x0 = np.array([
        [market_state['S']],
        [market_state['delta_opt']],
        [market_state['q']]
    ])
    
    # Solve MPC
    u_optimal = mpc.make_step(x0)
    
    return float(u_optimal[0, 0])
```

---

## Validation Tests

Before deploying, run these tests:

### Test 1: Aggressive-in-the-Money Behavior (OU Path)

**Setup:**
- Generate synthetic OU price path with known κ
- Start with Δ_portfolio = 0
- Inject price deviation from S̄

**Expected:**
- When price favorable (e.g., S < S̄ and need to buy): Trade more aggressively
- When price unfavorable: Trade less, wait for reversion

**Pass criteria:** Correlation between (S - S̄) × sign(Δ) and |u| should be positive

### Test 2: Trending Path Robustness

**Setup:**
- Generate trending price path (no mean-reversion)
- Use same κ = 0.2 (misspecified)

**Expected:**
- Delta should not blow out
- σ_scale penalty should increase hedging as vol rises

**Pass criteria:** |Δ_portfolio| stays within bounds, no catastrophic loss

### Test 3: Cost Sensitivity

**Setup:**
- Run MPC with varying c_lin values (0.001, 0.01, 0.1)

**Expected:**
- Higher costs → fewer, larger trades
- Lower costs → more frequent, smaller trades

**Pass criteria:** Monotonic relationship between cost and trade frequency

### Test 4: Constraint Stress

**Setup:**
- Start with q near q_max
- Inject large delta deviation requiring hedge

**Expected:**
- MPC should recognize position limit
- Find feasible solution (partial hedge or wait)
- Not exceed q_max

**Pass criteria:** q stays within bounds, solution is feasible

---

## Next Steps

1. Implement the CasADi/do-mpc model as specified
2. Run validation tests on synthetic data
3. Tune weights based on test results
4. Integrate with Greeks engine and market data
5. Paper trade on IBKR
6. Add MHE for κ estimation (v2)

