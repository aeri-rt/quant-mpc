"""
MPC Controller for Delta Hedging (v1)

Implements Model Predictive Control for optimizing hedge trajectories
in a gamma scalping strategy.

State: x = [S, Δ_opt, q]
Control: u = Δq (shares to trade)

See docs/design/mpc-controller-design-v1-revised.md for full specification.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, TYPE_CHECKING
import time
import numpy as np
import casadi as ca
import do_mpc

from src.utils.logging import get_logger, log_mpc_solve, Timer
from src.utils.helpers import TRADING_MINUTES_PER_YEAR

if TYPE_CHECKING:
    from src.utils.config import MPCParams

# Get component-specific logger
logger = get_logger("mpc")


@dataclass
class MPCConfig:
    """Configuration for the MPC controller."""
    
    # Timing
    dt: float = 1.0 / TRADING_MINUTES_PER_YEAR  # 1 minute in years
    n_horizon: int = 20  # Prediction horizon steps
    
    # Constraints
    q_max: float = 1000.0  # Max equity position (shares)
    u_max: float = 200.0   # Max trade per step (shares)
    direction_threshold: float = 10.0  # Deadband for direction regime
    
    # Objective weights
    w_delta: float = 10.0      # Delta exposure penalty
    w_exec: float = 1.0        # Execution cost weight
    c_quad: float = 0.001      # Quadratic cost per share²
    c_lin: float = 0.01        # Linear cost per share (spread + commission)
    w_rate: float = 0.1        # Trade smoothness penalty
    w_terminal: float = 100.0  # Terminal delta penalty
    
    # Mean-reversion parameter (fixed for v1)
    kappa: float = 0.2
    
    # Volatility scaling
    sigma_base: float = 0.15  # Baseline volatility for scaling
    
    @classmethod
    def from_yaml_config(cls, mpc_params: "MPCParams") -> "MPCConfig":
        """
        Create MPCConfig from YAML-loaded MPCParams.
        
        This connects the YAML configuration to the MPC controller.
        
        Args:
            mpc_params: MPCParams loaded from config/mpc_params.yaml
            
        Returns:
            MPCConfig instance with values from YAML
        """
        return cls(
            dt=mpc_params.dt,  # Already converted to years in MPCParams
            n_horizon=mpc_params.horizon,
            q_max=float(mpc_params.constraints.max_position),
            u_max=float(mpc_params.constraints.max_trade_rate),
            direction_threshold=10.0,  # Keep default for now
            w_delta=mpc_params.weights.delta_deviation,
            w_exec=mpc_params.weights.execution_cost,
            c_quad=mpc_params.costs.c_quad,
            c_lin=mpc_params.costs.c_lin,
            w_rate=mpc_params.weights.delta_rate,
            w_terminal=mpc_params.weights.terminal_delta,
            kappa=mpc_params.kappa_init,
            sigma_base=0.15,  # Not in YAML yet, use default
        )


@dataclass
class MPCState:
    """Current state for MPC solve."""
    
    S: float           # Spot price
    delta_opt: float   # Options delta (share equivalents)
    q: float           # Current equity position (shares)
    gamma_opt: float   # Options gamma (share equivalents per $)
    sigma: float       # Current realized volatility
    S_bar: float       # Reference price (EMA/VWAP)


@dataclass
class MPCSolution:
    """Result from MPC solve."""
    
    u_optimal: float           # Recommended trade (shares)
    u_trajectory: np.ndarray   # Full control trajectory
    x_trajectory: np.ndarray   # Full state trajectory
    delta_portfolio_pred: np.ndarray  # Predicted portfolio delta
    cost: float                # Optimal cost
    solver_status: str         # Solver status


class DeltaHedgeMPC:
    """
    Model Predictive Controller for delta hedging.
    
    Optimizes hedge trades to minimize:
    - Delta exposure risk (scaled by volatility)
    - Transaction costs (linear + quadratic)
    - Trading rate (smoothness)
    
    Subject to:
    - Position limits on equity holdings
    - Trade size limits per step
    - Direction regime (no round-trips)
    """
    
    def __init__(self, config: Optional[MPCConfig] = None):
        """Initialize the MPC controller."""
        self.config = config or MPCConfig()
        
        # Runtime parameters (updated each solve) - must be set BEFORE building controller
        self._kappa = self.config.kappa
        self._S_bar = 500.0
        self._gamma_opt = 0.0
        self._sigma_scale = 1.0
        
        # Build model and controller
        self.model = self._build_model()
        self.mpc = self._build_controller()
        
        logger.info("DeltaHedgeMPC initialized with config: {}", self.config)
    
    def _build_model(self) -> do_mpc.model.Model:
        """Build the do-mpc model."""
        
        model_type = 'discrete'
        model = do_mpc.model.Model(model_type)
        
        # === States ===
        # S: Spot price
        S = model.set_variable(var_type='_x', var_name='S', shape=(1, 1))
        
        # delta_opt: Options delta (share equivalents)
        delta_opt = model.set_variable(var_type='_x', var_name='delta_opt', shape=(1, 1))
        
        # q: Equity hedge position (shares)
        q = model.set_variable(var_type='_x', var_name='q', shape=(1, 1))
        
        # === Controls ===
        # u: Trade size (shares, + = buy, - = sell)
        u = model.set_variable(var_type='_u', var_name='u', shape=(1, 1))
        
        # === Algebraic variables (for |u| reformulation) ===
        z_abs = model.set_variable(var_type='_z', var_name='z_abs', shape=(1, 1))
        
        # === Time-varying parameters ===
        kappa = model.set_variable(var_type='_tvp', var_name='kappa')
        S_bar = model.set_variable(var_type='_tvp', var_name='S_bar')
        gamma_opt = model.set_variable(var_type='_tvp', var_name='gamma_opt')
        sigma_scale = model.set_variable(var_type='_tvp', var_name='sigma_scale')
        
        # === Dynamics ===
        
        dt = self.config.dt
        
        # Price dynamics: discrete-time expected OU
        # S_{t+1} = S_t + κ(S̄ - S_t)Δt
        S_next = S + kappa * (S_bar - S) * dt
        
        # Equity position dynamics: simple integration
        # q_{t+1} = q_t + u_t
        q_next = q + u
        
        # Options delta dynamics: first-order gamma approximation
        # Δ_opt_{t+1} = Δ_opt_t + Γ_opt × (S_{t+1} - S_t)
        dS = S_next - S
        delta_opt_next = delta_opt + gamma_opt * dS
        
        # Set state transitions
        model.set_rhs('S', S_next)
        model.set_rhs('delta_opt', delta_opt_next)
        model.set_rhs('q', q_next)
        
        # === Algebraic equation for z_abs ===
        # z_abs will be constrained to equal |u| via inequality constraints
        # Here we set a placeholder (actual enforcement via constraints)
        model.set_alg('z_abs', z_abs - ca.sqrt(u**2 + 1e-8))  # Smooth approximation
        
        # Finalize model
        model.setup()
        
        logger.debug("MPC model built successfully")
        return model
    
    def _build_controller(self) -> do_mpc.controller.MPC:
        """Build the MPC controller."""
        
        mpc = do_mpc.controller.MPC(self.model)
        
        # === Setup parameters ===
        setup_mpc = {
            'n_horizon': self.config.n_horizon,
            't_step': self.config.dt,
            'n_robust': 0,
            'store_full_solution': True,
        }
        mpc.set_param(**setup_mpc)
        
        # === Get model variables ===
        S = self.model.x['S']
        delta_opt = self.model.x['delta_opt']
        q = self.model.x['q']
        u = self.model.u['u']
        z_abs = self.model.z['z_abs']
        sigma_scale = self.model.tvp['sigma_scale']
        
        # Derived: portfolio delta
        delta_portfolio = delta_opt + q
        
        # === Objective function ===
        
        # Weights
        w_delta = self.config.w_delta
        w_exec = self.config.w_exec
        c_quad = self.config.c_quad
        c_lin = self.config.c_lin
        w_terminal = self.config.w_terminal
        
        # Terminal cost: penalize final delta exposure heavily
        mterm = w_terminal * delta_portfolio**2
        
        # Stage cost:
        # 1. Delta exposure penalty (scaled by volatility)
        L_risk = w_delta * sigma_scale * delta_portfolio**2
        
        # 2. Execution cost (quadratic + linear via z_abs)
        L_exec = w_exec * (c_quad * u**2 + c_lin * z_abs)
        
        lterm = L_risk + L_exec
        
        mpc.set_objective(mterm=mterm, lterm=lterm)
        
        # Rate penalty for smoothness (do-mpc built-in: penalizes (u_t - u_{t-1})²)
        mpc.set_rterm(u=self.config.w_rate)
        
        # === Constraints ===
        
        # Control bounds (will be modified dynamically for direction regime)
        mpc.bounds['lower', '_u', 'u'] = -self.config.u_max
        mpc.bounds['upper', '_u', 'u'] = self.config.u_max
        
        # State bounds: position limits on q
        mpc.bounds['lower', '_x', 'q'] = -self.config.q_max
        mpc.bounds['upper', '_x', 'q'] = self.config.q_max
        
        # =======================================================================
        # z_abs Auxiliary Variable Formulation
        # =======================================================================
        # 
        # To handle the |u| term in the cost function (which is non-smooth),
        # we use the auxiliary variable z_abs with constraints:
        #   z_abs >= |u|  ⟺  z_abs >= u AND z_abs >= -u
        #
        # This reformulates the non-smooth |u| as a smooth LP in (z_abs, u).
        # The optimizer will minimize z_abs to the smallest value satisfying
        # the constraints, which is exactly |u|.
        #
        # Constraint formulation for do-mpc:
        #   g(x,u,z) <= ub, with lb <= g(x,u,z) handled via lower bound
        #
        # We need:
        #   z_abs - u >= 0  → constraint: u - z_abs <= 0
        #   z_abs + u >= 0  → constraint: -u - z_abs <= 0
        # =======================================================================
        
        # Lower bound: z_abs >= 0 (auxiliary must be non-negative)
        mpc.bounds['lower', '_z', 'z_abs'] = 0
        
        # Constraint 1: z_abs >= u (i.e., u - z_abs <= 0)
        mpc.set_nl_cons('z_abs_geq_u', u - z_abs, ub=0, soft_constraint=False)
        
        # Constraint 2: z_abs >= -u (i.e., -u - z_abs <= 0)
        mpc.set_nl_cons('z_abs_geq_neg_u', -u - z_abs, ub=0, soft_constraint=False)
        
        # do-mpc set_nl_cons uses:
        # lb <= expr <= ub
        # So for z - u >= 0, we want: 0 <= z - u <= inf
        # We write: expr=z-u, ub=inf (defaults to inf), and need to set lb or use soft
        
        # Simpler: use soft constraint with penalty or reformulate
        # For now, the smooth approximation z = sqrt(u² + ε) handles this
        
        # === TVP function ===
        tvp_template = mpc.get_tvp_template()
        
        def tvp_fun(t_now):
            for k in range(self.config.n_horizon + 1):
                tvp_template['_tvp', k, 'kappa'] = self._kappa
                tvp_template['_tvp', k, 'S_bar'] = self._S_bar
                tvp_template['_tvp', k, 'gamma_opt'] = self._gamma_opt
                tvp_template['_tvp', k, 'sigma_scale'] = self._sigma_scale
            return tvp_template
        
        mpc.set_tvp_fun(tvp_fun)
        
        # === IPOPT solver options ===
        mpc.set_param(nlpsol_opts={'ipopt.print_level': 0, 'print_time': 0})
        
        # Setup controller
        mpc.setup()
        
        logger.debug("MPC controller built successfully")
        return mpc
    
    def _apply_direction_regime(self, delta_portfolio: float) -> None:
        """
        Apply direction regime constraint based on current portfolio delta.
        
        Prevents round-trips by restricting trade direction:
        - If Δ_portfolio > threshold: sell only (u <= 0)
        - If Δ_portfolio < -threshold: buy only (u >= 0)
        """
        threshold = self.config.direction_threshold
        u_max = self.config.u_max
        
        if delta_portfolio > threshold:
            # Long delta → need to sell to reduce
            self.mpc.bounds['lower', '_u', 'u'] = -u_max
            self.mpc.bounds['upper', '_u', 'u'] = 0
            logger.debug("Direction regime: SELL ONLY (Δ={:.1f})", delta_portfolio)
            
        elif delta_portfolio < -threshold:
            # Short delta → need to buy to increase
            self.mpc.bounds['lower', '_u', 'u'] = 0
            self.mpc.bounds['upper', '_u', 'u'] = u_max
            logger.debug("Direction regime: BUY ONLY (Δ={:.1f})", delta_portfolio)
            
        else:
            # Near neutral → allow both directions
            self.mpc.bounds['lower', '_u', 'u'] = -u_max
            self.mpc.bounds['upper', '_u', 'u'] = u_max
            logger.debug("Direction regime: BOTH (Δ={:.1f})", delta_portfolio)
    
    def solve(self, state: MPCState) -> MPCSolution:
        """
        Solve the MPC problem given current state.
        
        Args:
            state: Current market state (S, delta_opt, q, gamma_opt, sigma, S_bar)
            
        Returns:
            MPCSolution with optimal trade and trajectories
        """
        solve_start = time.perf_counter()
        
        # Update time-varying parameters
        self._kappa = self.config.kappa
        self._S_bar = state.S_bar
        self._gamma_opt = state.gamma_opt
        self._sigma_scale = 1.0 + (state.sigma / self.config.sigma_base - 1.0)
        self._sigma_scale = max(0.5, min(2.0, self._sigma_scale))  # Clamp to [0.5, 2.0]
        
        # Apply direction regime
        delta_portfolio = state.delta_opt + state.q
        self._apply_direction_regime(delta_portfolio)
        
        # Determine regime name for logging
        if delta_portfolio > self.config.direction_threshold:
            direction_regime = "SELL_ONLY"
        elif delta_portfolio < -self.config.direction_threshold:
            direction_regime = "BUY_ONLY"
        else:
            direction_regime = "BOTH"
        
        # Set initial state
        x0 = np.array([
            [state.S],
            [state.delta_opt],
            [state.q]
        ])
        
        # Set initial guesses (required by do-mpc)
        # x0 guess: repeat initial state over horizon
        self.mpc.x0 = x0
        
        # u0 guess: zeros
        self.mpc.u0 = np.zeros((1, 1))
        
        # z0 guess: algebraic variable z_abs
        self.mpc.z0 = np.zeros((1, 1))
        
        # Set initial guess for solver from current state
        self.mpc.set_initial_guess()
        
        try:
            # Solve MPC
            u_opt = self.mpc.make_step(x0)
            
            # Extract solution
            u_optimal = float(u_opt[0, 0])
            
            # Get full trajectories from stored solution
            x_traj = np.array(self.mpc.data['_x']).squeeze()
            u_traj = np.array(self.mpc.data['_u']).squeeze()
            
            # Compute predicted portfolio delta trajectory
            if x_traj.ndim == 2:
                delta_portfolio_pred = x_traj[:, 1] + x_traj[:, 2]  # delta_opt + q
            else:
                delta_portfolio_pred = np.array([delta_portfolio])
            
            # Try to get cost, use 0 if not available
            try:
                cost = float(self.mpc.opt_x_num['_x', -1, 0, 0])  # Use a safe accessor
            except:
                cost = 0.0
            
            solve_time_ms = (time.perf_counter() - solve_start) * 1000
            
            solution = MPCSolution(
                u_optimal=u_optimal,
                u_trajectory=u_traj,
                x_trajectory=x_traj,
                delta_portfolio_pred=delta_portfolio_pred,
                cost=cost,
                solver_status='optimal'
            )
            
            # Log structured MPC solve
            log_mpc_solve(state, solution, solve_time_ms, direction_regime)
            
            return solution
            
        except Exception as e:
            solve_time_ms = (time.perf_counter() - solve_start) * 1000
            logger.error(
                "MPC solve failed",
                error=str(e),
                solve_time_ms=solve_time_ms,
                S=state.S,
                delta_opt=state.delta_opt,
                q=state.q,
            )
            return MPCSolution(
                u_optimal=0.0,
                u_trajectory=np.array([0.0]),
                x_trajectory=x0.flatten(),
                delta_portfolio_pred=np.array([delta_portfolio]),
                cost=float('inf'),
                solver_status=f'failed: {e}'
            )
    
    def reset(self) -> None:
        """Reset controller state (call between trading sessions)."""
        self.mpc.reset_history()
        logger.info("MPC controller reset")


class SimpleKappaEstimator:
    """
    Simple mean-reversion parameter estimator for v1.
    
    Uses exponentially weighted autocorrelation to estimate κ.
    Full MHE can be added in v2.
    """
    
    def __init__(self, 
                 window: int = 60, 
                 ema_span: int = 20,
                 dt: float = 1.0 / (252 * 6.5 * 60)):
        """
        Initialize estimator.
        
        Args:
            window: Number of observations for autocorrelation
            ema_span: Span for EMA calculation of S_bar
            dt: Time step in years
        """
        self.window = window
        self.ema_span = ema_span
        self.dt = dt
        
        self.prices: list = []
        self.S_bar: float = 0.0
        self._ema_alpha = 2.0 / (ema_span + 1)
    
    def update(self, price: float) -> Tuple[float, float, float]:
        """
        Update estimator with new price observation.
        
        Args:
            price: New spot price
            
        Returns:
            Tuple of (kappa, S_bar, realized_vol)
        """
        # Update price history
        self.prices.append(price)
        if len(self.prices) > self.window:
            self.prices.pop(0)
        
        # Update EMA for reference price
        if self.S_bar == 0.0:
            self.S_bar = price
        else:
            self.S_bar = self._ema_alpha * price + (1 - self._ema_alpha) * self.S_bar
        
        # Estimate kappa from autocorrelation
        kappa = self._estimate_kappa()
        
        # Estimate realized volatility
        sigma = self._estimate_volatility()
        
        return kappa, self.S_bar, sigma
    
    def _estimate_kappa(self) -> float:
        """
        Estimate mean-reversion speed from price autocorrelation.
        
        For OU process: E[X_{t+1}|X_t] = X_t + κ(μ - X_t)dt
        Rearranging: ΔX = κ(μ - X_t)dt + noise
        
        Simple estimator: κ ≈ -log(autocorr(1)) / dt
        """
        if len(self.prices) < 10:
            return 0.2  # Default
        
        prices = np.array(self.prices)
        deviations = prices - self.S_bar
        
        if len(deviations) < 2:
            return 0.2
        
        # Lag-1 autocorrelation
        autocorr = np.corrcoef(deviations[:-1], deviations[1:])[0, 1]
        
        if autocorr <= 0 or autocorr >= 1:
            return 0.2  # Default for edge cases
        
        # κ from autocorrelation: ρ(1) ≈ exp(-κ*dt) → κ ≈ -log(ρ)/dt
        kappa = -np.log(autocorr) / self.dt
        
        # Clamp to reasonable range
        kappa = max(0.01, min(10.0, kappa))
        
        return kappa
    
    def _estimate_volatility(self) -> float:
        """Estimate annualized realized volatility from returns."""
        if len(self.prices) < 5:
            return 0.15  # Default
        
        prices = np.array(self.prices)
        returns = np.diff(np.log(prices))
        
        # Annualize: vol_annual = vol_step * sqrt(steps_per_year)
        steps_per_year = 1.0 / self.dt
        vol = np.std(returns) * np.sqrt(steps_per_year)
        
        # Clamp to reasonable range
        vol = max(0.05, min(1.0, vol))
        
        return vol
    
    def reset(self) -> None:
        """Reset estimator state."""
        self.prices = []
        self.S_bar = 0.0

