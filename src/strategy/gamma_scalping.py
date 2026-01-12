"""
Gamma Scalping Strategy

Integrates Greeks calculation with MPC-based delta hedging to implement
a gamma scalping strategy for options portfolios.

The strategy:
1. Maintains a portfolio of options (typically long gamma)
2. Continuously delta-hedges using the MPC controller
3. Profits from gamma when price moves, theta decays on the other side

See docs/decisions/002-greeks-mpc-integration.md for design rationale.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Tuple
import numpy as np

from src.models.greeks import (
    PortfolioGreeksCalculator,
    PortfolioGreeks,
    OptionPosition,
    EquityPosition,
    PositionType,
)
from src.control.mpc_controller import (
    DeltaHedgeMPC,
    MPCConfig,
    MPCState,
    MPCSolution,
    SimpleKappaEstimator,
)
from src.utils.logging import get_logger, log_greeks, log_mpc_solve, Timer

logger = get_logger("strategy")


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class GammaScalpingConfig:
    """Configuration for the gamma scalping strategy."""
    
    # Greeks calculation
    risk_free_rate: float = 0.05
    dividend_yield: float = 0.0106  # SPY ~1.06% as of Dec 2025
    
    # EMA parameters for reference price
    ema_span: int = 20
    
    # Volatility estimation
    vol_window: int = 60  # Rolling window for realized vol
    
    # MPC parameters (forwarded to MPCConfig)
    mpc_horizon: int = 20
    mpc_q_max: float = 1000.0
    mpc_u_max: float = 200.0
    mpc_w_delta: float = 10.0
    mpc_w_exec: float = 1.0
    mpc_c_lin: float = 0.01
    mpc_kappa: float = 0.2
    
    # Minimum trade threshold (don't trade below this)
    min_trade_size: float = 1.0
    
    # Greeks update frequency
    update_greeks_every: int = 1  # Recalculate every N ticks


@dataclass
class HedgeRecommendation:
    """Output from the strategy: a hedge trade recommendation."""
    
    timestamp: datetime
    
    # Recommended action
    trade_quantity: float  # Shares to trade (+ buy, - sell)
    trade_direction: str   # "BUY", "SELL", or "HOLD"
    
    # Current state
    spot_price: float
    portfolio_delta: float  # Options delta
    equity_position: float  # Current hedge position
    net_delta: float        # delta_opt + q
    portfolio_gamma: float
    
    # MPC details
    mpc_status: str
    mpc_solve_time_ms: float
    direction_regime: str
    
    # Reasoning
    reason: str
    
    def __repr__(self) -> str:
        return (
            f"HedgeRecommendation({self.trade_direction} {abs(self.trade_quantity):.0f} shares, "
            f"net_Δ={self.net_delta:.1f}, Γ={self.portfolio_gamma:.2f})"
        )


# =============================================================================
# Strategy Implementation
# =============================================================================

class GammaScalpingStrategy:
    """
    Gamma scalping strategy with MPC-based delta hedging.
    
    Usage:
        # Initialize
        config = GammaScalpingConfig()
        strategy = GammaScalpingStrategy(config)
        
        # Set initial positions
        strategy.set_option_positions([
            OptionPosition(symbol="SPY240315C500", position_type=PositionType.CALL,
                          strike=500, expiry=..., quantity=10, iv=0.15, underlying_price=500)
        ])
        strategy.set_equity_position(symbol="SPY", quantity=0, price=500)
        
        # On each market update
        recommendation = strategy.update(spot_price=501.5)
        if recommendation.trade_quantity != 0:
            execute_trade(recommendation)
    """
    
    def __init__(self, config: Optional[GammaScalpingConfig] = None):
        """Initialize the strategy."""
        self.config = config or GammaScalpingConfig()
        
        # Initialize Greeks calculator
        self.greeks_calc = PortfolioGreeksCalculator(
            risk_free_rate=self.config.risk_free_rate,
            dividend_yield=self.config.dividend_yield,
        )
        
        # Initialize MPC controller
        mpc_config = MPCConfig(
            n_horizon=self.config.mpc_horizon,
            q_max=self.config.mpc_q_max,
            u_max=self.config.mpc_u_max,
            w_delta=self.config.mpc_w_delta,
            w_exec=self.config.mpc_w_exec,
            c_lin=self.config.mpc_c_lin,
            kappa=self.config.mpc_kappa,
        )
        self.mpc = DeltaHedgeMPC(mpc_config)
        
        # Parameter estimator
        self.kappa_estimator = SimpleKappaEstimator(
            window=self.config.vol_window,
            ema_span=self.config.ema_span,
        )
        
        # State
        self._option_positions: List[OptionPosition] = []
        self._equity_position: Optional[EquityPosition] = None
        self._last_greeks: Optional[PortfolioGreeks] = None
        self._tick_count: int = 0
        
        # EMA state
        self._ema_price: float = 0.0
        self._ema_initialized: bool = False
        
        logger.info(
            "GammaScalpingStrategy initialized",
            risk_free_rate=self.config.risk_free_rate,
            dividend_yield=self.config.dividend_yield,
            mpc_horizon=self.config.mpc_horizon,
        )
    
    # -------------------------------------------------------------------------
    # Position Management
    # -------------------------------------------------------------------------
    
    def set_option_positions(self, positions: List[OptionPosition]) -> None:
        """
        Set option positions for the strategy.
        
        Args:
            positions: List of OptionPosition objects
        """
        self._option_positions = list(positions)
        self.greeks_calc.clear()
        for pos in positions:
            self.greeks_calc.add_option_position(pos)
        
        logger.info(
            "Option positions updated",
            n_positions=len(positions),
            total_contracts=sum(p.quantity for p in positions),
        )
    
    def set_equity_position(self, symbol: str, quantity: int, price: float) -> None:
        """
        Set the current equity hedge position.
        
        Args:
            symbol: Equity symbol (e.g., "SPY")
            quantity: Number of shares (+ long, - short)
            price: Current price
        """
        self._equity_position = EquityPosition(
            symbol=symbol,
            quantity=quantity,
            current_price=price,
        )
        
        logger.debug(
            "Equity position set",
            symbol=symbol,
            quantity=quantity,
            price=price,
        )
    
    def update_option_prices(self, underlying_price: float, ivs: Optional[dict] = None) -> None:
        """
        Update underlying price and optionally IVs for all option positions.
        
        Args:
            underlying_price: Current spot price
            ivs: Optional dict mapping symbol to IV
        """
        for pos in self._option_positions:
            pos.underlying_price = underlying_price
            if ivs and pos.symbol in ivs:
                pos.iv = ivs[pos.symbol]
        
        if self._equity_position:
            self._equity_position.current_price = underlying_price
    
    # -------------------------------------------------------------------------
    # Core Strategy Logic
    # -------------------------------------------------------------------------
    
    def update(self, spot_price: float, ivs: Optional[dict] = None) -> HedgeRecommendation:
        """
        Update strategy with new market data and get hedge recommendation.
        
        This is the main entry point called on each market update.
        
        Args:
            spot_price: Current underlying spot price
            ivs: Optional dict of implied volatilities by symbol
            
        Returns:
            HedgeRecommendation with trade details
        """
        self._tick_count += 1
        
        # Update option prices
        self.update_option_prices(spot_price, ivs)
        
        # Update EMA
        self._update_ema(spot_price)
        
        # Update parameter estimates
        kappa_est, s_bar_est, sigma_est = self.kappa_estimator.update(spot_price)
        
        # Calculate portfolio Greeks
        with Timer("greeks_calculation", component="strategy", log_level="DEBUG"):
            greeks = self._calculate_greeks(spot_price)
        
        # Get current equity position
        q = self._equity_position.quantity if self._equity_position else 0.0
        
        # Build MPC state
        mpc_state = MPCState(
            S=spot_price,
            delta_opt=greeks.delta,  # Options delta only
            q=q,                      # Current equity hedge
            gamma_opt=greeks.gamma,   # Options gamma
            sigma=sigma_est,          # Estimated volatility
            S_bar=s_bar_est,          # Reference price
        )
        
        # Solve MPC
        with Timer("mpc_solve", component="strategy", log_level="DEBUG") as timer:
            solution = self.mpc.solve(mpc_state)
        
        # Determine direction regime for logging
        net_delta = greeks.delta + q
        if net_delta > self.mpc.config.direction_threshold:
            direction_regime = "SELL_ONLY"
        elif net_delta < -self.mpc.config.direction_threshold:
            direction_regime = "BUY_ONLY"
        else:
            direction_regime = "BOTH"
        
        # Apply minimum trade threshold
        trade_quantity = solution.u_optimal
        if abs(trade_quantity) < self.config.min_trade_size:
            trade_quantity = 0.0
        
        # Determine direction
        if trade_quantity > 0:
            trade_direction = "BUY"
        elif trade_quantity < 0:
            trade_direction = "SELL"
        else:
            trade_direction = "HOLD"
        
        # Generate reason
        reason = self._generate_reason(greeks, q, trade_quantity, direction_regime)
        
        # Build recommendation
        recommendation = HedgeRecommendation(
            timestamp=datetime.now(),
            trade_quantity=trade_quantity,
            trade_direction=trade_direction,
            spot_price=spot_price,
            portfolio_delta=greeks.delta,
            equity_position=q,
            net_delta=net_delta,
            portfolio_gamma=greeks.gamma,
            mpc_status=solution.solver_status,
            mpc_solve_time_ms=timer.elapsed_ms,
            direction_regime=direction_regime,
            reason=reason,
        )
        
        # Log the update
        self._log_update(greeks, mpc_state, solution, recommendation)
        
        return recommendation
    
    def execute_hedge(self, quantity: float) -> None:
        """
        Record that a hedge trade was executed.
        
        Call this after successfully executing the recommended trade
        to update internal state.
        
        Args:
            quantity: Shares traded (+ bought, - sold)
        """
        if self._equity_position:
            self._equity_position.quantity += int(quantity)
            logger.info(
                "Hedge executed",
                trade_quantity=quantity,
                new_position=self._equity_position.quantity,
            )
    
    # -------------------------------------------------------------------------
    # Internal Methods
    # -------------------------------------------------------------------------
    
    def _calculate_greeks(self, spot_price: float) -> PortfolioGreeks:
        """Calculate portfolio Greeks at current spot."""
        greeks = self.greeks_calc.calculate(underlying_price=spot_price)
        self._last_greeks = greeks
        return greeks
    
    def _update_ema(self, price: float) -> None:
        """Update exponential moving average of price."""
        if not self._ema_initialized:
            self._ema_price = price
            self._ema_initialized = True
        else:
            alpha = 2.0 / (self.config.ema_span + 1)
            self._ema_price = alpha * price + (1 - alpha) * self._ema_price
    
    def _generate_reason(
        self,
        greeks: PortfolioGreeks,
        q: float,
        trade_quantity: float,
        direction_regime: str,
    ) -> str:
        """Generate human-readable reason for the trade."""
        net_delta = greeks.delta + q
        
        if trade_quantity == 0:
            if abs(net_delta) < self.mpc.config.direction_threshold:
                return f"Delta hedged (Δ_net={net_delta:.1f})"
            else:
                return f"No trade (below min size, Δ_net={net_delta:.1f})"
        
        if trade_quantity > 0:
            return f"Buy {trade_quantity:.0f} to reduce short delta (Δ_net={net_delta:.1f}→{net_delta+trade_quantity:.1f})"
        else:
            return f"Sell {abs(trade_quantity):.0f} to reduce long delta (Δ_net={net_delta:.1f}→{net_delta+trade_quantity:.1f})"
    
    def _log_update(
        self,
        greeks: PortfolioGreeks,
        mpc_state: MPCState,
        solution: MPCSolution,
        recommendation: HedgeRecommendation,
    ) -> None:
        """Log the strategy update."""
        # Log Greeks
        log_greeks(
            delta=greeks.delta,
            gamma=greeks.gamma,
            theta=greeks.theta,
            vega=greeks.vega,
            n_options=len(self._option_positions),
            n_equity=1 if self._equity_position else 0,
            spot_price=greeks.underlying_price,
        )
        
        # Log at INFO level if there's a trade
        if recommendation.trade_quantity != 0:
            logger.info(
                f"Hedge recommendation: {recommendation.trade_direction} {abs(recommendation.trade_quantity):.0f}",
                spot=recommendation.spot_price,
                delta_opt=recommendation.portfolio_delta,
                equity_pos=recommendation.equity_position,
                net_delta=recommendation.net_delta,
                gamma=recommendation.portfolio_gamma,
                reason=recommendation.reason,
            )
    
    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------
    
    @property
    def current_greeks(self) -> Optional[PortfolioGreeks]:
        """Get the most recently calculated Greeks."""
        return self._last_greeks
    
    @property
    def equity_position(self) -> int:
        """Get current equity hedge position."""
        return self._equity_position.quantity if self._equity_position else 0
    
    @property
    def net_delta(self) -> float:
        """Get current net delta (options + equity)."""
        if self._last_greeks is None:
            return 0.0
        return self._last_greeks.delta + self.equity_position
    
    def get_state_summary(self) -> dict:
        """Get a summary of current strategy state."""
        return {
            "n_option_positions": len(self._option_positions),
            "equity_position": self.equity_position,
            "net_delta": self.net_delta,
            "gamma": self._last_greeks.gamma if self._last_greeks else 0.0,
            "ema_price": self._ema_price,
            "tick_count": self._tick_count,
        }

