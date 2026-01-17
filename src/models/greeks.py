"""
Portfolio-level Greeks aggregation and management.

This module provides utilities for:
1. Aggregating Greeks across multiple option positions
2. Converting between different Greek representations (per contract vs dollar)
3. Creating symbolic portfolio Greeks for MPC

The key insight is that portfolio Greeks are LINEAR in positions:
    Portfolio_Delta = Σ (position_i × delta_i × multiplier_i)

This linearity makes them easy to work with in the MPC framework.
"""

import casadi as ca
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional
from enum import Enum

from .black_scholes import BlackScholesSymbolic, BlackScholesNumeric, OptionGreeks


# =============================================================================
# Data Structures
# =============================================================================

class PositionType(str, Enum):
    """Type of position."""
    EQUITY = "EQUITY"
    CALL = "CALL"
    PUT = "PUT"


@dataclass
class OptionPosition:
    """
    A single option position with all relevant information.
    
    Attributes:
        symbol: Option symbol
        position_type: CALL or PUT
        strike: Strike price
        expiry: Expiration datetime
        quantity: Number of contracts (+ long, - short)
        multiplier: Contract multiplier (typically 100)
        current_price: Current option price
        underlying_price: Current price of underlying
        iv: Implied volatility
        greeks: Calculated Greeks for this position
        greeks_source: Source of Greeks data ("ibkr" or "calculated")
    """
    symbol: str
    position_type: PositionType
    strike: float
    expiry: datetime
    quantity: int  # + = long, - = short
    multiplier: int = 100
    
    # Market data
    current_price: float = 0.0
    underlying_price: float = 0.0
    iv: float = 0.0
    
    # Calculated Greeks (per contract, not per share)
    greeks: Optional[OptionGreeks] = None
    greeks_source: str = "calculated"  # "ibkr" or "calculated"
    
    def set_ibkr_greeks(
        self,
        delta: float,
        gamma: float,
        vega: float,
        theta: float,
        iv: float = 0.0,
    ) -> None:
        """
        Set Greeks from IBKR market data.
        
        IBKR provides model Greeks via TWS/Gateway which may be more accurate
        than our Black-Scholes calculation, especially for:
        - American options (early exercise)
        - Dividend adjustments
        - Interest rate environment
        
        Args:
            delta: Delta (per contract)
            gamma: Gamma (per contract)
            vega: Vega (per contract)
            theta: Theta (per contract)
            iv: Implied volatility
        """
        # Only use IBKR Greeks if they're non-zero (valid)
        if abs(delta) > 0 or abs(gamma) > 0:
            self.greeks = OptionGreeks(
                delta=delta,
                gamma=gamma,
                vega=vega,
                theta=theta,
                rho=0.0,  # IBKR doesn't typically provide rho
            )
            self.greeks_source = "ibkr"
            if iv > 0:
                self.iv = iv
    
    def has_valid_ibkr_greeks(self) -> bool:
        """Check if position has valid IBKR-provided Greeks."""
        return (
            self.greeks is not None 
            and self.greeks_source == "ibkr"
            and (abs(self.greeks.delta) > 0 or abs(self.greeks.gamma) > 0)
        )
    
    @property
    def is_call(self) -> bool:
        return self.position_type == PositionType.CALL
    
    @property
    def is_put(self) -> bool:
        return self.position_type == PositionType.PUT
    
    @property
    def time_to_expiry(self) -> float:
        """Time to expiry in years."""
        now = datetime.now()
        delta = self.expiry - now
        return max(delta.total_seconds() / (365.25 * 24 * 3600), 0.0)
    
    @property
    def notional_value(self) -> float:
        """Notional value of the position."""
        return abs(self.quantity) * self.multiplier * self.underlying_price


@dataclass
class EquityPosition:
    """
    An equity position (for delta hedging).
    
    Attributes:
        symbol: Stock symbol (e.g., "SPY")
        quantity: Number of shares (+ long, - short)
        current_price: Current stock price
    """
    symbol: str
    quantity: int  # + = long, - = short
    current_price: float = 0.0
    
    @property
    def delta(self) -> float:
        """Equity delta is simply the quantity (1 delta per share)."""
        return float(self.quantity)
    
    @property
    def market_value(self) -> float:
        """Current market value."""
        return self.quantity * self.current_price


@dataclass
class PortfolioGreeks:
    """
    Aggregated Greeks for the entire portfolio.
    
    All values are in "share equivalent" or dollar terms, not per-contract.
    This makes them directly comparable and usable for hedging calculations.
    """
    # First-order Greeks
    delta: float = 0.0          # Share equivalent (how many shares to hedge)
    gamma: float = 0.0          # Change in delta per $1 move
    vega: float = 0.0           # $ change per 1% vol move
    theta: float = 0.0          # $ decay per day
    rho: float = 0.0            # $ change per 1% rate move
    
    # Dollar-denominated versions
    delta_dollars: float = 0.0  # delta × spot price
    gamma_dollars: float = 0.0  # 0.5 × gamma × spot² (P&L from $1 move)
    
    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)
    underlying_price: float = 0.0
    
    # Position counts
    n_options: int = 0
    n_equity: int = 0
    
    def __repr__(self) -> str:
        return (
            f"PortfolioGreeks("
            f"Δ={self.delta:.1f}, "
            f"Γ={self.gamma:.4f}, "
            f"V={self.vega:.2f}, "
            f"Θ={self.theta:.2f}/day)"
        )


# =============================================================================
# Portfolio Greeks Calculator
# =============================================================================

class PortfolioGreeksCalculator:
    """
    Calculate aggregated portfolio Greeks from positions.
    
    This class handles:
    1. Calculating Greeks for individual options (using BlackScholesNumeric)
    2. Aggregating across all positions
    3. Converting to different representations (per share, per dollar)
    
    Example:
        calc = PortfolioGreeksCalculator(risk_free_rate=0.05, dividend_yield=0.013)
        
        # Add positions
        calc.add_option_position(OptionPosition(...))
        calc.add_equity_position(EquityPosition(...))
        
        # Get aggregated Greeks
        greeks = calc.calculate()
    """
    
    def __init__(self, risk_free_rate: float = 0.05, dividend_yield: float = 0.0106):
        """
        Initialize the calculator.
        
        Args:
            risk_free_rate: Risk-free interest rate for BS calculations
            dividend_yield: Continuous dividend yield (default ~1.06% for SPY as of Dec 2025)
                           Should be loaded from config in production.
        """
        self.risk_free_rate = risk_free_rate
        self.dividend_yield = dividend_yield
        self.option_positions: List[OptionPosition] = []
        self.equity_positions: List[EquityPosition] = []
        self._bs = BlackScholesNumeric()
    
    def clear(self):
        """Clear all positions."""
        self.option_positions.clear()
        self.equity_positions.clear()
    
    def add_option_position(self, position: OptionPosition):
        """Add an option position."""
        self.option_positions.append(position)
    
    def add_equity_position(self, position: EquityPosition):
        """Add an equity position."""
        self.equity_positions.append(position)
    
    def set_positions(self, 
                      options: List[OptionPosition],
                      equities: List[EquityPosition]):
        """Set all positions at once."""
        self.option_positions = list(options)
        self.equity_positions = list(equities)
    
    def _calculate_option_greeks(self, pos: OptionPosition) -> OptionGreeks:
        """
        Calculate Greeks for a single option position.
        
        Prefers IBKR-provided Greeks when available (more accurate for American options,
        dividend adjustments, etc.). Falls back to Black-Scholes calculation when
        IBKR Greeks are unavailable.
        """
        # Use IBKR Greeks if available and valid
        if pos.has_valid_ibkr_greeks():
            return pos.greeks
        
        # Fall back to Black-Scholes calculation
        S = pos.underlying_price
        K = pos.strike
        T = pos.time_to_expiry
        r = self.risk_free_rate
        q = self.dividend_yield
        sigma = pos.iv if pos.iv > 0 else 0.20  # Default IV if not provided
        
        if pos.is_call:
            return self._bs.call_greeks(S, K, T, r, sigma, q)
        else:
            return self._bs.put_greeks(S, K, T, r, sigma, q)
    
    def calculate(self, underlying_price: Optional[float] = None) -> PortfolioGreeks:
        """
        Calculate aggregated portfolio Greeks.
        
        Args:
            underlying_price: Override underlying price (uses positions' prices if None)
            
        Returns:
            PortfolioGreeks with aggregated values
        """
        result = PortfolioGreeks()
        
        # Determine underlying price
        if underlying_price is not None:
            spot = underlying_price
        elif self.option_positions:
            spot = self.option_positions[0].underlying_price
        elif self.equity_positions:
            spot = self.equity_positions[0].current_price
        else:
            spot = 0.0
        
        result.underlying_price = spot
        
        # Aggregate option Greeks
        for pos in self.option_positions:
            # Calculate Greeks for this option
            greeks = self._calculate_option_greeks(pos)
            pos.greeks = greeks  # Store on position for reference
            
            # Scale by position size and multiplier
            # Note: quantity is signed (+ long, - short)
            scale = pos.quantity * pos.multiplier
            
            result.delta += greeks.delta * scale
            result.gamma += greeks.gamma * scale
            result.vega += greeks.vega * scale / 100  # Convert to per 1% move
            result.theta += greeks.theta * scale / 365  # Convert to per day
            result.rho += greeks.rho * scale / 100  # Convert to per 1% move
        
        # Add equity delta
        for pos in self.equity_positions:
            result.delta += pos.quantity  # 1 delta per share
        
        # Calculate dollar-denominated Greeks
        if spot > 0:
            result.delta_dollars = result.delta * spot
            # Gamma dollars: expected P&L from $1 move = 0.5 × Γ × (ΔS)²
            # For convenience, we show gamma × spot (delta change per 1% move)
            result.gamma_dollars = result.gamma * spot * 0.01
        
        # Metadata
        result.timestamp = datetime.now()
        result.n_options = len(self.option_positions)
        result.n_equity = len(self.equity_positions)
        
        return result
    
    def delta_hedge_quantity(self, target_delta: float = 0.0) -> int:
        """
        Calculate shares of underlying needed to reach target delta.
        
        Args:
            target_delta: Target portfolio delta (default 0 = delta neutral)
            
        Returns:
            Number of shares to trade (+ buy, - sell)
        """
        current_greeks = self.calculate()
        delta_gap = target_delta - current_greeks.delta
        
        # Round to whole shares
        return int(round(delta_gap))


# =============================================================================
# Symbolic Portfolio Greeks for MPC
# =============================================================================

class SymbolicPortfolioGreeks:
    """
    Create CasADi symbolic expressions for portfolio Greeks.
    
    This is used in the MPC formulation where we need derivatives
    of portfolio Greeks with respect to positions and prices.
    
    The key expressions:
    - Portfolio delta as function of (S, positions)
    - Portfolio gamma as function of (S, positions)
    - How delta changes as we hedge (for MPC constraints)
    """
    
    def __init__(self):
        self._bs = BlackScholesSymbolic()
    
    def create_portfolio_delta_function(
        self,
        n_options: int,
        strikes: List[float],
        expiries: List[float],  # Time to expiry in years
        is_calls: List[bool],
        multiplier: int = 100
    ) -> ca.Function:
        """
        Create a CasADi function for portfolio delta.
        
        Args:
            n_options: Number of option positions
            strikes: List of strike prices (fixed)
            expiries: List of times to expiry in years (fixed for now)
            is_calls: List of booleans (True = call)
            multiplier: Contract multiplier
            
        Returns:
            CasADi Function:
                Inputs: S (spot), sigma (vol), r (rate), 
                        q_options (option quantities), q_equity (equity quantity)
                Output: portfolio_delta
        """
        # Symbolic variables
        S = ca.MX.sym('S')
        sigma = ca.MX.sym('sigma')
        r = ca.MX.sym('r')
        q_options = ca.MX.sym('q_options', n_options)  # Option quantities
        q_equity = ca.MX.sym('q_equity')  # Equity quantity
        
        # Calculate portfolio delta
        portfolio_delta = q_equity  # Start with equity
        
        for i in range(n_options):
            K = strikes[i]
            T = expiries[i]
            is_call = is_calls[i]
            
            # Get delta for this option
            if is_call:
                delta_i = self._bs.call_delta(S, K, T, r, sigma)
            else:
                delta_i = self._bs.put_delta(S, K, T, r, sigma)
            
            # Add to portfolio (scaled by quantity and multiplier)
            portfolio_delta += q_options[i] * multiplier * delta_i
        
        return ca.Function(
            'portfolio_delta',
            [S, sigma, r, q_options, q_equity],
            [portfolio_delta],
            ['S', 'sigma', 'r', 'q_options', 'q_equity'],
            ['delta']
        )
    
    def create_portfolio_gamma_function(
        self,
        n_options: int,
        strikes: List[float],
        expiries: List[float],
        multiplier: int = 100
    ) -> ca.Function:
        """
        Create a CasADi function for portfolio gamma.
        
        Note: Equity has zero gamma, so only options contribute.
        
        Returns:
            CasADi Function:
                Inputs: S (spot), sigma (vol), r (rate), q_options (quantities)
                Output: portfolio_gamma
        """
        S = ca.MX.sym('S')
        sigma = ca.MX.sym('sigma')
        r = ca.MX.sym('r')
        q_options = ca.MX.sym('q_options', n_options)
        
        portfolio_gamma = 0
        
        for i in range(n_options):
            K = strikes[i]
            T = expiries[i]
            
            gamma_i = self._bs.gamma(S, K, T, r, sigma)
            portfolio_gamma += q_options[i] * multiplier * gamma_i
        
        return ca.Function(
            'portfolio_gamma',
            [S, sigma, r, q_options],
            [portfolio_gamma],
            ['S', 'sigma', 'r', 'q_options'],
            ['gamma']
        )
    
    def create_greeks_function(
        self,
        n_options: int,
        strikes: List[float],
        expiries: List[float],
        is_calls: List[bool],
        multiplier: int = 100
    ) -> ca.Function:
        """
        Create a CasADi function returning all portfolio Greeks.
        
        Returns:
            CasADi Function:
                Inputs: S, sigma, r, q_options, q_equity
                Outputs: [delta, gamma, vega, theta]
        """
        S = ca.MX.sym('S')
        sigma = ca.MX.sym('sigma')
        r = ca.MX.sym('r')
        q_options = ca.MX.sym('q_options', n_options)
        q_equity = ca.MX.sym('q_equity')
        
        # Initialize
        portfolio_delta = q_equity
        portfolio_gamma = 0
        portfolio_vega = 0
        portfolio_theta = 0
        
        for i in range(n_options):
            K = strikes[i]
            T = expiries[i]
            is_call = is_calls[i]
            scale = q_options[i] * multiplier
            
            # Delta
            if is_call:
                delta_i = self._bs.call_delta(S, K, T, r, sigma)
                theta_i = self._bs.call_theta(S, K, T, r, sigma)
            else:
                delta_i = self._bs.put_delta(S, K, T, r, sigma)
                theta_i = self._bs.put_theta(S, K, T, r, sigma)
            
            # Gamma and Vega (same for calls and puts)
            gamma_i = self._bs.gamma(S, K, T, r, sigma)
            vega_i = self._bs.vega(S, K, T, r, sigma)
            
            # Aggregate
            portfolio_delta += scale * delta_i
            portfolio_gamma += scale * gamma_i
            portfolio_vega += scale * vega_i / 100  # Per 1% move
            portfolio_theta += scale * theta_i / 365  # Per day
        
        return ca.Function(
            'portfolio_greeks',
            [S, sigma, r, q_options, q_equity],
            [portfolio_delta, portfolio_gamma, portfolio_vega, portfolio_theta],
            ['S', 'sigma', 'r', 'q_options', 'q_equity'],
            ['delta', 'gamma', 'vega', 'theta']
        )


# =============================================================================
# Convenience Functions
# =============================================================================

def calculate_hedge_trade(
    current_delta: float,
    target_delta: float = 0.0,
    min_trade_size: int = 1
) -> int:
    """
    Calculate the equity trade needed to reach target delta.
    
    Args:
        current_delta: Current portfolio delta
        target_delta: Target portfolio delta (0 = delta neutral)
        min_trade_size: Minimum trade size (won't trade if below this)
        
    Returns:
        Number of shares to trade (+ buy, - sell)
    """
    delta_gap = target_delta - current_delta
    trade = int(round(delta_gap))
    
    if abs(trade) < min_trade_size:
        return 0
    
    return trade


def delta_neutral_hedge(portfolio_delta: float) -> int:
    """
    Calculate shares to trade for delta neutral.
    
    Simply negates the current delta.
    
    Args:
        portfolio_delta: Current portfolio delta
        
    Returns:
        Shares to trade (+ buy SPY if delta negative, - sell if positive)
    """
    return -int(round(portfolio_delta))

