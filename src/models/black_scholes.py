"""
Black-Scholes option pricing and Greeks using CasADi symbolic framework.

This module provides both:
1. Symbolic (CasADi) implementations for use in MPC optimization
2. Numeric (NumPy) implementations for fast evaluation

The symbolic versions allow automatic differentiation, which is essential
for the MPC controller to compute gradients efficiently.

References:
    Black, F., & Scholes, M. (1973). The Pricing of Options and Corporate Liabilities.
    Journal of Political Economy, 81(3), 637-654.
"""

import casadi as ca
import numpy as np
from scipy import stats
from enum import Enum
from typing import Union, Tuple, Optional
from dataclasses import dataclass


class OptionType(str, Enum):
    """Option type enumeration."""
    CALL = "CALL"
    PUT = "PUT"


# Default dividend yield for SPY
# NOTE: This should be loaded from config/settings.yaml in production
# As of Dec 2025, SPY yield is approximately 1.06%
# This constant is a fallback default only
SPY_DIVIDEND_YIELD_DEFAULT = 0.0106


# =============================================================================
# Symbolic (CasADi) Implementation
# =============================================================================

class BlackScholesSymbolic:
    """
    Symbolic Black-Scholes implementation using CasADi.
    
    This class creates CasADi symbolic expressions for option pricing and Greeks.
    The expressions can be used directly in do-mpc models and automatically
    support differentiation.
    
    Note on T=0 handling:
        To avoid division by zero when T approaches 0, a minimum time threshold
        (T_MIN) is enforced in calculations. This ensures numerical stability
        in MPC optimization. Options should be rolled before reaching this threshold.
    
    Example:
        bs = BlackScholesSymbolic()
        
        # Create symbolic variables
        S = ca.MX.sym('S')  # Spot price
        K = ca.MX.sym('K')  # Strike
        T = ca.MX.sym('T')  # Time to expiry
        r = ca.MX.sym('r')  # Risk-free rate
        sigma = ca.MX.sym('sigma')  # Volatility
        
        # Get symbolic expressions
        price = bs.call_price(S, K, T, r, sigma)
        delta = bs.call_delta(S, K, T, r, sigma)
        
        # Create CasADi function for fast evaluation
        price_fn = ca.Function('call_price', [S, K, T, r, sigma], [price])
        
        # Evaluate
        result = price_fn(100, 100, 0.25, 0.05, 0.2)
    """
    
    # Minimum time to expiry (in years) to avoid division by zero
    # Equivalent to about 1 hour of trading time
    T_MIN = 1.0 / (365.25 * 24)  # ~0.000114 years
    
    def __init__(self):
        """Initialize the symbolic Black-Scholes calculator."""
        # Pre-create commonly used functions for efficiency
        self._create_standard_functions()
    
    def _create_standard_functions(self):
        """Create standard CasADi functions for common operations."""
        # Standard normal CDF approximation using error function
        # Φ(x) = 0.5 * (1 + erf(x / sqrt(2)))
        x = ca.MX.sym('x')
        self._norm_cdf_expr = 0.5 * (1 + ca.erf(x / ca.sqrt(2)))
        self._norm_cdf = ca.Function('norm_cdf', [x], [self._norm_cdf_expr])
        
        # Standard normal PDF
        # φ(x) = exp(-x²/2) / sqrt(2π)
        self._norm_pdf_expr = ca.exp(-x**2 / 2) / ca.sqrt(2 * ca.pi)
        self._norm_pdf = ca.Function('norm_pdf', [x], [self._norm_pdf_expr])
    
    def norm_cdf(self, x: ca.MX) -> ca.MX:
        """
        Standard normal cumulative distribution function.
        
        Uses the error function: Φ(x) = 0.5 * (1 + erf(x / √2))
        
        Args:
            x: CasADi symbolic expression
            
        Returns:
            Φ(x) as CasADi expression
        """
        return 0.5 * (1 + ca.erf(x / ca.sqrt(2)))
    
    def norm_pdf(self, x: ca.MX) -> ca.MX:
        """
        Standard normal probability density function.
        
        φ(x) = exp(-x²/2) / √(2π)
        
        Args:
            x: CasADi symbolic expression
            
        Returns:
            φ(x) as CasADi expression
        """
        return ca.exp(-x**2 / 2) / ca.sqrt(2 * ca.pi)
    
    def _safe_T(self, T: ca.MX) -> ca.MX:
        """
        Ensure T is at least T_MIN to avoid division by zero.
        
        Uses CasADi's fmax for smooth, differentiable clamping.
        """
        return ca.fmax(T, self.T_MIN)
    
    def d1(self, S: ca.MX, K: ca.MX, T: ca.MX, r: ca.MX, sigma: ca.MX, 
           q: ca.MX = 0.0) -> ca.MX:
        """
        Calculate d1 parameter for Black-Scholes formula with dividends.
        
        d1 = [ln(S/K) + (r - q + σ²/2)T] / (σ√T)
        
        Args:
            S: Spot price
            K: Strike price
            T: Time to expiration (years) - clamped to T_MIN minimum
            r: Risk-free interest rate
            sigma: Volatility (annualized)
            q: Continuous dividend yield (default 0)
            
        Returns:
            d1 as CasADi expression
        """
        T_safe = self._safe_T(T)
        return (ca.log(S / K) + (r - q + sigma**2 / 2) * T_safe) / (sigma * ca.sqrt(T_safe))
    
    def d2(self, S: ca.MX, K: ca.MX, T: ca.MX, r: ca.MX, sigma: ca.MX,
           q: ca.MX = 0.0) -> ca.MX:
        """
        Calculate d2 parameter for Black-Scholes formula with dividends.
        
        d2 = d1 - σ√T
        
        Args:
            S: Spot price
            K: Strike price
            T: Time to expiration (years) - clamped to T_MIN minimum
            r: Risk-free interest rate
            sigma: Volatility (annualized)
            q: Continuous dividend yield (default 0)
            
        Returns:
            d2 as CasADi expression
        """
        T_safe = self._safe_T(T)
        return self.d1(S, K, T, r, sigma, q) - sigma * ca.sqrt(T_safe)
    
    # -------------------------------------------------------------------------
    # Option Prices
    # -------------------------------------------------------------------------
    
    def call_price(self, S: ca.MX, K: ca.MX, T: ca.MX, r: ca.MX, sigma: ca.MX,
                   q: ca.MX = 0.0) -> ca.MX:
        """
        Black-Scholes call option price with dividends.
        
        C = S·e^(-qT)·Φ(d1) - K·e^(-rT)·Φ(d2)
        
        Args:
            S: Spot price
            K: Strike price
            T: Time to expiration (years)
            r: Risk-free interest rate
            sigma: Volatility (annualized)
            q: Continuous dividend yield (default 0)
            
        Returns:
            Call price as CasADi expression
        """
        d1 = self.d1(S, K, T, r, sigma, q)
        d2 = self.d2(S, K, T, r, sigma, q)
        return S * ca.exp(-q * T) * self.norm_cdf(d1) - K * ca.exp(-r * T) * self.norm_cdf(d2)
    
    def put_price(self, S: ca.MX, K: ca.MX, T: ca.MX, r: ca.MX, sigma: ca.MX,
                  q: ca.MX = 0.0) -> ca.MX:
        """
        Black-Scholes put option price with dividends.
        
        P = K·e^(-rT)·Φ(-d2) - S·e^(-qT)·Φ(-d1)
        
        Args:
            S: Spot price
            K: Strike price
            T: Time to expiration (years)
            r: Risk-free interest rate
            sigma: Volatility (annualized)
            q: Continuous dividend yield (default 0)
            
        Returns:
            Put price as CasADi expression
        """
        d1 = self.d1(S, K, T, r, sigma, q)
        d2 = self.d2(S, K, T, r, sigma, q)
        return K * ca.exp(-r * T) * self.norm_cdf(-d2) - S * ca.exp(-q * T) * self.norm_cdf(-d1)
    
    def option_price(self, S: ca.MX, K: ca.MX, T: ca.MX, r: ca.MX, sigma: ca.MX, 
                     is_call: bool = True) -> ca.MX:
        """
        Black-Scholes option price (call or put).
        
        Args:
            S: Spot price
            K: Strike price
            T: Time to expiration (years)
            r: Risk-free interest rate
            sigma: Volatility (annualized)
            is_call: True for call, False for put
            
        Returns:
            Option price as CasADi expression
        """
        if is_call:
            return self.call_price(S, K, T, r, sigma)
        else:
            return self.put_price(S, K, T, r, sigma)
    
    # -------------------------------------------------------------------------
    # First-Order Greeks
    # -------------------------------------------------------------------------
    
    def call_delta(self, S: ca.MX, K: ca.MX, T: ca.MX, r: ca.MX, sigma: ca.MX,
                   q: ca.MX = 0.0) -> ca.MX:
        """
        Call option delta with dividends: ∂C/∂S = e^(-qT)·Φ(d1)
        
        Delta represents the change in option price for a $1 change in underlying.
        
        Returns:
            Delta as CasADi expression (range: 0 to 1 for calls)
        """
        d1 = self.d1(S, K, T, r, sigma, q)
        return ca.exp(-q * T) * self.norm_cdf(d1)
    
    def put_delta(self, S: ca.MX, K: ca.MX, T: ca.MX, r: ca.MX, sigma: ca.MX,
                  q: ca.MX = 0.0) -> ca.MX:
        """
        Put option delta with dividends: ∂P/∂S = e^(-qT)·(Φ(d1) - 1)
        
        Returns:
            Delta as CasADi expression (range: -1 to 0 for puts)
        """
        d1 = self.d1(S, K, T, r, sigma, q)
        return ca.exp(-q * T) * (self.norm_cdf(d1) - 1)
    
    def delta(self, S: ca.MX, K: ca.MX, T: ca.MX, r: ca.MX, sigma: ca.MX,
              is_call: bool = True, q: ca.MX = 0.0) -> ca.MX:
        """Delta for call or put."""
        if is_call:
            return self.call_delta(S, K, T, r, sigma, q)
        else:
            return self.put_delta(S, K, T, r, sigma, q)
    
    def gamma(self, S: ca.MX, K: ca.MX, T: ca.MX, r: ca.MX, sigma: ca.MX,
              q: ca.MX = 0.0) -> ca.MX:
        """
        Option gamma with dividends: ∂²V/∂S² = e^(-qT)·φ(d1) / (S·σ·√T)
        
        Gamma is the same for calls and puts.
        Represents the rate of change of delta.
        
        Returns:
            Gamma as CasADi expression
        """
        T_safe = self._safe_T(T)
        d1 = self.d1(S, K, T, r, sigma, q)
        return ca.exp(-q * T_safe) * self.norm_pdf(d1) / (S * sigma * ca.sqrt(T_safe))
    
    def vega(self, S: ca.MX, K: ca.MX, T: ca.MX, r: ca.MX, sigma: ca.MX,
             q: ca.MX = 0.0) -> ca.MX:
        """
        Option vega with dividends: ∂V/∂σ = S·e^(-qT)·φ(d1)·√T
        
        Vega is the same for calls and puts.
        Represents sensitivity to volatility (per 1 unit change in σ).
        Note: Often quoted per 1% change, so divide by 100 for that.
        
        Returns:
            Vega as CasADi expression
        """
        T_safe = self._safe_T(T)
        d1 = self.d1(S, K, T, r, sigma, q)
        return S * ca.exp(-q * T_safe) * self.norm_pdf(d1) * ca.sqrt(T_safe)
    
    def call_theta(self, S: ca.MX, K: ca.MX, T: ca.MX, r: ca.MX, sigma: ca.MX,
                   q: ca.MX = 0.0) -> ca.MX:
        """
        Call option theta with dividends: -∂C/∂T
        
        Θ_call = -[S·e^(-qT)·φ(d1)·σ / (2√T)] - r·K·e^(-rT)·Φ(d2) + q·S·e^(-qT)·Φ(d1)
        
        Returns:
            Theta as CasADi expression (typically negative, per year)
        """
        T_safe = self._safe_T(T)
        d1 = self.d1(S, K, T, r, sigma, q)
        d2 = self.d2(S, K, T, r, sigma, q)
        
        term1 = -S * ca.exp(-q * T_safe) * self.norm_pdf(d1) * sigma / (2 * ca.sqrt(T_safe))
        term2 = -r * K * ca.exp(-r * T_safe) * self.norm_cdf(d2)
        term3 = q * S * ca.exp(-q * T_safe) * self.norm_cdf(d1)
        
        return term1 + term2 + term3
    
    def put_theta(self, S: ca.MX, K: ca.MX, T: ca.MX, r: ca.MX, sigma: ca.MX,
                  q: ca.MX = 0.0) -> ca.MX:
        """
        Put option theta with dividends: -∂P/∂T
        
        Θ_put = -[S·e^(-qT)·φ(d1)·σ / (2√T)] + r·K·e^(-rT)·Φ(-d2) - q·S·e^(-qT)·Φ(-d1)
        
        Returns:
            Theta as CasADi expression (per year)
        """
        T_safe = self._safe_T(T)
        d1 = self.d1(S, K, T, r, sigma, q)
        d2 = self.d2(S, K, T, r, sigma, q)
        
        term1 = -S * ca.exp(-q * T_safe) * self.norm_pdf(d1) * sigma / (2 * ca.sqrt(T_safe))
        term2 = r * K * ca.exp(-r * T_safe) * self.norm_cdf(-d2)
        term3 = -q * S * ca.exp(-q * T_safe) * self.norm_cdf(-d1)
        
        return term1 + term2 + term3
    
    def theta(self, S: ca.MX, K: ca.MX, T: ca.MX, r: ca.MX, sigma: ca.MX,
              is_call: bool = True, q: ca.MX = 0.0) -> ca.MX:
        """Theta for call or put."""
        if is_call:
            return self.call_theta(S, K, T, r, sigma, q)
        else:
            return self.put_theta(S, K, T, r, sigma, q)
    
    def rho(self, S: ca.MX, K: ca.MX, T: ca.MX, r: ca.MX, sigma: ca.MX,
            is_call: bool = True) -> ca.MX:
        """
        Option rho: ∂V/∂r
        
        ρ_call = K·T·e^(-rT)·Φ(d2)
        ρ_put = -K·T·e^(-rT)·Φ(-d2)
        
        Returns:
            Rho as CasADi expression
        """
        d2 = self.d2(S, K, T, r, sigma)
        
        if is_call:
            return K * T * ca.exp(-r * T) * self.norm_cdf(d2)
        else:
            return -K * T * ca.exp(-r * T) * self.norm_cdf(-d2)
    
    # -------------------------------------------------------------------------
    # Second-Order Greeks
    # -------------------------------------------------------------------------
    
    def vanna(self, S: ca.MX, K: ca.MX, T: ca.MX, r: ca.MX, sigma: ca.MX) -> ca.MX:
        """
        Vanna: ∂²V/∂S∂σ = ∂Δ/∂σ = ∂ν/∂S
        
        Vanna = -φ(d1)·d2 / σ
        
        Cross-gamma between spot and volatility.
        
        Returns:
            Vanna as CasADi expression
        """
        d1 = self.d1(S, K, T, r, sigma)
        d2 = self.d2(S, K, T, r, sigma)
        return -self.norm_pdf(d1) * d2 / sigma
    
    def charm(self, S: ca.MX, K: ca.MX, T: ca.MX, r: ca.MX, sigma: ca.MX,
              is_call: bool = True) -> ca.MX:
        """
        Charm (delta decay): ∂Δ/∂T = ∂Θ/∂S
        
        Rate of change of delta with respect to time.
        
        Returns:
            Charm as CasADi expression
        """
        d1 = self.d1(S, K, T, r, sigma)
        d2 = self.d2(S, K, T, r, sigma)
        
        term1 = self.norm_pdf(d1) * (2 * r * T - d2 * sigma * ca.sqrt(T))
        term1 = term1 / (2 * T * sigma * ca.sqrt(T))
        
        if is_call:
            return -term1
        else:
            return -term1
    
    def volga(self, S: ca.MX, K: ca.MX, T: ca.MX, r: ca.MX, sigma: ca.MX) -> ca.MX:
        """
        Volga (vomma): ∂²V/∂σ² = ∂ν/∂σ
        
        Volga = ν·d1·d2 / σ
        
        Sensitivity of vega to volatility.
        
        Returns:
            Volga as CasADi expression
        """
        d1 = self.d1(S, K, T, r, sigma)
        d2 = self.d2(S, K, T, r, sigma)
        vega = self.vega(S, K, T, r, sigma)
        return vega * d1 * d2 / sigma
    
    # -------------------------------------------------------------------------
    # Factory Methods for CasADi Functions
    # -------------------------------------------------------------------------
    
    def create_pricing_function(self, is_call: bool = True) -> ca.Function:
        """
        Create a CasADi Function for option pricing.
        
        Args:
            is_call: True for call, False for put
            
        Returns:
            CasADi Function: f(S, K, T, r, sigma) -> price
        """
        S = ca.MX.sym('S')
        K = ca.MX.sym('K')
        T = ca.MX.sym('T')
        r = ca.MX.sym('r')
        sigma = ca.MX.sym('sigma')
        
        price = self.option_price(S, K, T, r, sigma, is_call)
        name = 'call_price' if is_call else 'put_price'
        
        return ca.Function(name, [S, K, T, r, sigma], [price],
                          ['S', 'K', 'T', 'r', 'sigma'], ['price'])
    
    def create_greeks_function(self, is_call: bool = True) -> ca.Function:
        """
        Create a CasADi Function that returns all Greeks at once.
        
        Args:
            is_call: True for call, False for put
            
        Returns:
            CasADi Function: f(S, K, T, r, sigma) -> [delta, gamma, vega, theta, rho]
        """
        S = ca.MX.sym('S')
        K = ca.MX.sym('K')
        T = ca.MX.sym('T')
        r = ca.MX.sym('r')
        sigma = ca.MX.sym('sigma')
        
        delta = self.delta(S, K, T, r, sigma, is_call)
        gamma = self.gamma(S, K, T, r, sigma)
        vega = self.vega(S, K, T, r, sigma)
        theta = self.theta(S, K, T, r, sigma, is_call)
        rho_val = self.rho(S, K, T, r, sigma, is_call)
        
        name = 'call_greeks' if is_call else 'put_greeks'
        
        return ca.Function(name, 
                          [S, K, T, r, sigma], 
                          [delta, gamma, vega, theta, rho_val],
                          ['S', 'K', 'T', 'r', 'sigma'],
                          ['delta', 'gamma', 'vega', 'theta', 'rho'])
    
    def create_full_function(self, is_call: bool = True) -> ca.Function:
        """
        Create a CasADi Function that returns price and all Greeks.
        
        Args:
            is_call: True for call, False for put
            
        Returns:
            CasADi Function: f(S, K, T, r, sigma) -> [price, delta, gamma, vega, theta, rho]
        """
        S = ca.MX.sym('S')
        K = ca.MX.sym('K')
        T = ca.MX.sym('T')
        r = ca.MX.sym('r')
        sigma = ca.MX.sym('sigma')
        
        price = self.option_price(S, K, T, r, sigma, is_call)
        delta = self.delta(S, K, T, r, sigma, is_call)
        gamma = self.gamma(S, K, T, r, sigma)
        vega = self.vega(S, K, T, r, sigma)
        theta = self.theta(S, K, T, r, sigma, is_call)
        rho_val = self.rho(S, K, T, r, sigma, is_call)
        
        name = 'call_full' if is_call else 'put_full'
        
        return ca.Function(name,
                          [S, K, T, r, sigma],
                          [price, delta, gamma, vega, theta, rho_val],
                          ['S', 'K', 'T', 'r', 'sigma'],
                          ['price', 'delta', 'gamma', 'vega', 'theta', 'rho'])


# =============================================================================
# Numeric (NumPy) Implementation
# =============================================================================

@dataclass
class OptionGreeks:
    """Container for option Greeks values."""
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float
    
    # Second-order (optional)
    vanna: Optional[float] = None
    charm: Optional[float] = None
    volga: Optional[float] = None


class BlackScholesNumeric:
    """
    Numeric Black-Scholes implementation using NumPy/SciPy with dividend support.
    
    Fast evaluation for single options or arrays. Use this for:
    - Quick calculations outside MPC
    - Backtesting
    - Validation of symbolic implementation
    
    Example:
        bs = BlackScholesNumeric()
        
        # Single option (with dividends)
        price = bs.call_price(S=100, K=100, T=0.25, r=0.05, sigma=0.2, q=0.013)
        greeks = bs.call_greeks(S=100, K=100, T=0.25, r=0.05, sigma=0.2, q=0.013)
        
        # Without dividends (backward compatible)
        price = bs.call_price(S=100, K=100, T=0.25, r=0.05, sigma=0.2)
    """
    
    @staticmethod
    def d1(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
        """Calculate d1 parameter with dividends."""
        return (np.log(S / K) + (r - q + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    
    @staticmethod
    def d2(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
        """Calculate d2 parameter with dividends."""
        return BlackScholesNumeric.d1(S, K, T, r, sigma, q) - sigma * np.sqrt(T)
    
    # -------------------------------------------------------------------------
    # Prices
    # -------------------------------------------------------------------------
    
    @staticmethod
    def call_price(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
        """Black-Scholes call price with dividends."""
        if T <= 0:
            return max(S - K, 0)  # Intrinsic value at expiry
        
        d1 = BlackScholesNumeric.d1(S, K, T, r, sigma, q)
        d2 = BlackScholesNumeric.d2(S, K, T, r, sigma, q)
        
        return S * np.exp(-q * T) * stats.norm.cdf(d1) - K * np.exp(-r * T) * stats.norm.cdf(d2)
    
    @staticmethod
    def put_price(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
        """Black-Scholes put price with dividends."""
        if T <= 0:
            return max(K - S, 0)  # Intrinsic value at expiry
        
        d1 = BlackScholesNumeric.d1(S, K, T, r, sigma, q)
        d2 = BlackScholesNumeric.d2(S, K, T, r, sigma, q)
        
        return K * np.exp(-r * T) * stats.norm.cdf(-d2) - S * np.exp(-q * T) * stats.norm.cdf(-d1)
    
    # -------------------------------------------------------------------------
    # Greeks
    # -------------------------------------------------------------------------
    
    @staticmethod
    def call_delta(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
        """Call delta with dividends."""
        if T <= 0:
            return 1.0 if S > K else 0.0
        d1 = BlackScholesNumeric.d1(S, K, T, r, sigma, q)
        return np.exp(-q * T) * stats.norm.cdf(d1)
    
    @staticmethod
    def put_delta(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
        """Put delta with dividends."""
        if T <= 0:
            return -1.0 if S < K else 0.0
        d1 = BlackScholesNumeric.d1(S, K, T, r, sigma, q)
        return np.exp(-q * T) * (stats.norm.cdf(d1) - 1)
    
    @staticmethod
    def gamma(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
        """Option gamma with dividends (same for call and put)."""
        if T <= 0:
            return 0.0
        d1 = BlackScholesNumeric.d1(S, K, T, r, sigma, q)
        return np.exp(-q * T) * stats.norm.pdf(d1) / (S * sigma * np.sqrt(T))
    
    @staticmethod
    def vega(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
        """Option vega with dividends (same for call and put)."""
        if T <= 0:
            return 0.0
        d1 = BlackScholesNumeric.d1(S, K, T, r, sigma, q)
        return S * np.exp(-q * T) * stats.norm.pdf(d1) * np.sqrt(T)
    
    @staticmethod
    def call_theta(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
        """Call theta with dividends (per year)."""
        if T <= 0:
            return 0.0
        d1 = BlackScholesNumeric.d1(S, K, T, r, sigma, q)
        d2 = BlackScholesNumeric.d2(S, K, T, r, sigma, q)
        
        term1 = -S * np.exp(-q * T) * stats.norm.pdf(d1) * sigma / (2 * np.sqrt(T))
        term2 = -r * K * np.exp(-r * T) * stats.norm.cdf(d2)
        term3 = q * S * np.exp(-q * T) * stats.norm.cdf(d1)
        
        return term1 + term2 + term3
    
    @staticmethod
    def put_theta(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
        """Put theta with dividends (per year)."""
        if T <= 0:
            return 0.0
        d1 = BlackScholesNumeric.d1(S, K, T, r, sigma, q)
        d2 = BlackScholesNumeric.d2(S, K, T, r, sigma, q)
        
        term1 = -S * np.exp(-q * T) * stats.norm.pdf(d1) * sigma / (2 * np.sqrt(T))
        term2 = r * K * np.exp(-r * T) * stats.norm.cdf(-d2)
        term3 = -q * S * np.exp(-q * T) * stats.norm.cdf(-d1)
        
        return term1 + term2 + term3
    
    @staticmethod
    def call_rho(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
        """Call rho with dividends."""
        if T <= 0:
            return 0.0
        d2 = BlackScholesNumeric.d2(S, K, T, r, sigma, q)
        return K * T * np.exp(-r * T) * stats.norm.cdf(d2)
    
    @staticmethod
    def put_rho(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
        """Put rho with dividends."""
        if T <= 0:
            return 0.0
        d2 = BlackScholesNumeric.d2(S, K, T, r, sigma, q)
        return -K * T * np.exp(-r * T) * stats.norm.cdf(-d2)
    
    # -------------------------------------------------------------------------
    # Convenience Methods
    # -------------------------------------------------------------------------
    
    @classmethod
    def call_greeks(cls, S: float, K: float, T: float, r: float, sigma: float, 
                    q: float = 0.0) -> OptionGreeks:
        """Get all Greeks for a call option with dividends."""
        return OptionGreeks(
            delta=cls.call_delta(S, K, T, r, sigma, q),
            gamma=cls.gamma(S, K, T, r, sigma, q),
            vega=cls.vega(S, K, T, r, sigma, q),
            theta=cls.call_theta(S, K, T, r, sigma, q),
            rho=cls.call_rho(S, K, T, r, sigma, q),
        )
    
    @classmethod
    def put_greeks(cls, S: float, K: float, T: float, r: float, sigma: float,
                   q: float = 0.0) -> OptionGreeks:
        """Get all Greeks for a put option with dividends."""
        return OptionGreeks(
            delta=cls.put_delta(S, K, T, r, sigma, q),
            gamma=cls.gamma(S, K, T, r, sigma, q),
            vega=cls.vega(S, K, T, r, sigma, q),
            theta=cls.put_theta(S, K, T, r, sigma, q),
            rho=cls.put_rho(S, K, T, r, sigma, q),
        )


# =============================================================================
# Implied Volatility
# =============================================================================

def implied_volatility_newton(
    price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    is_call: bool = True,
    tol: float = 1e-6,
    max_iter: int = 100,
    initial_guess: float = 0.2
) -> Optional[float]:
    """
    Calculate implied volatility using Newton-Raphson method.
    
    Args:
        price: Market price of the option
        S: Spot price
        K: Strike price
        T: Time to expiration (years)
        r: Risk-free rate
        is_call: True for call, False for put
        tol: Tolerance for convergence
        max_iter: Maximum iterations
        initial_guess: Starting volatility guess
        
    Returns:
        Implied volatility, or None if not converged
    """
    if T <= 0:
        return None
    
    bs = BlackScholesNumeric()
    sigma = initial_guess
    
    for _ in range(max_iter):
        # Calculate price and vega
        if is_call:
            calc_price = bs.call_price(S, K, T, r, sigma)
        else:
            calc_price = bs.put_price(S, K, T, r, sigma)
        
        vega = bs.vega(S, K, T, r, sigma)
        
        # Check convergence
        diff = calc_price - price
        if abs(diff) < tol:
            return sigma
        
        # Avoid division by zero
        if vega < 1e-10:
            return None
        
        # Newton update
        sigma = sigma - diff / vega
        
        # Bound sigma to reasonable range
        sigma = max(0.001, min(sigma, 5.0))
    
    return None  # Did not converge


# =============================================================================
# Module-level convenience functions
# =============================================================================

# Create singleton instances
_bs_symbolic = None
_bs_numeric = None


def get_symbolic_bs() -> BlackScholesSymbolic:
    """Get singleton BlackScholesSymbolic instance."""
    global _bs_symbolic
    if _bs_symbolic is None:
        _bs_symbolic = BlackScholesSymbolic()
    return _bs_symbolic


def get_numeric_bs() -> BlackScholesNumeric:
    """Get singleton BlackScholesNumeric instance."""
    global _bs_numeric
    if _bs_numeric is None:
        _bs_numeric = BlackScholesNumeric()
    return _bs_numeric

