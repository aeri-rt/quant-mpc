"""
Core type definitions for the quant-mpc system.

These dataclasses define the fundamental data structures used throughout
the system. They are designed to be:
- Immutable where possible (frozen=True for value types)
- Self-documenting with clear field names
- Serializable for logging and persistence
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional
import numpy as np


# =============================================================================
# Enums
# =============================================================================

class OptionType(str, Enum):
    """Option type: call or put."""
    CALL = "CALL"
    PUT = "PUT"


class OrderSide(str, Enum):
    """Order side: buy or sell."""
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    """Order type."""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"


class OrderStatus(str, Enum):
    """Order status."""
    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


class TradingMode(str, Enum):
    """Trading mode."""
    BACKTEST = "backtest"
    PAPER = "paper"
    LIVE = "live"


# =============================================================================
# Market Data Types
# =============================================================================

@dataclass(frozen=True)
class Quote:
    """A price quote for any instrument."""
    bid: float
    ask: float
    bid_size: int
    ask_size: int
    last: float
    last_size: int
    timestamp: datetime
    
    @property
    def mid(self) -> float:
        """Mid price."""
        return (self.bid + self.ask) / 2
    
    @property
    def spread(self) -> float:
        """Bid-ask spread."""
        return self.ask - self.bid
    
    @property
    def spread_pct(self) -> float:
        """Spread as percentage of mid."""
        return self.spread / self.mid if self.mid > 0 else 0.0


@dataclass(frozen=True)
class OptionQuote:
    """Quote for an option contract with Greeks."""
    symbol: str                    # Option symbol (e.g., "SPY240315C00500000")
    underlying: str                # Underlying symbol (e.g., "SPY")
    option_type: OptionType
    strike: float
    expiry: datetime
    
    # Quote data
    bid: float
    ask: float
    last: float
    volume: int
    open_interest: int
    timestamp: datetime
    
    # Greeks (from market data or calculated)
    delta: float = 0.0
    gamma: float = 0.0
    vega: float = 0.0
    theta: float = 0.0
    iv: float = 0.0                # Implied volatility
    
    @property
    def mid(self) -> float:
        """Mid price."""
        return (self.bid + self.ask) / 2
    
    @property
    def dte(self) -> int:
        """Days to expiration."""
        return (self.expiry.date() - datetime.now().date()).days
    
    @property
    def moneyness(self) -> float:
        """Moneyness: strike / spot. Requires spot price context."""
        # This would need spot price, leaving as placeholder
        return 0.0


# =============================================================================
# Position Types
# =============================================================================

@dataclass
class Position:
    """A position in an instrument."""
    symbol: str
    quantity: int                  # Positive = long, negative = short
    avg_cost: float               # Average cost per unit
    market_value: float           # Current market value
    unrealized_pnl: float         # Unrealized P&L
    realized_pnl: float           # Realized P&L
    
    # For options
    is_option: bool = False
    option_type: Optional[OptionType] = None
    strike: Optional[float] = None
    expiry: Optional[datetime] = None
    multiplier: int = 100         # Contract multiplier (100 for equity options)


@dataclass
class PortfolioGreeks:
    """Aggregated Greeks for the entire portfolio."""
    delta: float                  # Portfolio delta (share equivalent)
    gamma: float                  # Portfolio gamma
    vega: float                   # Portfolio vega ($ per 1% vol move)
    theta: float                  # Portfolio theta ($ per day)
    
    # Dollar-denominated versions
    delta_dollars: float = 0.0    # Delta * spot * multiplier
    gamma_dollars: float = 0.0    # Gamma * spot^2 * multiplier * 0.01
    
    timestamp: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Set timestamp if not provided."""
        if self.timestamp is None:
            self.timestamp = datetime.now()


# =============================================================================
# Market State
# =============================================================================

@dataclass
class MarketState:
    """
    Complete market state at a point in time.
    
    This is the central state container that gets updated by data providers
    and read by all other components.
    """
    timestamp: datetime
    
    # Underlying
    spy_price: float
    spy_bid: float
    spy_ask: float
    spy_volume: int = 0
    
    # Options chain (keyed by symbol)
    options: dict[str, OptionQuote] = field(default_factory=dict)
    
    # Current positions (keyed by symbol)
    positions: dict[str, Position] = field(default_factory=dict)
    
    # Aggregated portfolio Greeks
    portfolio_greeks: Optional[PortfolioGreeks] = None
    
    # Estimated parameters (from MHE)
    realized_vol: Optional[float] = None      # Estimated realized vol
    mean_reversion_kappa: Optional[float] = None  # Estimated κ
    
    # Account info
    cash: float = 0.0
    buying_power: float = 0.0
    margin_used: float = 0.0
    
    @property
    def spy_mid(self) -> float:
        """SPY mid price."""
        return (self.spy_bid + self.spy_ask) / 2
    
    def get_atm_options(self, expiry: Optional[datetime] = None) -> tuple[Optional[OptionQuote], Optional[OptionQuote]]:
        """
        Get ATM call and put for given expiry.
        
        Returns:
            Tuple of (call, put) or (None, None) if not found.
        """
        # Find strike closest to current price
        calls = []
        puts = []
        
        for opt in self.options.values():
            if expiry and opt.expiry.date() != expiry.date():
                continue
            if opt.option_type == OptionType.CALL:
                calls.append(opt)
            else:
                puts.append(opt)
        
        if not calls or not puts:
            return None, None
        
        # Find closest strike to spot
        target_strike = round(self.spy_price)  # Round to nearest dollar
        
        atm_call = min(calls, key=lambda x: abs(x.strike - target_strike), default=None)
        atm_put = min(puts, key=lambda x: abs(x.strike - target_strike), default=None)
        
        return atm_call, atm_put


# =============================================================================
# Control Types
# =============================================================================

@dataclass
class MPCOutput:
    """Output from the MPC controller."""
    hedge_quantity: float         # Shares of SPY to trade (+ = buy, - = sell)
    optimal_trajectory: np.ndarray  # Planned control over horizon [N x 1]
    predicted_states: np.ndarray    # Predicted states over horizon [N x n_states]
    
    # Solver info
    solve_time: float             # Time to solve (seconds)
    status: str                   # 'optimal', 'infeasible', 'max_iter', etc.
    objective_value: float        # Optimal objective value
    
    # For debugging
    iterations: int = 0
    
    @property
    def is_valid(self) -> bool:
        """Check if solution is valid."""
        return self.status in ('optimal', 'solved')


@dataclass
class MHEOutput:
    """Output from the Moving Horizon Estimator."""
    realized_vol: float           # Estimated realized volatility
    mean_reversion_kappa: float   # Estimated mean-reversion parameter
    
    # Confidence/uncertainty
    vol_std: float = 0.0          # Standard deviation of vol estimate
    kappa_std: float = 0.0        # Standard deviation of kappa estimate
    
    # Solver info
    solve_time: float = 0.0
    status: str = "unknown"


# =============================================================================
# Trading Types
# =============================================================================

@dataclass
class TradeSignal:
    """A signal to execute a trade."""
    instrument: str               # Symbol (SPY or option symbol)
    quantity: int                 # Number of shares/contracts (+ = buy, - = sell)
    side: OrderSide              # BUY or SELL
    order_type: OrderType        # MARKET, LIMIT, etc.
    limit_price: Optional[float] = None
    
    # Metadata
    strategy: str = "gamma_scalping"
    reason: str = ""              # Why this trade
    timestamp: datetime = field(default_factory=datetime.now)
    
    # For options
    is_option: bool = False
    
    def __post_init__(self):
        """Validate and set derived fields."""
        if self.quantity == 0:
            raise ValueError("Trade quantity cannot be zero")
        
        # Ensure side matches quantity sign
        if self.quantity > 0:
            self.side = OrderSide.BUY
        else:
            self.side = OrderSide.SELL


@dataclass
class Order:
    """An order submitted to the broker."""
    id: str                       # Internal order ID
    broker_id: Optional[str]      # Broker's order ID
    signal: TradeSignal          # Original signal
    
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: int = 0
    avg_fill_price: float = 0.0
    
    submitted_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None
    
    # Error info
    error_message: Optional[str] = None


@dataclass
class Fill:
    """A trade fill (execution)."""
    order_id: str
    instrument: str
    quantity: int
    price: float
    commission: float
    timestamp: datetime
    
    @property
    def notional(self) -> float:
        """Notional value of the fill."""
        return abs(self.quantity) * self.price

