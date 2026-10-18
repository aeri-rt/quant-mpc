"""
Abstract data provider interface.

Defines the contract that all data providers must implement,
allowing the system to work with different data sources
(IBKR, Polygon, historical data) without changing business logic.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Dict, List, Optional, Set

from src.core.types import Quote, OptionQuote, Position, OptionType


# =============================================================================
# Callbacks
# =============================================================================

# Type aliases for callbacks
QuoteCallback = Callable[[str, Quote], None]  # (symbol, quote)
OptionChainCallback = Callable[[str, List[OptionQuote]], None]  # (underlying, chain)
PositionCallback = Callable[[Dict[str, Position]], None]  # (all positions)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class DataProviderConfig:
    """Configuration for data providers."""
    
    # Connection settings
    host: str = "127.0.0.1"
    port: int = 7497  # TWS paper: 7497, TWS live: 7496, Gateway: 4001/4002
    client_id: int = 1
    
    # Timeout settings (seconds)
    connect_timeout: float = 10.0
    request_timeout: float = 30.0
    
    # Reconnection settings
    auto_reconnect: bool = True
    reconnect_delay: float = 5.0
    max_reconnect_attempts: int = 10
    
    # Data settings
    use_delayed_data: bool = False  # True for paper trading without market data subscription


# =============================================================================
# Abstract Provider
# =============================================================================

class DataProvider(ABC):
    """
    Abstract base class for market data providers.
    
    Implementations must provide:
    - Connection management (connect, disconnect, is_connected)
    - Quote retrieval (spot prices)
    - Options chain retrieval
    - Position retrieval
    - Real-time subscriptions
    """
    
    def __init__(self, config: Optional[DataProviderConfig] = None):
        """Initialize with configuration."""
        self.config = config or DataProviderConfig()
        self._quote_callbacks: List[QuoteCallback] = []
        self._option_callbacks: List[OptionChainCallback] = []
        self._position_callbacks: List[PositionCallback] = []
    
    # -------------------------------------------------------------------------
    # Connection Management
    # -------------------------------------------------------------------------
    
    @abstractmethod
    async def connect(self) -> None:
        """
        Establish connection to the data source.
        
        Raises:
            ConnectionError: If connection fails
        """
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from the data source."""
        pass
    
    @abstractmethod
    def is_connected(self) -> bool:
        """Check if currently connected."""
        pass
    
    # -------------------------------------------------------------------------
    # Spot Data
    # -------------------------------------------------------------------------
    
    @abstractmethod
    async def get_quote(self, symbol: str) -> Quote:
        """
        Get current quote for a symbol.
        
        Args:
            symbol: Stock symbol (e.g., "SPY")
            
        Returns:
            Quote with bid, ask, last, sizes, timestamp
            
        Raises:
            ValueError: If symbol not found
            ConnectionError: If not connected
        """
        pass
    
    @abstractmethod
    async def subscribe_quotes(self, symbols: List[str]) -> None:
        """
        Subscribe to real-time quotes for symbols.
        
        Updates will be delivered via registered callbacks.
        
        Args:
            symbols: List of stock symbols to subscribe
        """
        pass
    
    @abstractmethod
    async def unsubscribe_quotes(self, symbols: List[str]) -> None:
        """Unsubscribe from quote updates."""
        pass
    
    # -------------------------------------------------------------------------
    # Options Data
    # -------------------------------------------------------------------------
    
    @abstractmethod
    async def get_options_chain(
        self,
        underlying: str,
        expiry: Optional[datetime] = None,
        strike_range: Optional[tuple[float, float]] = None,
    ) -> List[OptionQuote]:
        """
        Get options chain for an underlying.
        
        Args:
            underlying: Underlying symbol (e.g., "SPY")
            expiry: Optional specific expiry to filter
            strike_range: Optional (min, max) strike range
            
        Returns:
            List of OptionQuote objects
        """
        pass
    
    @abstractmethod
    async def get_option_quote(
        self,
        underlying: str,
        expiry: datetime,
        strike: float,
        option_type: OptionType,
    ) -> OptionQuote:
        """
        Get quote for a specific option contract.
        
        Args:
            underlying: Underlying symbol
            expiry: Expiration date
            strike: Strike price
            option_type: CALL or PUT
            
        Returns:
            OptionQuote for the specific contract
        """
        pass
    
    @abstractmethod
    async def subscribe_options(
        self,
        underlying: str,
        expiry: Optional[datetime] = None,
    ) -> None:
        """
        Subscribe to real-time options chain updates.
        
        Args:
            underlying: Underlying symbol
            expiry: Optional specific expiry
        """
        pass
    
    # -------------------------------------------------------------------------
    # Position Data
    # -------------------------------------------------------------------------
    
    @abstractmethod
    async def get_positions(self) -> Dict[str, Position]:
        """
        Get current positions.
        
        Returns:
            Dict mapping symbol to Position
        """
        pass
    
    @abstractmethod
    async def get_account_info(self) -> dict:
        """
        Get account information (cash, buying power, etc.).
        
        Returns:
            Dict with account fields
        """
        pass
    
    # -------------------------------------------------------------------------
    # Historical Data
    # -------------------------------------------------------------------------
    
    @abstractmethod
    async def get_historical_bars(
        self,
        symbol: str,
        duration: str,  # e.g., "1 D", "1 W", "1 M"
        bar_size: str,  # e.g., "1 min", "5 mins", "1 hour"
        end_time: Optional[datetime] = None,
    ) -> List[dict]:
        """
        Get historical price bars.
        
        Args:
            symbol: Stock or option symbol
            duration: How far back to go
            bar_size: Bar granularity
            end_time: End of period (defaults to now)
            
        Returns:
            List of bar dicts with open, high, low, close, volume
        """
        pass
    
    # -------------------------------------------------------------------------
    # Callbacks
    # -------------------------------------------------------------------------
    
    def on_quote(self, callback: QuoteCallback) -> None:
        """Register a callback for quote updates."""
        self._quote_callbacks.append(callback)
    
    def on_option_chain(self, callback: OptionChainCallback) -> None:
        """Register a callback for options chain updates."""
        self._option_callbacks.append(callback)
    
    def on_positions(self, callback: PositionCallback) -> None:
        """Register a callback for position updates."""
        self._position_callbacks.append(callback)
    
    def _emit_quote(self, symbol: str, quote: Quote) -> None:
        """Emit quote update to all registered callbacks."""
        for callback in self._quote_callbacks:
            try:
                callback(symbol, quote)
            except Exception as e:
                # Log but don't fail
                pass
    
    def _emit_option_chain(self, underlying: str, chain: List[OptionQuote]) -> None:
        """Emit options chain update to all registered callbacks."""
        for callback in self._option_callbacks:
            try:
                callback(underlying, chain)
            except Exception as e:
                pass
    
    def _emit_positions(self, positions: Dict[str, Position]) -> None:
        """Emit position update to all registered callbacks."""
        for callback in self._position_callbacks:
            try:
                callback(positions)
            except Exception as e:
                pass

