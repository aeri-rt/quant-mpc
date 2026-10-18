"""
Data module for Quant-MPC.

Provides data providers and utilities for:
- Real-time market data
- Options chain management
- Historical data access
- Position sync

Main Components:
- DataProvider: Abstract interface for data sources
- IBKRDataProvider: Interactive Brokers implementation
- OptionsChainManager: Options chain caching and queries
"""

from .provider import (
    DataProvider,
    DataProviderConfig,
    QuoteCallback,
    OptionChainCallback,
    PositionCallback,
)
from .ibkr_provider import IBKRDataProvider, ConnectionState
from .options_chain import OptionsChainManager, OptionChainEntry, ExpiryChain
from .trade_store import (
    TradeStore,
    StoredOrder,
    StoredFill,
    StoredSession,
)

__all__ = [
    # Provider interface
    'DataProvider',
    'DataProviderConfig',
    'QuoteCallback',
    'OptionChainCallback',
    'PositionCallback',
    
    # Implementations
    'IBKRDataProvider',
    'ConnectionState',
    
    # Options chain
    'OptionsChainManager',
    'OptionChainEntry',
    'ExpiryChain',
    
    # Trade persistence
    'TradeStore',
    'StoredOrder',
    'StoredFill',
    'StoredSession',
]