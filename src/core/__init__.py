"""
Core module: fundamental types, events, and exceptions.
"""

from .types import (
    OptionType,
    OrderSide,
    OrderType,
    Position,
    OptionQuote,
    PortfolioGreeks,
    MarketState,
    MPCOutput,
    TradeSignal,
)
from .exceptions import (
    QuantMPCError,
    DataError,
    BrokerError,
    ConfigError,
    ControlError,
)
from .state_machine import (
    TradingState,
    StateEvent,
    TradingStateMachine,
    StateTransition,
)

__all__ = [
    # Types
    "OptionType",
    "OrderSide",
    "OrderType",
    "Position",
    "OptionQuote",
    "PortfolioGreeks",
    "MarketState",
    "MPCOutput",
    "TradeSignal",
    # Exceptions
    "QuantMPCError",
    "DataError",
    "BrokerError",
    "ConfigError",
    "ControlError",
    # State machine
    "TradingState",
    "StateEvent",
    "TradingStateMachine",
    "StateTransition",
]

