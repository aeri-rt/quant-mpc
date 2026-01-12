"""
Event system for component communication.

Uses a simple publish-subscribe pattern for loose coupling between components.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Optional
from collections import defaultdict
import asyncio


class EventType(str, Enum):
    """Types of events in the system."""
    
    # Market data events
    QUOTE_UPDATE = "quote_update"
    OPTION_CHAIN_UPDATE = "option_chain_update"
    
    # Position events
    POSITION_UPDATE = "position_update"
    FILL_RECEIVED = "fill_received"
    
    # Control events
    MPC_SOLUTION = "mpc_solution"
    MHE_UPDATE = "mhe_update"
    
    # Strategy events
    SIGNAL_GENERATED = "signal_generated"
    
    # Order events
    ORDER_SUBMITTED = "order_submitted"
    ORDER_FILLED = "order_filled"
    ORDER_CANCELLED = "order_cancelled"
    ORDER_REJECTED = "order_rejected"
    
    # Risk events
    RISK_LIMIT_WARNING = "risk_limit_warning"
    RISK_LIMIT_BREACH = "risk_limit_breach"
    KILL_SWITCH = "kill_switch"
    
    # System events
    SYSTEM_START = "system_start"
    SYSTEM_STOP = "system_stop"
    ERROR = "error"


@dataclass
class Event:
    """An event that can be published and subscribed to."""
    type: EventType
    data: Any
    timestamp: datetime = field(default_factory=datetime.now)
    source: str = ""  # Component that generated the event


# Type alias for event handlers
EventHandler = Callable[[Event], None]
AsyncEventHandler = Callable[[Event], asyncio.coroutine]


class EventBus:
    """
    Simple event bus for publish-subscribe communication.
    
    Supports both sync and async handlers.
    
    Example:
        bus = EventBus()
        
        def on_quote(event):
            print(f"Got quote: {event.data}")
        
        bus.subscribe(EventType.QUOTE_UPDATE, on_quote)
        bus.publish(Event(EventType.QUOTE_UPDATE, data={'price': 500.0}))
    """
    
    def __init__(self):
        self._handlers: dict[EventType, list[EventHandler]] = defaultdict(list)
        self._async_handlers: dict[EventType, list[AsyncEventHandler]] = defaultdict(list)
    
    def subscribe(self, event_type: EventType, handler: EventHandler) -> None:
        """
        Subscribe to an event type with a sync handler.
        
        Args:
            event_type: Type of event to subscribe to
            handler: Callback function that takes an Event
        """
        self._handlers[event_type].append(handler)
    
    def subscribe_async(self, event_type: EventType, handler: AsyncEventHandler) -> None:
        """
        Subscribe to an event type with an async handler.
        
        Args:
            event_type: Type of event to subscribe to
            handler: Async callback function that takes an Event
        """
        self._async_handlers[event_type].append(handler)
    
    def unsubscribe(self, event_type: EventType, handler: EventHandler) -> None:
        """Unsubscribe a handler from an event type."""
        if handler in self._handlers[event_type]:
            self._handlers[event_type].remove(handler)
        if handler in self._async_handlers[event_type]:
            self._async_handlers[event_type].remove(handler)
    
    def publish(self, event: Event) -> None:
        """
        Publish an event to all sync subscribers.
        
        Args:
            event: Event to publish
        """
        for handler in self._handlers[event.type]:
            try:
                handler(event)
            except Exception as e:
                # Log error but don't stop other handlers
                print(f"Error in event handler: {e}")
    
    async def publish_async(self, event: Event) -> None:
        """
        Publish an event to all subscribers (sync and async).
        
        Args:
            event: Event to publish
        """
        # Call sync handlers
        self.publish(event)
        
        # Call async handlers
        tasks = []
        for handler in self._async_handlers[event.type]:
            tasks.append(handler(event))
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    def clear(self, event_type: Optional[EventType] = None) -> None:
        """
        Clear handlers for an event type, or all handlers.
        
        Args:
            event_type: Specific event type to clear, or None for all
        """
        if event_type:
            self._handlers[event_type].clear()
            self._async_handlers[event_type].clear()
        else:
            self._handlers.clear()
            self._async_handlers.clear()


# Global event bus instance (singleton pattern)
_event_bus: Optional[EventBus] = None


def get_event_bus() -> EventBus:
    """Get the global event bus instance."""
    global _event_bus
    if _event_bus is None:
        _event_bus = EventBus()
    return _event_bus

