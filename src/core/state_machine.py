"""
Trading State Machine for managing application lifecycle.

Provides a formal state machine with clear transitions to manage
the trading loop's lifecycle states.

States:
    INITIALIZING -> CONNECTING -> POSITIONING -> RUNNING -> PAUSED -> STOPPING -> STOPPED
    
Transitions can be triggered by events (user signals, errors, etc.)
"""

from enum import Enum
from typing import Callable, Dict, List, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime

from src.utils.logging import get_logger

logger = get_logger("state_machine")


class TradingState(str, Enum):
    """Trading application states."""
    
    INITIALIZING = "INITIALIZING"  # Loading config, setting up components
    CONNECTING = "CONNECTING"      # Connecting to broker
    POSITIONING = "POSITIONING"    # Setting up initial positions
    RUNNING = "RUNNING"            # Active trading loop
    PAUSED = "PAUSED"              # Temporarily paused (can resume)
    STOPPING = "STOPPING"          # Graceful shutdown in progress
    STOPPED = "STOPPED"            # Fully stopped
    ERROR = "ERROR"                # Error state (requires intervention)


class StateEvent(str, Enum):
    """Events that can trigger state transitions."""
    
    # Success events
    INIT_COMPLETE = "INIT_COMPLETE"        # Initialization done
    CONNECTED = "CONNECTED"                # Broker connected
    POSITIONS_SET = "POSITIONS_SET"        # Initial positions established
    
    # User control events
    PAUSE = "PAUSE"                        # User requests pause
    RESUME = "RESUME"                      # User requests resume
    STOP = "STOP"                          # User requests stop
    
    # Error events
    CONNECTION_LOST = "CONNECTION_LOST"    # Broker disconnected
    FATAL_ERROR = "FATAL_ERROR"            # Unrecoverable error
    RECOVERABLE_ERROR = "RECOVERABLE_ERROR"  # Error but can continue


@dataclass
class StateTransition:
    """Record of a state transition."""
    from_state: TradingState
    to_state: TradingState
    event: StateEvent
    timestamp: datetime = field(default_factory=datetime.now)
    details: str = ""


# Valid state transitions
# Format: (from_state, event) -> to_state
VALID_TRANSITIONS: Dict[tuple, TradingState] = {
    # From INITIALIZING
    (TradingState.INITIALIZING, StateEvent.INIT_COMPLETE): TradingState.CONNECTING,
    (TradingState.INITIALIZING, StateEvent.FATAL_ERROR): TradingState.ERROR,
    (TradingState.INITIALIZING, StateEvent.STOP): TradingState.STOPPED,
    
    # From CONNECTING
    (TradingState.CONNECTING, StateEvent.CONNECTED): TradingState.POSITIONING,
    (TradingState.CONNECTING, StateEvent.CONNECTION_LOST): TradingState.ERROR,
    (TradingState.CONNECTING, StateEvent.FATAL_ERROR): TradingState.ERROR,
    (TradingState.CONNECTING, StateEvent.STOP): TradingState.STOPPED,
    
    # From POSITIONING
    (TradingState.POSITIONING, StateEvent.POSITIONS_SET): TradingState.RUNNING,
    (TradingState.POSITIONING, StateEvent.CONNECTION_LOST): TradingState.CONNECTING,
    (TradingState.POSITIONING, StateEvent.FATAL_ERROR): TradingState.ERROR,
    (TradingState.POSITIONING, StateEvent.STOP): TradingState.STOPPING,
    
    # From RUNNING
    (TradingState.RUNNING, StateEvent.PAUSE): TradingState.PAUSED,
    (TradingState.RUNNING, StateEvent.STOP): TradingState.STOPPING,
    (TradingState.RUNNING, StateEvent.CONNECTION_LOST): TradingState.CONNECTING,
    (TradingState.RUNNING, StateEvent.FATAL_ERROR): TradingState.ERROR,
    (TradingState.RUNNING, StateEvent.RECOVERABLE_ERROR): TradingState.RUNNING,  # Stay in RUNNING
    
    # From PAUSED
    (TradingState.PAUSED, StateEvent.RESUME): TradingState.RUNNING,
    (TradingState.PAUSED, StateEvent.STOP): TradingState.STOPPING,
    (TradingState.PAUSED, StateEvent.CONNECTION_LOST): TradingState.CONNECTING,
    (TradingState.PAUSED, StateEvent.FATAL_ERROR): TradingState.ERROR,
    
    # From STOPPING
    (TradingState.STOPPING, StateEvent.STOP): TradingState.STOPPED,  # Shutdown complete
    
    # From ERROR
    (TradingState.ERROR, StateEvent.STOP): TradingState.STOPPED,
    (TradingState.ERROR, StateEvent.RESUME): TradingState.INITIALIZING,  # Restart from beginning
}


class TradingStateMachine:
    """
    State machine for trading application lifecycle.
    
    Manages transitions between states and notifies registered callbacks
    when state changes occur.
    
    Usage:
        sm = TradingStateMachine()
        
        # Register callback for state changes
        sm.on_state_change(lambda old, new, event: print(f"{old} -> {new}"))
        
        # Trigger transitions
        sm.trigger(StateEvent.INIT_COMPLETE)  # INITIALIZING -> CONNECTING
        sm.trigger(StateEvent.CONNECTED)       # CONNECTING -> POSITIONING
        sm.trigger(StateEvent.POSITIONS_SET)   # POSITIONING -> RUNNING
        
        # Check state
        if sm.is_running:
            # ... do trading ...
        
        # Pause/resume
        sm.trigger(StateEvent.PAUSE)
        sm.trigger(StateEvent.RESUME)
        
        # Graceful shutdown
        sm.trigger(StateEvent.STOP)
    """
    
    def __init__(self, initial_state: TradingState = TradingState.INITIALIZING):
        """Initialize state machine."""
        self._state = initial_state
        self._history: List[StateTransition] = []
        self._callbacks: List[Callable[[TradingState, TradingState, StateEvent], None]] = []
        
        logger.info(f"State machine initialized in {initial_state.value}")
    
    @property
    def state(self) -> TradingState:
        """Current state."""
        return self._state
    
    @property
    def is_running(self) -> bool:
        """Check if in RUNNING state."""
        return self._state == TradingState.RUNNING
    
    @property
    def is_paused(self) -> bool:
        """Check if in PAUSED state."""
        return self._state == TradingState.PAUSED
    
    @property
    def is_stopped(self) -> bool:
        """Check if in STOPPED state."""
        return self._state == TradingState.STOPPED
    
    @property
    def is_error(self) -> bool:
        """Check if in ERROR state."""
        return self._state == TradingState.ERROR
    
    @property
    def can_trade(self) -> bool:
        """Check if trading is allowed (RUNNING state)."""
        return self._state == TradingState.RUNNING
    
    @property
    def history(self) -> List[StateTransition]:
        """Get transition history."""
        return list(self._history)
    
    def trigger(self, event: StateEvent, details: str = "") -> bool:
        """
        Trigger a state transition.
        
        Args:
            event: Event that triggers the transition
            details: Optional details about the event
            
        Returns:
            True if transition was successful, False if invalid
        """
        key = (self._state, event)
        
        if key not in VALID_TRANSITIONS:
            logger.warning(
                f"Invalid transition: {self._state.value} + {event.value}",
                current_state=self._state.value,
                event=event.value,
            )
            return False
        
        old_state = self._state
        new_state = VALID_TRANSITIONS[key]
        
        # Record transition
        transition = StateTransition(
            from_state=old_state,
            to_state=new_state,
            event=event,
            details=details,
        )
        self._history.append(transition)
        
        # Update state
        self._state = new_state
        
        logger.info(
            f"State transition: {old_state.value} -> {new_state.value}",
            event=event.value,
            details=details,
        )
        
        # Notify callbacks
        for callback in self._callbacks:
            try:
                callback(old_state, new_state, event)
            except Exception as e:
                logger.error(f"State change callback error: {e}")
        
        return True
    
    def on_state_change(
        self,
        callback: Callable[[TradingState, TradingState, StateEvent], None],
    ) -> None:
        """
        Register callback for state changes.
        
        Callback signature: callback(old_state, new_state, event)
        """
        self._callbacks.append(callback)
    
    def get_valid_events(self) -> Set[StateEvent]:
        """Get events that are valid from the current state."""
        valid = set()
        for (state, event), _ in VALID_TRANSITIONS.items():
            if state == self._state:
                valid.add(event)
        return valid
    
    def reset(self) -> None:
        """Reset state machine to INITIALIZING."""
        self._state = TradingState.INITIALIZING
        self._history.clear()
        logger.info("State machine reset")
