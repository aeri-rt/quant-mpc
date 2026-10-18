"""
Tests for the TradingStateMachine.

Tests all valid transitions, invalid transitions, and callbacks.
"""

import pytest
from src.core.state_machine import (
    TradingStateMachine,
    TradingState,
    StateEvent,
)


class TestTradingStateMachine:
    """Tests for TradingStateMachine transitions and state checks."""
    
    def test_initial_state(self):
        sm = TradingStateMachine()
        assert sm.state == TradingState.INITIALIZING
        assert not sm.is_running
        assert not sm.can_trade
        assert not sm.is_stopped
    
    def test_happy_path(self):
        """Full lifecycle: INIT -> CONNECT -> POSITION -> RUN -> STOP -> STOPPED."""
        sm = TradingStateMachine()
        
        assert sm.trigger(StateEvent.INIT_COMPLETE)
        assert sm.state == TradingState.CONNECTING
        
        assert sm.trigger(StateEvent.CONNECTED)
        assert sm.state == TradingState.POSITIONING
        
        assert sm.trigger(StateEvent.POSITIONS_SET)
        assert sm.state == TradingState.RUNNING
        assert sm.is_running
        assert sm.can_trade
        
        assert sm.trigger(StateEvent.STOP)
        assert sm.state == TradingState.STOPPING
        assert not sm.can_trade
        
        assert sm.trigger(StateEvent.STOP)
        assert sm.state == TradingState.STOPPED
        assert sm.is_stopped
    
    def test_pause_resume(self):
        sm = TradingStateMachine()
        sm.trigger(StateEvent.INIT_COMPLETE)
        sm.trigger(StateEvent.CONNECTED)
        sm.trigger(StateEvent.POSITIONS_SET)
        assert sm.is_running
        
        assert sm.trigger(StateEvent.PAUSE)
        assert sm.is_paused
        assert not sm.can_trade
        
        assert sm.trigger(StateEvent.RESUME)
        assert sm.is_running
        assert sm.can_trade
    
    def test_connection_lost_from_running(self):
        sm = TradingStateMachine()
        sm.trigger(StateEvent.INIT_COMPLETE)
        sm.trigger(StateEvent.CONNECTED)
        sm.trigger(StateEvent.POSITIONS_SET)
        
        assert sm.trigger(StateEvent.CONNECTION_LOST)
        assert sm.state == TradingState.CONNECTING
    
    def test_fatal_error_from_running(self):
        sm = TradingStateMachine()
        sm.trigger(StateEvent.INIT_COMPLETE)
        sm.trigger(StateEvent.CONNECTED)
        sm.trigger(StateEvent.POSITIONS_SET)
        
        assert sm.trigger(StateEvent.FATAL_ERROR)
        assert sm.is_error
    
    def test_recoverable_error_stays_running(self):
        sm = TradingStateMachine()
        sm.trigger(StateEvent.INIT_COMPLETE)
        sm.trigger(StateEvent.CONNECTED)
        sm.trigger(StateEvent.POSITIONS_SET)
        
        assert sm.trigger(StateEvent.RECOVERABLE_ERROR)
        assert sm.is_running  # Should stay in RUNNING
    
    def test_invalid_transition_returns_false(self):
        sm = TradingStateMachine()
        # Can't go directly from INITIALIZING to RUNNING
        assert not sm.trigger(StateEvent.POSITIONS_SET)
        assert sm.state == TradingState.INITIALIZING  # Unchanged
    
    def test_error_to_stopped(self):
        sm = TradingStateMachine()
        sm.trigger(StateEvent.INIT_COMPLETE)
        sm.trigger(StateEvent.FATAL_ERROR)
        assert sm.is_error
        
        assert sm.trigger(StateEvent.STOP)
        assert sm.is_stopped
    
    def test_error_resume_restarts(self):
        sm = TradingStateMachine()
        sm.trigger(StateEvent.INIT_COMPLETE)
        sm.trigger(StateEvent.FATAL_ERROR)
        
        assert sm.trigger(StateEvent.RESUME)
        assert sm.state == TradingState.INITIALIZING
    
    def test_history_recorded(self):
        sm = TradingStateMachine()
        sm.trigger(StateEvent.INIT_COMPLETE)
        sm.trigger(StateEvent.CONNECTED)
        
        assert len(sm.history) == 2
        assert sm.history[0].from_state == TradingState.INITIALIZING
        assert sm.history[0].to_state == TradingState.CONNECTING
        assert sm.history[1].from_state == TradingState.CONNECTING
        assert sm.history[1].to_state == TradingState.POSITIONING
    
    def test_callback_called(self):
        sm = TradingStateMachine()
        transitions = []
        
        sm.on_state_change(lambda old, new, event: transitions.append((old, new, event)))
        
        sm.trigger(StateEvent.INIT_COMPLETE)
        
        assert len(transitions) == 1
        assert transitions[0] == (TradingState.INITIALIZING, TradingState.CONNECTING, StateEvent.INIT_COMPLETE)
    
    def test_get_valid_events(self):
        sm = TradingStateMachine()
        
        valid = sm.get_valid_events()
        assert StateEvent.INIT_COMPLETE in valid
        assert StateEvent.POSITIONS_SET not in valid
    
    def test_reset(self):
        sm = TradingStateMachine()
        sm.trigger(StateEvent.INIT_COMPLETE)
        sm.trigger(StateEvent.CONNECTED)
        
        sm.reset()
        assert sm.state == TradingState.INITIALIZING
        assert len(sm.history) == 0
    
    def test_stop_from_initializing(self):
        sm = TradingStateMachine()
        assert sm.trigger(StateEvent.STOP)
        assert sm.is_stopped
