"""
Tests for the TradeStore SQLite persistence layer.

Tests session lifecycle, order/fill persistence, and recovery queries.
"""

import os
import pytest
import tempfile
from pathlib import Path
from datetime import datetime

from src.data.trade_store import TradeStore, StoredOrder, StoredFill, StoredSession


@pytest.fixture
def trade_store(tmp_path):
    """Create a TradeStore with a temporary database."""
    db_path = tmp_path / "test_trades.db"
    store = TradeStore(db_path)
    return store


class TestSessionLifecycle:
    """Tests for session management."""
    
    def test_start_session(self, trade_store):
        session_id = trade_store.start_session(mode="paper", initial_position=0)
        assert session_id  # Non-empty
        
        session = trade_store.get_session(session_id)
        assert session is not None
        assert session.mode == "paper"
        assert session.status == "running"
        assert session.initial_equity_position == 0
    
    def test_end_session(self, trade_store):
        session_id = trade_store.start_session(mode="paper")
        
        trade_store.end_session(
            session_id=session_id,
            final_position=50,
            realized_pnl=125.50,
            status="completed",
        )
        
        session = trade_store.get_session(session_id)
        assert session.status == "completed"
        assert session.final_equity_position == 50
        assert session.realized_pnl == 125.50
        assert session.end_time is not None
    
    def test_get_last_session(self, trade_store):
        import time
        s1 = trade_store.start_session(mode="paper")
        time.sleep(1.1)  # Ensure different session IDs (second-resolution)
        s2 = trade_store.start_session(mode="paper")
        
        last = trade_store.get_last_session()
        assert last.id == s2
    
    def test_get_nonexistent_session(self, trade_store):
        assert trade_store.get_session("nonexistent") is None


class TestOrderPersistence:
    """Tests for order storage and retrieval."""
    
    def _make_order(self, session_id, order_id="test-001", status="SUBMITTED"):
        return StoredOrder(
            id=order_id,
            symbol="SPY",
            side="BUY",
            quantity=100,
            order_type="MARKET",
            limit_price=None,
            status=status,
            broker_order_id=12345,
            submitted_at=datetime.now().isoformat(),
            filled_at=None,
            cancelled_at=None,
            filled_quantity=0,
            avg_fill_price=0.0,
            error_message=None,
            strategy="gamma_scalping",
            reason="Delta hedge",
            session_id=session_id,
        )
    
    def test_save_and_get_order(self, trade_store):
        sid = trade_store.start_session()
        order = self._make_order(sid)
        
        trade_store.save_order(order)
        
        retrieved = trade_store.get_order("test-001")
        assert retrieved is not None
        assert retrieved.symbol == "SPY"
        assert retrieved.quantity == 100
        assert retrieved.status == "SUBMITTED"
    
    def test_update_order_status(self, trade_store):
        sid = trade_store.start_session()
        order = self._make_order(sid)
        trade_store.save_order(order)
        
        trade_store.update_order_status(
            order_id="test-001",
            status="FILLED",
            filled_quantity=100,
            avg_fill_price=595.50,
        )
        
        retrieved = trade_store.get_order("test-001")
        assert retrieved.status == "FILLED"
        assert retrieved.filled_quantity == 100
        assert retrieved.avg_fill_price == 595.50
        assert retrieved.filled_at is not None
    
    def test_get_session_orders(self, trade_store):
        sid = trade_store.start_session()
        trade_store.save_order(self._make_order(sid, "order-1"))
        trade_store.save_order(self._make_order(sid, "order-2"))
        
        orders = trade_store.get_session_orders(sid)
        assert len(orders) == 2
    
    def test_get_open_orders(self, trade_store):
        sid = trade_store.start_session()
        trade_store.save_order(self._make_order(sid, "open-1", "SUBMITTED"))
        trade_store.save_order(self._make_order(sid, "filled-1", "FILLED"))
        
        open_orders = trade_store.get_open_orders(sid)
        assert len(open_orders) == 1
        assert open_orders[0].id == "open-1"


class TestFillPersistence:
    """Tests for fill storage and retrieval."""
    
    def test_save_and_get_fill(self, trade_store):
        sid = trade_store.start_session()
        
        fill = StoredFill(
            id=0,  # Auto-generated
            order_id="test-001",
            symbol="SPY",
            quantity=100,
            price=595.50,
            commission=0.65,
            timestamp=datetime.now().isoformat(),
            session_id=sid,
        )
        
        fill_id = trade_store.save_fill(fill)
        assert fill_id > 0
        
        fills = trade_store.get_session_fills(sid)
        assert len(fills) == 1
        assert fills[0].quantity == 100
        assert fills[0].price == 595.50
        assert fills[0].commission == 0.65
    
    def test_multiple_fills(self, trade_store):
        sid = trade_store.start_session()
        
        for i in range(5):
            fill = StoredFill(
                id=0,
                order_id=f"order-{i}",
                symbol="SPY",
                quantity=50,
                price=595.0 + i * 0.1,
                commission=0.65,
                timestamp=datetime.now().isoformat(),
                session_id=sid,
            )
            trade_store.save_fill(fill)
        
        fills = trade_store.get_session_fills(sid)
        assert len(fills) == 5


class TestSessionStats:
    """Tests for session statistics."""
    
    def test_session_stats(self, trade_store):
        sid = trade_store.start_session()
        
        # Add buy order
        buy_order = StoredOrder(
            id="buy-1", symbol="SPY", side="BUY", quantity=100,
            order_type="MARKET", limit_price=None, status="FILLED",
            broker_order_id=1, submitted_at=datetime.now().isoformat(),
            filled_at=datetime.now().isoformat(), cancelled_at=None,
            filled_quantity=100, avg_fill_price=595.0, error_message=None,
            strategy="gamma_scalping", reason="test", session_id=sid,
        )
        trade_store.save_order(buy_order)
        
        # Add fill
        fill = StoredFill(
            id=0, order_id="buy-1", symbol="SPY", quantity=100,
            price=595.0, commission=0.65,
            timestamp=datetime.now().isoformat(), session_id=sid,
        )
        trade_store.save_fill(fill)
        
        stats = trade_store.get_session_stats(sid)
        assert stats["total_orders"] == 1
        assert stats["filled_orders"] == 1
        assert stats["total_bought"] == 100
        assert stats["total_fills"] == 1
        assert stats["total_commission"] == 0.65
    
    def test_empty_session_stats(self, trade_store):
        sid = trade_store.start_session()
        stats = trade_store.get_session_stats(sid)
        assert stats["total_orders"] == 0
        assert stats["total_fills"] == 0
