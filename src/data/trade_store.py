"""
Trade persistence layer using SQLite.

Provides durable storage for:
- Orders (submitted, filled, cancelled)
- Fills (execution details)
- Position history
- Session metadata

This enables recovery on restart and provides an audit trail.
"""

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import json

from src.utils.logging import get_logger

logger = get_logger("trade_store")

# Default database location
DEFAULT_DB_PATH = Path("data/trades.db")


@dataclass
class StoredOrder:
    """Order record for persistence."""
    id: str
    symbol: str
    side: str  # BUY or SELL
    quantity: int
    order_type: str
    limit_price: Optional[float]
    status: str
    broker_order_id: Optional[int]
    submitted_at: Optional[str]
    filled_at: Optional[str]
    cancelled_at: Optional[str]
    filled_quantity: int
    avg_fill_price: float
    error_message: Optional[str]
    strategy: str
    reason: str
    session_id: str


@dataclass
class StoredFill:
    """Fill record for persistence."""
    id: int  # Auto-increment
    order_id: str
    symbol: str
    quantity: int
    price: float
    commission: float
    timestamp: str
    session_id: str


@dataclass
class StoredSession:
    """Session metadata for persistence."""
    id: str
    start_time: str
    end_time: Optional[str]
    mode: str  # paper, live
    initial_equity_position: int
    final_equity_position: Optional[int]
    total_orders: int
    total_fills: int
    realized_pnl: float
    status: str  # running, completed, error


class TradeStore:
    """
    SQLite-based trade persistence.
    
    Usage:
        store = TradeStore()
        
        # Start a session
        session_id = store.start_session(mode="paper")
        
        # Save orders
        store.save_order(order)
        
        # Update order status
        store.update_order_status(order_id, "FILLED", filled_qty=100, avg_price=595.50)
        
        # Save fills
        store.save_fill(fill)
        
        # End session
        store.end_session(session_id, final_position=100, realized_pnl=150.00)
        
        # Recover on restart
        orders = store.get_session_orders(session_id)
    """
    
    def __init__(self, db_path: Optional[Path] = None):
        """
        Initialize trade store.
        
        Args:
            db_path: Path to SQLite database file. Defaults to data/trades.db
        """
        self.db_path = db_path or DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self._init_database()
        logger.info(f"TradeStore initialized at {self.db_path}")
    
    @contextmanager
    def _get_connection(self):
        """Get database connection with context manager."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
    
    def _init_database(self) -> None:
        """Initialize database schema."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Sessions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    start_time TEXT NOT NULL,
                    end_time TEXT,
                    mode TEXT NOT NULL,
                    initial_equity_position INTEGER DEFAULT 0,
                    final_equity_position INTEGER,
                    total_orders INTEGER DEFAULT 0,
                    total_fills INTEGER DEFAULT 0,
                    realized_pnl REAL DEFAULT 0,
                    status TEXT DEFAULT 'running'
                )
            """)
            
            # Orders table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS orders (
                    id TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    quantity INTEGER NOT NULL,
                    order_type TEXT NOT NULL,
                    limit_price REAL,
                    status TEXT NOT NULL,
                    broker_order_id INTEGER,
                    submitted_at TEXT,
                    filled_at TEXT,
                    cancelled_at TEXT,
                    filled_quantity INTEGER DEFAULT 0,
                    avg_fill_price REAL DEFAULT 0,
                    error_message TEXT,
                    strategy TEXT,
                    reason TEXT,
                    session_id TEXT NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES sessions(id)
                )
            """)
            
            # Fills table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS fills (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    order_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    quantity INTEGER NOT NULL,
                    price REAL NOT NULL,
                    commission REAL DEFAULT 0,
                    timestamp TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    FOREIGN KEY (order_id) REFERENCES orders(id),
                    FOREIGN KEY (session_id) REFERENCES sessions(id)
                )
            """)
            
            # Create indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_orders_session ON orders(session_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_orders_status ON orders(status)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_fills_order ON fills(order_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_fills_session ON fills(session_id)")
            
            logger.debug("Database schema initialized")
    
    # -------------------------------------------------------------------------
    # Session Management
    # -------------------------------------------------------------------------
    
    def start_session(
        self,
        mode: str = "paper",
        initial_position: int = 0,
    ) -> str:
        """
        Start a new trading session.
        
        Args:
            mode: Trading mode (paper, live)
            initial_position: Initial equity position
            
        Returns:
            Session ID
        """
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO sessions (id, start_time, mode, initial_equity_position, status)
                VALUES (?, ?, ?, ?, 'running')
            """, (session_id, datetime.now().isoformat(), mode, initial_position))
        
        logger.info(f"Started session {session_id} (mode={mode})")
        return session_id
    
    def end_session(
        self,
        session_id: str,
        final_position: Optional[int] = None,
        realized_pnl: float = 0.0,
        status: str = "completed",
    ) -> None:
        """
        End a trading session.
        
        Args:
            session_id: Session to end
            final_position: Final equity position
            realized_pnl: Total realized P&L
            status: Final status (completed, error)
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Count orders and fills
            cursor.execute("SELECT COUNT(*) FROM orders WHERE session_id = ?", (session_id,))
            total_orders = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM fills WHERE session_id = ?", (session_id,))
            total_fills = cursor.fetchone()[0]
            
            cursor.execute("""
                UPDATE sessions SET
                    end_time = ?,
                    final_equity_position = ?,
                    total_orders = ?,
                    total_fills = ?,
                    realized_pnl = ?,
                    status = ?
                WHERE id = ?
            """, (
                datetime.now().isoformat(),
                final_position,
                total_orders,
                total_fills,
                realized_pnl,
                status,
                session_id,
            ))
        
        logger.info(
            f"Ended session {session_id}: {total_orders} orders, {total_fills} fills, "
            f"P&L=${realized_pnl:.2f}"
        )
    
    def get_session(self, session_id: str) -> Optional[StoredSession]:
        """Get session by ID."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM sessions WHERE id = ?", (session_id,))
            row = cursor.fetchone()
            
            if row:
                return StoredSession(
                    id=row["id"],
                    start_time=row["start_time"],
                    end_time=row["end_time"],
                    mode=row["mode"],
                    initial_equity_position=row["initial_equity_position"],
                    final_equity_position=row["final_equity_position"],
                    total_orders=row["total_orders"],
                    total_fills=row["total_fills"],
                    realized_pnl=row["realized_pnl"],
                    status=row["status"],
                )
        return None
    
    def get_last_session(self) -> Optional[StoredSession]:
        """Get the most recent session."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM sessions ORDER BY start_time DESC LIMIT 1")
            row = cursor.fetchone()
            
            if row:
                return StoredSession(
                    id=row["id"],
                    start_time=row["start_time"],
                    end_time=row["end_time"],
                    mode=row["mode"],
                    initial_equity_position=row["initial_equity_position"],
                    final_equity_position=row["final_equity_position"],
                    total_orders=row["total_orders"],
                    total_fills=row["total_fills"],
                    realized_pnl=row["realized_pnl"],
                    status=row["status"],
                )
        return None
    
    # -------------------------------------------------------------------------
    # Order Management
    # -------------------------------------------------------------------------
    
    def save_order(self, order: StoredOrder) -> None:
        """Save an order to the database."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO orders (
                    id, symbol, side, quantity, order_type, limit_price,
                    status, broker_order_id, submitted_at, filled_at, cancelled_at,
                    filled_quantity, avg_fill_price, error_message, strategy, reason,
                    session_id, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                order.id, order.symbol, order.side, order.quantity, order.order_type,
                order.limit_price, order.status, order.broker_order_id,
                order.submitted_at, order.filled_at, order.cancelled_at,
                order.filled_quantity, order.avg_fill_price, order.error_message,
                order.strategy, order.reason, order.session_id,
                datetime.now().isoformat(),
            ))
        
        logger.debug(f"Saved order {order.id}: {order.status}")
    
    def update_order_status(
        self,
        order_id: str,
        status: str,
        filled_quantity: Optional[int] = None,
        avg_fill_price: Optional[float] = None,
        broker_order_id: Optional[int] = None,
        error_message: Optional[str] = None,
    ) -> None:
        """Update order status and fill details."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            updates = ["status = ?", "updated_at = ?"]
            values = [status, datetime.now().isoformat()]
            
            if filled_quantity is not None:
                updates.append("filled_quantity = ?")
                values.append(filled_quantity)
            
            if avg_fill_price is not None:
                updates.append("avg_fill_price = ?")
                values.append(avg_fill_price)
            
            if broker_order_id is not None:
                updates.append("broker_order_id = ?")
                values.append(broker_order_id)
            
            if error_message is not None:
                updates.append("error_message = ?")
                values.append(error_message)
            
            if status == "FILLED":
                updates.append("filled_at = ?")
                values.append(datetime.now().isoformat())
            elif status == "CANCELLED":
                updates.append("cancelled_at = ?")
                values.append(datetime.now().isoformat())
            
            values.append(order_id)
            
            cursor.execute(
                f"UPDATE orders SET {', '.join(updates)} WHERE id = ?",
                values,
            )
        
        logger.debug(f"Updated order {order_id}: status={status}")
    
    def get_order(self, order_id: str) -> Optional[StoredOrder]:
        """Get order by ID."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM orders WHERE id = ?", (order_id,))
            row = cursor.fetchone()
            
            if row:
                return self._row_to_order(row)
        return None
    
    def get_session_orders(self, session_id: str) -> List[StoredOrder]:
        """Get all orders for a session."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM orders WHERE session_id = ? ORDER BY created_at",
                (session_id,),
            )
            return [self._row_to_order(row) for row in cursor.fetchall()]
    
    def get_open_orders(self, session_id: str) -> List[StoredOrder]:
        """Get open (non-terminal) orders for a session."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM orders WHERE session_id = ? AND status NOT IN ('FILLED', 'CANCELLED', 'REJECTED')",
                (session_id,),
            )
            return [self._row_to_order(row) for row in cursor.fetchall()]
    
    def _row_to_order(self, row: sqlite3.Row) -> StoredOrder:
        """Convert database row to StoredOrder."""
        return StoredOrder(
            id=row["id"],
            symbol=row["symbol"],
            side=row["side"],
            quantity=row["quantity"],
            order_type=row["order_type"],
            limit_price=row["limit_price"],
            status=row["status"],
            broker_order_id=row["broker_order_id"],
            submitted_at=row["submitted_at"],
            filled_at=row["filled_at"],
            cancelled_at=row["cancelled_at"],
            filled_quantity=row["filled_quantity"],
            avg_fill_price=row["avg_fill_price"],
            error_message=row["error_message"],
            strategy=row["strategy"],
            reason=row["reason"],
            session_id=row["session_id"],
        )
    
    # -------------------------------------------------------------------------
    # Fill Management
    # -------------------------------------------------------------------------
    
    def save_fill(self, fill: StoredFill) -> int:
        """
        Save a fill to the database.
        
        Returns:
            Fill ID (auto-generated)
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO fills (order_id, symbol, quantity, price, commission, timestamp, session_id)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                fill.order_id, fill.symbol, fill.quantity, fill.price,
                fill.commission, fill.timestamp, fill.session_id,
            ))
            fill_id = cursor.lastrowid
        
        logger.debug(f"Saved fill {fill_id}: {fill.quantity} @ ${fill.price:.2f}")
        return fill_id
    
    def get_session_fills(self, session_id: str) -> List[StoredFill]:
        """Get all fills for a session."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM fills WHERE session_id = ? ORDER BY timestamp",
                (session_id,),
            )
            return [self._row_to_fill(row) for row in cursor.fetchall()]
    
    def _row_to_fill(self, row: sqlite3.Row) -> StoredFill:
        """Convert database row to StoredFill."""
        return StoredFill(
            id=row["id"],
            order_id=row["order_id"],
            symbol=row["symbol"],
            quantity=row["quantity"],
            price=row["price"],
            commission=row["commission"],
            timestamp=row["timestamp"],
            session_id=row["session_id"],
        )
    
    # -------------------------------------------------------------------------
    # Statistics
    # -------------------------------------------------------------------------
    
    def get_session_stats(self, session_id: str) -> Dict:
        """Get statistics for a session."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Order stats
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_orders,
                    SUM(CASE WHEN status = 'FILLED' THEN 1 ELSE 0 END) as filled_orders,
                    SUM(CASE WHEN side = 'BUY' THEN filled_quantity ELSE 0 END) as total_bought,
                    SUM(CASE WHEN side = 'SELL' THEN filled_quantity ELSE 0 END) as total_sold
                FROM orders WHERE session_id = ?
            """, (session_id,))
            order_row = cursor.fetchone()
            
            # Fill stats
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_fills,
                    SUM(commission) as total_commission,
                    SUM(quantity * price) as total_volume
                FROM fills WHERE session_id = ?
            """, (session_id,))
            fill_row = cursor.fetchone()
            
            return {
                "total_orders": order_row["total_orders"] or 0,
                "filled_orders": order_row["filled_orders"] or 0,
                "total_bought": order_row["total_bought"] or 0,
                "total_sold": order_row["total_sold"] or 0,
                "total_fills": fill_row["total_fills"] or 0,
                "total_commission": fill_row["total_commission"] or 0.0,
                "total_volume": fill_row["total_volume"] or 0.0,
            }
