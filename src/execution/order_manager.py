"""
Order Manager for IBKR execution.

Handles order lifecycle management including:
- Order submission (market and limit)
- Order state tracking
- Fill handling and logging
- Order cancellation

Integrates with the IBKR data provider for broker communication.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Callable, Dict, List, Optional
import uuid

from ib_insync import IB, Stock, MarketOrder, LimitOrder, Trade, Order as IBOrder

from src.core.types import OrderSide, OrderType, OrderStatus, Fill
from src.utils.logging import get_logger, log_trade

logger = get_logger("execution")


# =============================================================================
# Order Types
# =============================================================================

@dataclass
class OrderRequest:
    """Request to submit an order."""
    
    symbol: str
    quantity: int              # Positive = buy, negative = sell
    order_type: OrderType = OrderType.MARKET
    limit_price: Optional[float] = None
    
    # Metadata
    strategy: str = "gamma_scalping"
    reason: str = ""
    
    # Internal
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: datetime = field(default_factory=datetime.now)
    
    @property
    def side(self) -> OrderSide:
        """Get order side from quantity."""
        return OrderSide.BUY if self.quantity > 0 else OrderSide.SELL
    
    @property
    def abs_quantity(self) -> int:
        """Absolute quantity."""
        return abs(self.quantity)


@dataclass
class ManagedOrder:
    """An order being managed by the OrderManager."""
    
    request: OrderRequest
    
    # Broker info
    broker_order_id: Optional[int] = None
    ib_trade: Optional[Trade] = None
    
    # State
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: int = 0
    avg_fill_price: float = 0.0
    
    # Timestamps
    submitted_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None
    cancelled_at: Optional[datetime] = None
    
    # Error info
    error_message: Optional[str] = None
    
    @property
    def is_complete(self) -> bool:
        """Check if order is in a terminal state."""
        return self.status in (
            OrderStatus.FILLED,
            OrderStatus.CANCELLED,
            OrderStatus.REJECTED,
        )
    
    @property
    def remaining_quantity(self) -> int:
        """Quantity still to be filled."""
        return self.request.abs_quantity - self.filled_quantity


# =============================================================================
# Order Manager
# =============================================================================

class OrderManager:
    """
    Manages order submission and lifecycle for IBKR.
    
    Usage:
        # Initialize with IB connection
        order_mgr = OrderManager(ib_client)
        
        # Submit an order
        request = OrderRequest(symbol="SPY", quantity=100)
        order = await order_mgr.submit_order(request)
        
        # Check status
        print(f"Status: {order.status}, Filled: {order.filled_quantity}")
        
        # Cancel if needed
        await order_mgr.cancel_order(order.request.id)
    """
    
    def __init__(self, ib: IB):
        """
        Initialize order manager.
        
        Args:
            ib: Connected ib_insync IB instance
        """
        self._ib = ib
        self._orders: Dict[str, ManagedOrder] = {}
        self._fills: List[Fill] = []
        
        # Callbacks
        self._on_fill_callbacks: List[Callable[[ManagedOrder, Fill], None]] = []
        self._on_status_change_callbacks: List[Callable[[ManagedOrder], None]] = []
        
        # Set up IB event handlers
        self._setup_event_handlers()
        
        logger.info("OrderManager initialized")
    
    def _setup_event_handlers(self) -> None:
        """Set up ib_insync order event handlers."""
        self._ib.orderStatusEvent += self._on_order_status
        self._ib.execDetailsEvent += self._on_exec_details
        self._ib.errorEvent += self._on_error
    
    # -------------------------------------------------------------------------
    # Order Submission
    # -------------------------------------------------------------------------
    
    async def submit_order(self, request: OrderRequest) -> ManagedOrder:
        """
        Submit an order to IBKR.
        
        Args:
            request: Order request with symbol, quantity, type
            
        Returns:
            ManagedOrder tracking the order lifecycle
        """
        if not self._ib.isConnected():
            raise ConnectionError("Not connected to IBKR")
        
        logger.info(
            f"Submitting order: {request.side.value} {request.abs_quantity} {request.symbol}",
            order_id=request.id,
            order_type=request.order_type.value,
            limit_price=request.limit_price,
            reason=request.reason,
        )
        
        # Create contract
        contract = Stock(request.symbol, 'SMART', 'USD')
        await self._ib.qualifyContractsAsync(contract)
        
        # Create IB order
        action = "BUY" if request.quantity > 0 else "SELL"
        
        if request.order_type == OrderType.MARKET:
            ib_order = MarketOrder(action, request.abs_quantity)
        elif request.order_type == OrderType.LIMIT:
            if request.limit_price is None:
                raise ValueError("Limit price required for limit orders")
            ib_order = LimitOrder(action, request.abs_quantity, request.limit_price)
        else:
            raise ValueError(f"Unsupported order type: {request.order_type}")
        
        # Create managed order
        managed_order = ManagedOrder(request=request)
        self._orders[request.id] = managed_order
        
        try:
            # Submit to IBKR
            trade = self._ib.placeOrder(contract, ib_order)
            
            managed_order.ib_trade = trade
            managed_order.broker_order_id = trade.order.orderId
            managed_order.status = OrderStatus.SUBMITTED
            managed_order.submitted_at = datetime.now()
            
            logger.info(
                f"Order submitted to IBKR",
                order_id=request.id,
                broker_order_id=managed_order.broker_order_id,
            )
            
            # Log trade submission
            log_trade(
                symbol=request.symbol,
                side=request.side.value,
                quantity=request.abs_quantity,
                order_type=request.order_type.value,
                status="SUBMITTED",
                order_id=request.id,
                broker_order_id=managed_order.broker_order_id,
                reason=request.reason,
            )
            
        except Exception as e:
            managed_order.status = OrderStatus.REJECTED
            managed_order.error_message = str(e)
            logger.error(f"Order submission failed: {e}", order_id=request.id)
            
            log_trade(
                symbol=request.symbol,
                side=request.side.value,
                quantity=request.abs_quantity,
                order_type=request.order_type.value,
                status="REJECTED",
                order_id=request.id,
                error=str(e),
            )
        
        return managed_order
    
    async def submit_hedge_order(
        self,
        symbol: str,
        quantity: float,
        use_limit: bool = False,
        limit_offset: float = 0.01,
    ) -> ManagedOrder:
        """
        Submit a hedge order (convenience method for strategy).
        
        Args:
            symbol: Symbol to trade (e.g., "SPY")
            quantity: Shares to trade (+ buy, - sell)
            use_limit: If True, use limit order at current price + offset
            limit_offset: Offset from mid price for limit orders
            
        Returns:
            ManagedOrder for the submitted order
        """
        # Round quantity to integer
        quantity = int(round(quantity))
        
        if quantity == 0:
            raise ValueError("Cannot submit order with zero quantity")
        
        order_type = OrderType.LIMIT if use_limit else OrderType.MARKET
        limit_price = None
        
        if use_limit:
            # Get current quote for limit price
            contract = Stock(symbol, 'SMART', 'USD')
            await self._ib.qualifyContractsAsync(contract)
            ticker = self._ib.reqMktData(contract, '', snapshot=True)
            await asyncio.sleep(0.5)
            self._ib.cancelMktData(contract)
            
            mid = (ticker.bid + ticker.ask) / 2 if ticker.bid and ticker.ask else ticker.last
            
            # Set limit price with offset (buy above mid, sell below mid)
            if quantity > 0:
                limit_price = round(mid + limit_offset, 2)
            else:
                limit_price = round(mid - limit_offset, 2)
        
        request = OrderRequest(
            symbol=symbol,
            quantity=quantity,
            order_type=order_type,
            limit_price=limit_price,
            strategy="gamma_scalping",
            reason="Delta hedge",
        )
        
        return await self.submit_order(request)
    
    # -------------------------------------------------------------------------
    # Order Management
    # -------------------------------------------------------------------------
    
    async def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an order.
        
        Args:
            order_id: Internal order ID
            
        Returns:
            True if cancellation was submitted
        """
        if order_id not in self._orders:
            logger.warning(f"Order not found for cancellation: {order_id}")
            return False
        
        managed_order = self._orders[order_id]
        
        if managed_order.is_complete:
            logger.warning(f"Cannot cancel completed order: {order_id}")
            return False
        
        if managed_order.ib_trade:
            self._ib.cancelOrder(managed_order.ib_trade.order)
            logger.info(f"Cancellation submitted for order: {order_id}")
            return True
        
        return False
    
    async def cancel_all_orders(self) -> int:
        """
        Cancel all open orders.
        
        Returns:
            Number of orders cancelled
        """
        cancelled = 0
        for order_id, order in self._orders.items():
            if not order.is_complete:
                if await self.cancel_order(order_id):
                    cancelled += 1
        
        logger.info(f"Cancelled {cancelled} orders")
        return cancelled
    
    def get_order(self, order_id: str) -> Optional[ManagedOrder]:
        """Get order by ID."""
        return self._orders.get(order_id)
    
    def get_open_orders(self) -> List[ManagedOrder]:
        """Get all open (non-complete) orders."""
        return [o for o in self._orders.values() if not o.is_complete]
    
    def get_filled_orders(self) -> List[ManagedOrder]:
        """Get all filled orders."""
        return [o for o in self._orders.values() if o.status == OrderStatus.FILLED]
    
    # -------------------------------------------------------------------------
    # Event Handlers
    # -------------------------------------------------------------------------
    
    def _on_order_status(self, trade: Trade) -> None:
        """Handle order status update from IBKR."""
        # Find our managed order
        managed_order = self._find_order_by_trade(trade)
        if not managed_order:
            return
        
        old_status = managed_order.status
        
        # Map IB status to our status
        ib_status = trade.orderStatus.status
        if ib_status == 'Filled':
            managed_order.status = OrderStatus.FILLED
            managed_order.filled_at = datetime.now()
        elif ib_status == 'Cancelled':
            managed_order.status = OrderStatus.CANCELLED
            managed_order.cancelled_at = datetime.now()
        elif ib_status in ('Submitted', 'PreSubmitted'):
            managed_order.status = OrderStatus.SUBMITTED
        elif ib_status == 'PendingCancel':
            pass  # Keep current status
        
        # Update fill info
        managed_order.filled_quantity = int(trade.orderStatus.filled)
        managed_order.avg_fill_price = trade.orderStatus.avgFillPrice or 0.0
        
        if old_status != managed_order.status:
            logger.info(
                f"Order status changed: {old_status.value} -> {managed_order.status.value}",
                order_id=managed_order.request.id,
                filled=managed_order.filled_quantity,
                avg_price=managed_order.avg_fill_price,
            )
            
            # Log status change
            log_trade(
                symbol=managed_order.request.symbol,
                side=managed_order.request.side.value,
                quantity=managed_order.request.abs_quantity,
                order_type=managed_order.request.order_type.value,
                status=managed_order.status.value,
                order_id=managed_order.request.id,
                filled_quantity=managed_order.filled_quantity,
                avg_fill_price=managed_order.avg_fill_price,
            )
            
            # Notify callbacks
            for callback in self._on_status_change_callbacks:
                try:
                    callback(managed_order)
                except Exception as e:
                    logger.error(f"Status callback error: {e}")
    
    def _on_exec_details(self, trade: Trade, fill) -> None:
        """Handle execution details (fill) from IBKR."""
        managed_order = self._find_order_by_trade(trade)
        if not managed_order:
            return
        
        # Create Fill record
        fill_record = Fill(
            order_id=managed_order.request.id,
            instrument=managed_order.request.symbol,
            quantity=int(fill.execution.shares),
            price=fill.execution.price,
            commission=fill.commissionReport.commission if fill.commissionReport else 0.0,
            timestamp=datetime.now(),
        )
        
        self._fills.append(fill_record)
        
        logger.info(
            f"Fill received: {fill_record.quantity} @ ${fill_record.price:.2f}",
            order_id=managed_order.request.id,
            commission=fill_record.commission,
        )
        
        # Log fill
        log_trade(
            symbol=managed_order.request.symbol,
            side=managed_order.request.side.value,
            quantity=fill_record.quantity,
            order_type=managed_order.request.order_type.value,
            status="FILL",
            order_id=managed_order.request.id,
            fill_price=fill_record.price,
            commission=fill_record.commission,
        )
        
        # Notify callbacks
        for callback in self._on_fill_callbacks:
            try:
                callback(managed_order, fill_record)
            except Exception as e:
                logger.error(f"Fill callback error: {e}")
    
    def _on_error(self, reqId: int, errorCode: int, errorString: str, contract) -> None:
        """Handle error from IBKR related to orders."""
        # Find order by reqId if possible
        for order in self._orders.values():
            if order.ib_trade and order.ib_trade.order.orderId == reqId:
                if errorCode in (201, 202, 203):  # Order rejection codes
                    order.status = OrderStatus.REJECTED
                    order.error_message = errorString
                    logger.error(
                        f"Order rejected: {errorString}",
                        order_id=order.request.id,
                        error_code=errorCode,
                    )
    
    def _find_order_by_trade(self, trade: Trade) -> Optional[ManagedOrder]:
        """Find managed order by IB trade."""
        for order in self._orders.values():
            if order.ib_trade and order.ib_trade.order.orderId == trade.order.orderId:
                return order
        return None
    
    # -------------------------------------------------------------------------
    # Callbacks
    # -------------------------------------------------------------------------
    
    def on_fill(self, callback: Callable[[ManagedOrder, Fill], None]) -> None:
        """Register callback for order fills."""
        self._on_fill_callbacks.append(callback)
    
    def on_status_change(self, callback: Callable[[ManagedOrder], None]) -> None:
        """Register callback for order status changes."""
        self._on_status_change_callbacks.append(callback)
    
    # -------------------------------------------------------------------------
    # Statistics
    # -------------------------------------------------------------------------
    
    def get_session_stats(self) -> dict:
        """Get statistics for the current session."""
        filled_orders = self.get_filled_orders()
        
        total_bought = sum(
            o.filled_quantity for o in filled_orders 
            if o.request.quantity > 0
        )
        total_sold = sum(
            o.filled_quantity for o in filled_orders 
            if o.request.quantity < 0
        )
        
        total_commission = sum(f.commission for f in self._fills)
        
        return {
            "total_orders": len(self._orders),
            "filled_orders": len(filled_orders),
            "open_orders": len(self.get_open_orders()),
            "total_bought": total_bought,
            "total_sold": total_sold,
            "net_position_change": total_bought - total_sold,
            "total_fills": len(self._fills),
            "total_commission": total_commission,
        }
