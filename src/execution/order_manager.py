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
from typing import Callable, Dict, List, Optional, Tuple
import uuid

from ib_insync import IB, Stock, Option, MarketOrder, LimitOrder, Trade, Order as IBOrder

from src.core.types import OrderSide, OrderType, OrderStatus, Fill, OptionType
from src.core.exceptions import BrokerConnectionError, BrokerOrderError
from src.utils.logging import get_logger, log_trade
from src.utils.helpers import safe_float

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
class OptionOrderRequest:
    """Request to submit an option order."""
    
    underlying: str            # e.g., "SPY"
    expiry: datetime           # Expiration date
    strike: float              # Strike price
    option_type: OptionType    # CALL or PUT
    quantity: int              # Positive = buy, negative = sell (in contracts)
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
        """Absolute quantity (contracts)."""
        return abs(self.quantity)
    
    @property
    def right(self) -> str:
        """IBKR option right (C or P)."""
        return 'C' if self.option_type == OptionType.CALL else 'P'
    
    @property
    def symbol(self) -> str:
        """Generate option symbol."""
        exp_str = self.expiry.strftime('%Y%m%d')
        return f"{self.underlying}{exp_str}{self.right}{int(self.strike*1000):08d}"


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
    
    def __init__(self, ib: IB, trade_store=None, session_id: str = ""):
        """
        Initialize order manager.
        
        Args:
            ib: Connected ib_insync IB instance
            trade_store: Optional TradeStore for persistence
            session_id: Current session ID for TradeStore
        """
        self._ib = ib
        self._orders: Dict[str, ManagedOrder] = {}
        self._fills: List[Fill] = []
        
        # Trade persistence (optional)
        self._trade_store = trade_store
        self._session_id = session_id
        
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
            raise BrokerConnectionError("Not connected to IBKR")
        
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
                raise BrokerOrderError("Limit price required for limit orders")
            ib_order = LimitOrder(action, request.abs_quantity, request.limit_price)
        else:
            raise BrokerOrderError(f"Unsupported order type: {request.order_type}")
        
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
            
            # Persist to TradeStore
            if self._trade_store and self._session_id:
                try:
                    from src.data.trade_store import StoredOrder
                    stored_order = StoredOrder(
                        id=request.id,
                        symbol=request.symbol,
                        side=request.side.value,
                        quantity=request.abs_quantity,
                        order_type=request.order_type.value,
                        limit_price=request.limit_price,
                        status="SUBMITTED",
                        broker_order_id=managed_order.broker_order_id,
                        submitted_at=datetime.now().isoformat(),
                        filled_at=None,
                        cancelled_at=None,
                        filled_quantity=0,
                        avg_fill_price=0.0,
                        error_message=None,
                        strategy=request.strategy,
                        reason=request.reason,
                        session_id=self._session_id,
                    )
                    self._trade_store.save_order(stored_order)
                except Exception as e:
                    logger.debug(f"Failed to persist order: {e}")
            
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
            raise BrokerOrderError("Cannot submit order with zero quantity")
        
        order_type = OrderType.LIMIT if use_limit else OrderType.MARKET
        limit_price = None
        
        if use_limit:
            # Get current quote for limit price
            contract = Stock(symbol, 'SMART', 'USD')
            await self._ib.qualifyContractsAsync(contract)
            ticker = self._ib.reqMktData(contract, '', snapshot=True)
            await asyncio.sleep(0.5)
            self._ib.cancelMktData(contract)
            
            # Safely compute mid price, guarding against NaN
            bid = safe_float(ticker.bid)
            ask = safe_float(ticker.ask)
            last = safe_float(ticker.last)
            
            if bid > 0 and ask > 0:
                mid = (bid + ask) / 2
            elif last > 0:
                mid = last
            else:
                # No valid price -- fall back to market order
                logger.warning(f"No valid quote for {symbol} limit order, falling back to market")
                order_type = OrderType.MARKET
                mid = 0.0
            
            # Set limit price with offset (buy above mid, sell below mid)
            if order_type == OrderType.LIMIT and mid > 0:
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
    
    async def submit_option_order(
        self,
        request: OptionOrderRequest,
    ) -> ManagedOrder:
        """
        Submit an option order to IBKR.
        
        Args:
            request: OptionOrderRequest with underlying, expiry, strike, type, quantity
            
        Returns:
            ManagedOrder tracking the order lifecycle
        """
        if not self._ib.isConnected():
            raise BrokerConnectionError("Not connected to IBKR")
        
        logger.info(
            f"Submitting option order: {request.side.value} {request.abs_quantity} "
            f"{request.underlying} {request.expiry.strftime('%Y-%m-%d')} "
            f"${request.strike:.0f} {request.option_type.value}",
            order_id=request.id,
            order_type=request.order_type.value,
            limit_price=request.limit_price,
            reason=request.reason,
        )
        
        # Create option contract
        contract = Option(
            symbol=request.underlying,
            lastTradeDateOrContractMonth=request.expiry.strftime('%Y%m%d'),
            strike=request.strike,
            right=request.right,
            exchange='SMART',
            currency='USD',
            tradingClass=request.underlying,
        )
        
        # Qualify the contract
        qualified = await self._ib.qualifyContractsAsync(contract)
        if not qualified:
            raise BrokerOrderError(f"Failed to qualify option contract: {request.symbol}")
        
        # Create IB order
        action = "BUY" if request.quantity > 0 else "SELL"
        
        if request.order_type == OrderType.MARKET:
            ib_order = MarketOrder(action, request.abs_quantity)
        elif request.order_type == OrderType.LIMIT:
            if request.limit_price is None:
                raise BrokerOrderError("Limit price required for limit orders")
            ib_order = LimitOrder(action, request.abs_quantity, request.limit_price)
        else:
            raise BrokerOrderError(f"Unsupported order type: {request.order_type}")
        
        # Create managed order using equity OrderRequest as base
        equity_request = OrderRequest(
            symbol=request.symbol,
            quantity=request.quantity,
            order_type=request.order_type,
            limit_price=request.limit_price,
            strategy=request.strategy,
            reason=request.reason,
            id=request.id,
        )
        
        managed_order = ManagedOrder(request=equity_request)
        self._orders[request.id] = managed_order
        
        try:
            # Submit to IBKR
            trade = self._ib.placeOrder(contract, ib_order)
            
            managed_order.ib_trade = trade
            managed_order.broker_order_id = trade.order.orderId
            managed_order.status = OrderStatus.SUBMITTED
            managed_order.submitted_at = datetime.now()
            
            logger.info(
                f"Option order submitted to IBKR",
                order_id=request.id,
                broker_order_id=managed_order.broker_order_id,
                contract=str(contract),
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
            logger.error(f"Option order submission failed: {e}", order_id=request.id)
            
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
    
    async def submit_straddle_orders(
        self,
        underlying: str,
        expiry: datetime,
        strike: float,
        quantity: int,
        use_limit: bool = False,
        limit_offset: float = 0.05,
    ) -> Tuple[ManagedOrder, ManagedOrder]:
        """
        Submit orders for an ATM straddle (call + put at same strike).
        
        Args:
            underlying: Symbol (e.g., "SPY")
            expiry: Expiration date
            strike: Strike price
            quantity: Number of contracts per leg (positive = buy)
            use_limit: If True, use limit orders
            limit_offset: Offset from mid for limit orders
            
        Returns:
            Tuple of (call_order, put_order)
        """
        # Get current option quotes for limit pricing
        limit_call = None
        limit_put = None
        
        if use_limit:
            # Request option quotes
            call_contract = Option(
                underlying, expiry.strftime('%Y%m%d'), strike, 'C', 'SMART', 'USD', underlying
            )
            put_contract = Option(
                underlying, expiry.strftime('%Y%m%d'), strike, 'P', 'SMART', 'USD', underlying
            )
            
            await self._ib.qualifyContractsAsync(call_contract, put_contract)
            
            call_ticker = self._ib.reqMktData(call_contract, '', snapshot=True)
            put_ticker = self._ib.reqMktData(put_contract, '', snapshot=True)
            await asyncio.sleep(0.5)
            
            self._ib.cancelMktData(call_contract)
            self._ib.cancelMktData(put_contract)
            
            # Set limit prices with NaN safety
            call_bid = safe_float(call_ticker.bid)
            call_ask = safe_float(call_ticker.ask)
            if call_bid > 0 and call_ask > 0:
                call_mid = (call_bid + call_ask) / 2
                limit_call = round(call_mid + limit_offset if quantity > 0 else call_mid - limit_offset, 2)
            else:
                logger.warning("No valid call quote for limit pricing, using market order")
            
            put_bid = safe_float(put_ticker.bid)
            put_ask = safe_float(put_ticker.ask)
            if put_bid > 0 and put_ask > 0:
                put_mid = (put_bid + put_ask) / 2
                limit_put = round(put_mid + limit_offset if quantity > 0 else put_mid - limit_offset, 2)
            else:
                logger.warning("No valid put quote for limit pricing, using market order")
        
        # Create order requests
        call_request = OptionOrderRequest(
            underlying=underlying,
            expiry=expiry,
            strike=strike,
            option_type=OptionType.CALL,
            quantity=quantity,
            order_type=OrderType.LIMIT if use_limit and limit_call else OrderType.MARKET,
            limit_price=limit_call,
            strategy="gamma_scalping",
            reason="Straddle entry - Call leg",
        )
        
        put_request = OptionOrderRequest(
            underlying=underlying,
            expiry=expiry,
            strike=strike,
            option_type=OptionType.PUT,
            quantity=quantity,
            order_type=OrderType.LIMIT if use_limit and limit_put else OrderType.MARKET,
            limit_price=limit_put,
            strategy="gamma_scalping",
            reason="Straddle entry - Put leg",
        )
        
        # Submit both orders
        call_order = await self.submit_option_order(call_request)
        put_order = await self.submit_option_order(put_request)
        
        logger.info(
            f"Straddle orders submitted: {underlying} {expiry.strftime('%Y-%m-%d')} ${strike:.0f}",
            call_order_id=call_request.id,
            put_order_id=put_request.id,
            quantity=quantity,
        )
        
        return call_order, put_order
    
    async def wait_for_fills(
        self,
        orders: List[ManagedOrder],
        timeout: float = 30.0,
        poll_interval: float = 0.5,
    ) -> bool:
        """
        Wait for multiple orders to fill.
        
        Args:
            orders: List of orders to wait for
            timeout: Maximum wait time in seconds
            poll_interval: Time between status checks
            
        Returns:
            True if all orders filled, False if timeout
        """
        start = datetime.now()
        
        while (datetime.now() - start).total_seconds() < timeout:
            # Check if all filled
            all_filled = all(o.status == OrderStatus.FILLED for o in orders)
            if all_filled:
                logger.info(f"All {len(orders)} orders filled")
                return True
            
            # Check for any rejections
            any_rejected = any(o.status == OrderStatus.REJECTED for o in orders)
            if any_rejected:
                rejected = [o for o in orders if o.status == OrderStatus.REJECTED]
                logger.warning(f"{len(rejected)} orders rejected")
                return False
            
            await asyncio.sleep(poll_interval)
        
        # Timeout - log status of unfilled orders
        unfilled = [o for o in orders if o.status != OrderStatus.FILLED]
        logger.warning(
            f"Timeout waiting for fills. {len(unfilled)}/{len(orders)} unfilled",
            unfilled_ids=[o.request.id for o in unfilled],
        )
        return False
    
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
            
            # Persist to TradeStore if available
            if self._trade_store and self._session_id:
                try:
                    self._trade_store.update_order_status(
                        order_id=managed_order.request.id,
                        status=managed_order.status.value,
                        filled_quantity=managed_order.filled_quantity,
                        avg_fill_price=managed_order.avg_fill_price,
                        broker_order_id=managed_order.broker_order_id,
                    )
                except Exception as e:
                    logger.debug(f"Failed to persist order status: {e}")
            
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
        
        # Persist fill to TradeStore if available
        if self._trade_store and self._session_id:
            try:
                from src.data.trade_store import StoredFill
                stored_fill = StoredFill(
                    id=0,  # Auto-generated
                    order_id=fill_record.order_id,
                    symbol=fill_record.instrument,
                    quantity=fill_record.quantity,
                    price=fill_record.price,
                    commission=fill_record.commission,
                    timestamp=fill_record.timestamp.isoformat(),
                    session_id=self._session_id,
                )
                self._trade_store.save_fill(stored_fill)
            except Exception as e:
                logger.debug(f"Failed to persist fill: {e}")
        
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
