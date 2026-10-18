"""
Interactive Brokers data provider implementation.

Uses ib_insync for async communication with TWS/Gateway.
Provides real-time and historical data for SPY and options.
"""

import asyncio
import math
from datetime import datetime, timedelta
from typing import Callable, Dict, List, Optional, Set
from collections import defaultdict
from enum import Enum

from ib_insync import IB, Stock, Option, Contract, Ticker, BarData, util

from src.core.types import Quote, OptionQuote, Position, OptionType
from src.data.provider import DataProvider, DataProviderConfig
from src.utils.logging import get_logger
from src.utils.helpers import safe_int, safe_float, safe_price
from src.core.exceptions import BrokerConnectionError, DataValidationError
from src.utils.validation import validate_quote, validate_option_quote, validate_spot_price, ValidationError

logger = get_logger("ibkr")


class ConnectionState(str, Enum):
    """IBKR connection state."""
    DISCONNECTED = "DISCONNECTED"
    CONNECTING = "CONNECTING"
    CONNECTED = "CONNECTED"
    RECONNECTING = "RECONNECTING"


class IBKRDataProvider(DataProvider):
    """
    IBKR data provider using ib_insync.
    
    Features:
    - Async connection management with auto-reconnect
    - Real-time streaming quotes for stocks and options
    - Options chain retrieval with Greeks
    - Position and account info sync
    - Historical data access
    
    Usage:
        provider = IBKRDataProvider()
        await provider.connect()
        
        # Get SPY quote
        quote = await provider.get_quote("SPY")
        
        # Subscribe to real-time updates
        provider.on_quote(lambda sym, q: print(f"{sym}: {q.mid}"))
        await provider.subscribe_quotes(["SPY"])
        
        # Get options chain
        chain = await provider.get_options_chain("SPY", expiry=datetime(2026, 2, 21))
    """
    
    def __init__(self, config: Optional[DataProviderConfig] = None):
        """Initialize IBKR provider."""
        super().__init__(config)
        
        self._ib = IB()
        self._connected = False
        self._subscribed_symbols: Set[str] = set()
        self._tickers: Dict[str, Ticker] = {}
        self._positions_cache: Dict[str, Position] = {}
        self._reconnect_task: Optional[asyncio.Task] = None
        
        # Option contract cache to avoid repeated lookups
        self._option_contracts: Dict[str, Contract] = {}
        
        # === Subscription tracking (C2/C3 fix) ===
        # Track all active subscriptions by reqId for proper cleanup
        self._active_mkt_data: Dict[int, Contract] = {}  # reqId -> Contract
        self._active_account_summary: Optional[int] = None  # Single account summary reqId
        self._active_pnl: Optional[int] = None  # Single P&L reqId
        self._snapshot_req_ids: Set[int] = set()  # Track snapshot requests for cleanup
        
        # === Connection state and heartbeat (M3) ===
        self._connection_state = ConnectionState.DISCONNECTED
        self._reconnect_attempts = 0
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._last_heartbeat: datetime = datetime.now()
        self._on_state_change_callbacks: List[Callable[[ConnectionState], None]] = []
        
        # Set up IB event handlers
        self._setup_event_handlers()
    
    def _setup_event_handlers(self) -> None:
        """Set up ib_insync event handlers."""
        self._ib.connectedEvent += self._on_connected
        self._ib.disconnectedEvent += self._on_disconnected
        self._ib.pendingTickersEvent += self._on_pending_tickers
        self._ib.positionEvent += self._on_position
        self._ib.errorEvent += self._on_error
    
    # -------------------------------------------------------------------------
    # Connection Management
    # -------------------------------------------------------------------------
    
    async def connect(self) -> None:
        """Connect to TWS/Gateway."""
        if self._connected:
            logger.warning("Already connected")
            return
        
        self._set_connection_state(ConnectionState.CONNECTING)
        
        try:
            logger.info(
                f"Connecting to IBKR at {self.config.host}:{self.config.port} "
                f"(client_id={self.config.client_id})"
            )
            
            await asyncio.wait_for(
                self._ib.connectAsync(
                    host=self.config.host,
                    port=self.config.port,
                    clientId=self.config.client_id,
                    timeout=self.config.connect_timeout,
                ),
                timeout=self.config.connect_timeout + 5,
            )
            
            self._connected = True
            self._set_connection_state(ConnectionState.CONNECTED)
            logger.info("Connected to IBKR")
            
            # Start heartbeat task
            if self._heartbeat_task is None or self._heartbeat_task.done():
                self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
            
            # Request delayed data if configured (for paper without subscription)
            if self.config.use_delayed_data:
                self._ib.reqMarketDataType(3)  # Delayed data
                logger.info("Using delayed market data")
            
        except asyncio.TimeoutError:
            self._set_connection_state(ConnectionState.DISCONNECTED)
            raise BrokerConnectionError(
                f"Connection timed out after {self.config.connect_timeout}s"
            )
        except Exception as e:
            self._set_connection_state(ConnectionState.DISCONNECTED)
            raise BrokerConnectionError(f"Failed to connect: {e}")
    
    async def disconnect(self) -> None:
        """Disconnect from IBKR."""
        # Cancel background tasks
        if self._reconnect_task:
            self._reconnect_task.cancel()
            self._reconnect_task = None
        
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            self._heartbeat_task = None
        
        if self._ib.isConnected():
            # Cancel all active subscriptions before disconnecting
            await self._cancel_all_subscriptions()
            
            self._ib.disconnect()
            self._connected = False
            self._set_connection_state(ConnectionState.DISCONNECTED)
            logger.info("Disconnected from IBKR")
    
    async def _cancel_all_subscriptions(self) -> None:
        """Cancel all active subscriptions (call before disconnect)."""
        cancelled = 0
        
        # Cancel market data subscriptions
        for req_id, contract in list(self._active_mkt_data.items()):
            try:
                self._ib.cancelMktData(contract)
                cancelled += 1
            except Exception as e:
                logger.debug(f"Error cancelling market data {req_id}: {e}")
        self._active_mkt_data.clear()
        
        # Cancel account summary if active
        if self._active_account_summary is not None:
            try:
                self._ib.cancelAccountSummary(self._active_account_summary)
                cancelled += 1
            except Exception as e:
                logger.debug(f"Error cancelling account summary: {e}")
            self._active_account_summary = None
        
        # Cancel P&L subscription if active
        if self._active_pnl is not None:
            try:
                self._ib.cancelPnL(self._active_pnl)
                cancelled += 1
            except Exception as e:
                logger.debug(f"Error cancelling P&L: {e}")
            self._active_pnl = None
        
        # Clear tracking sets
        self._subscribed_symbols.clear()
        self._tickers.clear()
        self._snapshot_req_ids.clear()
        
        if cancelled > 0:
            logger.debug(f"Cancelled {cancelled} subscriptions")
    
    def is_connected(self) -> bool:
        """Check connection status."""
        return self._ib.isConnected()
    
    async def _reconnect_loop(self) -> None:
        """
        Background task to reconnect on disconnect.
        
        Uses exponential backoff to avoid hammering the server.
        """
        self._set_connection_state(ConnectionState.RECONNECTING)
        self._reconnect_attempts = 0
        base_delay = self.config.reconnect_delay
        max_delay = 60.0  # Cap at 60 seconds
        
        while self._reconnect_attempts < self.config.max_reconnect_attempts:
            # Exponential backoff: delay = base * 2^attempts (capped at max_delay)
            delay = min(base_delay * (2 ** self._reconnect_attempts), max_delay)
            
            logger.info(
                f"Reconnect attempt {self._reconnect_attempts + 1}/"
                f"{self.config.max_reconnect_attempts} in {delay:.1f}s..."
            )
            
            await asyncio.sleep(delay)
            
            try:
                self._set_connection_state(ConnectionState.CONNECTING)
                await self.connect()
                
                # Re-subscribe to previous subscriptions
                if self._subscribed_symbols:
                    await self.subscribe_quotes(list(self._subscribed_symbols))
                
                self._reconnect_attempts = 0  # Reset on success
                logger.info("Reconnection successful!")
                return
                
            except Exception as e:
                logger.warning(f"Reconnect attempt {self._reconnect_attempts + 1} failed: {e}")
                self._reconnect_attempts += 1
                self._set_connection_state(ConnectionState.RECONNECTING)
        
        logger.error("Max reconnect attempts reached - giving up")
        self._set_connection_state(ConnectionState.DISCONNECTED)
    
    async def _heartbeat_loop(self) -> None:
        """
        Periodic heartbeat to verify connection is alive.
        
        Runs every 30 seconds and checks if the connection is still valid.
        If not, triggers reconnection.
        """
        heartbeat_interval = 30.0
        
        while self._connected:
            await asyncio.sleep(heartbeat_interval)
            
            if not self._ib.isConnected():
                logger.warning("Heartbeat detected connection loss")
                self._on_disconnected()
                break
            
            self._last_heartbeat = datetime.now()
            logger.debug("Heartbeat OK")
    
    def _set_connection_state(self, state: ConnectionState) -> None:
        """Update connection state and notify callbacks."""
        if state != self._connection_state:
            old_state = self._connection_state
            self._connection_state = state
            logger.info(f"Connection state: {old_state.value} -> {state.value}")
            
            # Notify callbacks
            for callback in self._on_state_change_callbacks:
                try:
                    callback(state)
                except Exception as e:
                    logger.debug(f"State change callback error: {e}")
    
    def on_connection_state_change(self, callback: Callable[[ConnectionState], None]) -> None:
        """Register callback for connection state changes."""
        self._on_state_change_callbacks.append(callback)
    
    @property
    def connection_state(self) -> ConnectionState:
        """Get current connection state."""
        return self._connection_state
    
    # -------------------------------------------------------------------------
    # Event Handlers
    # -------------------------------------------------------------------------
    
    def _on_connected(self) -> None:
        """Handle connection event."""
        self._connected = True
        self._set_connection_state(ConnectionState.CONNECTED)
        logger.info("IB connected event")
    
    def _on_disconnected(self) -> None:
        """Handle disconnection event."""
        self._connected = False
        
        # Cancel heartbeat task
        if self._heartbeat_task and not self._heartbeat_task.done():
            self._heartbeat_task.cancel()
            self._heartbeat_task = None
        
        logger.warning("IB disconnected")
        
        if self.config.auto_reconnect and not self._reconnect_task:
            self._reconnect_task = asyncio.create_task(self._reconnect_loop())
        else:
            self._set_connection_state(ConnectionState.DISCONNECTED)
    
    def _on_pending_tickers(self, tickers: Set[Ticker]) -> None:
        """Handle incoming ticker updates."""
        for ticker in tickers:
            try:
                symbol = ticker.contract.symbol
                close_price = safe_price(ticker.close, default=0.0)
                
                quote = Quote(
                    bid=safe_price(ticker.bid, fallback=close_price),
                    ask=safe_price(ticker.ask, fallback=close_price),
                    bid_size=safe_int(ticker.bidSize),
                    ask_size=safe_int(ticker.askSize),
                    last=safe_price(ticker.last, fallback=close_price),
                    last_size=safe_int(ticker.lastSize),
                    timestamp=datetime.now(),
                )
                
                # Validate before emitting
                try:
                    quote = validate_quote(quote, symbol)
                except ValidationError:
                    logger.debug(f"Skipping invalid quote for {symbol}")
                    continue
                
                # Emit to callbacks
                self._emit_quote(symbol, quote)
            except Exception as e:
                # Log but don't raise - prevents eventkit from printing stack traces
                logger.debug(f"Error processing ticker for {ticker.contract.symbol}: {e}")
    
    def _on_position(self, position) -> None:
        """Handle position update."""
        # Convert IB position to our Position type
        pos = Position(
            symbol=position.contract.symbol,
            quantity=int(position.position),
            avg_cost=position.avgCost,
            market_value=0.0,  # Will be updated with market data
            unrealized_pnl=0.0,
            realized_pnl=0.0,
            is_option=isinstance(position.contract, Option),
        )
        
        if pos.is_option:
            pos.option_type = OptionType.CALL if position.contract.right == 'C' else OptionType.PUT
            pos.strike = position.contract.strike
            pos.expiry = datetime.strptime(position.contract.lastTradeDateOrContractMonth, '%Y%m%d')
            pos.multiplier = int(position.contract.multiplier or 100)
        
        self._positions_cache[position.contract.symbol] = pos
        self._emit_positions(self._positions_cache.copy())
    
    def _on_error(self, reqId: int, errorCode: int, errorString: str, contract) -> None:
        """Handle error event."""
        # Non-critical / informational errors (log at debug level)
        non_critical = {
            300,   # Can't find EId - normal for snapshot cleanup
            322,   # Account summary request limit - ignore, we don't use it
            2103,  # Market data farm connection broken (non-US farms)
            2104,  # Market data farm connected
            2105,  # HMDS data farm connection broken
            2106,  # HMDS data farm connected
            2107,  # HMDS data farm inactive
            2108,  # Market data farm inactive but available on demand
            2119,  # Market data farm connected
            2157,  # Sec-def data farm connection broken
            2158,  # Sec-def data farm connected
        }
        
        # Connection status messages (logged at INFO level)
        connection_info = {1100, 1102}  # Connectivity lost/restored
        
        if errorCode in non_critical:
            logger.debug(f"IBKR info [{errorCode}]: {errorString}")
        elif errorCode in connection_info:
            logger.info(f"IBKR connection [{errorCode}]: {errorString}")
        else:
            logger.warning(f"IBKR error [{errorCode}]: {errorString}")
    
    # -------------------------------------------------------------------------
    # Spot Data
    # -------------------------------------------------------------------------
    
    async def get_quote(self, symbol: str) -> Quote:
        """Get current quote for a stock."""
        if not self.is_connected():
            raise BrokerConnectionError("Not connected to IBKR")
        
        contract = Stock(symbol, 'SMART', 'USD')
        
        # Qualify the contract
        await self._ib.qualifyContractsAsync(contract)
        
        # Request market data snapshot
        # Note: Snapshots auto-cancel after data is received, so we don't need to cancel
        ticker = self._ib.reqMktData(contract, '', snapshot=True)
        
        # Wait for data (with timeout)
        for _ in range(50):  # 5 second timeout
            await asyncio.sleep(0.1)
            if ticker.bid > 0 or ticker.last > 0:
                break
        
        # Don't cancel - snapshots auto-cancel and explicit cancel causes Error 300
        # The snapshot will be cleaned up automatically by IBKR
        
        close_price = safe_price(ticker.close, default=0.0)
        
        quote = Quote(
            bid=safe_price(ticker.bid, fallback=close_price),
            ask=safe_price(ticker.ask, fallback=close_price),
            bid_size=safe_int(ticker.bidSize),
            ask_size=safe_int(ticker.askSize),
            last=safe_price(ticker.last, fallback=close_price),
            last_size=safe_int(ticker.lastSize),
            timestamp=datetime.now(),
        )
        
        return validate_quote(quote, symbol)
    
    async def subscribe_quotes(self, symbols: List[str]) -> None:
        """
        Subscribe to real-time quote updates.
        
        Streaming subscriptions are more efficient than repeated snapshots:
        - Single request for continuous updates
        - Lower latency (no request/response cycle)
        - Better for high-frequency hedging
        
        Use `get_streaming_quote()` to retrieve the latest data.
        """
        if not self.is_connected():
            raise BrokerConnectionError("Not connected to IBKR")
        
        for symbol in symbols:
            if symbol in self._subscribed_symbols:
                continue
            
            contract = Stock(symbol, 'SMART', 'USD')
            await self._ib.qualifyContractsAsync(contract)
            
            ticker = self._ib.reqMktData(contract, '', snapshot=False)
            self._tickers[symbol] = ticker
            self._subscribed_symbols.add(symbol)
            
            # Track for proper cleanup
            if hasattr(ticker, 'reqId'):
                self._active_mkt_data[ticker.reqId] = contract
            
            logger.info(f"Subscribed to quotes for {symbol}")
    
    def get_streaming_quote(self, symbol: str) -> Optional[Quote]:
        """
        Get the latest quote from a streaming subscription.
        
        This is more efficient than get_quote() for frequently accessed symbols
        as it doesn't make a new API request.
        
        Args:
            symbol: Symbol that has been subscribed via subscribe_quotes()
            
        Returns:
            Latest Quote, or None if not subscribed or no data yet
        """
        ticker = self._tickers.get(symbol)
        if not ticker:
            return None
        
        close_price = safe_price(ticker.close, default=0.0)
        
        quote = Quote(
            bid=safe_price(ticker.bid, fallback=close_price),
            ask=safe_price(ticker.ask, fallback=close_price),
            bid_size=safe_int(ticker.bidSize),
            ask_size=safe_int(ticker.askSize),
            last=safe_price(ticker.last, fallback=close_price),
            last_size=safe_int(ticker.lastSize),
            timestamp=datetime.now(),
        )
        
        try:
            return validate_quote(quote, symbol)
        except ValidationError:
            logger.debug(f"Streaming quote for {symbol} has no valid prices")
            return None
    
    async def unsubscribe_quotes(self, symbols: List[str]) -> None:
        """Unsubscribe from quote updates."""
        for symbol in symbols:
            if symbol in self._tickers:
                ticker = self._tickers[symbol]
                self._ib.cancelMktData(ticker.contract)
                
                # Clean up tracking
                if hasattr(ticker, 'reqId') and ticker.reqId in self._active_mkt_data:
                    del self._active_mkt_data[ticker.reqId]
                
                del self._tickers[symbol]
                self._subscribed_symbols.discard(symbol)
                logger.info(f"Unsubscribed from quotes for {symbol}")
    
    # -------------------------------------------------------------------------
    # Options Data
    # -------------------------------------------------------------------------
    
    async def get_options_chain(
        self,
        underlying: str,
        expiry: Optional[datetime] = None,
        strike_range: Optional[tuple[float, float]] = None,
    ) -> List[OptionQuote]:
        """Get options chain for an underlying."""
        if not self.is_connected():
            raise BrokerConnectionError("Not connected to IBKR")
        
        # Get the underlying contract
        stock = Stock(underlying, 'SMART', 'USD')
        await self._ib.qualifyContractsAsync(stock)
        
        # Get available option chains
        chains = await self._ib.reqSecDefOptParamsAsync(
            stock.symbol,
            '',  # futFopExchange
            stock.secType,
            stock.conId,
        )
        
        if not chains:
            logger.warning(f"No option chains found for {underlying}")
            return []
        
        # Get current price to help select the right chain
        quote = await self.get_quote(underlying)
        current_price = quote.mid
        
        # Find chain with strikes actually near current price
        # We need strikes within ±$50 of current price to be useful
        def has_strikes_near_price(chain_strikes, price, tolerance=50):
            """Check if chain has at least 5 strikes within tolerance of price."""
            if not chain_strikes:
                return False
            nearby = [s for s in chain_strikes if abs(s - price) <= tolerance]
            return len(nearby) >= 5
        
        chain = None
        
        # Prefer real exchanges over SMART for option chain data
        # CBOE is the primary exchange for SPY options
        preferred_exchanges = ['CBOE', 'AMEX', 'PHLX', 'ISE', 'BOX', 'MIAX', 'BATS', 'SMART']
        
        for exchange in preferred_exchanges:
            for c in chains:
                if c.exchange == exchange and has_strikes_near_price(c.strikes, current_price):
                    chain = c
                    logger.info(f"Selected chain from exchange {c.exchange} with {len(c.strikes)} strikes")
                    break
            if chain:
                break
        
        # Last fallback: any chain with strikes near price
        if not chain:
            for c in chains:
                if has_strikes_near_price(c.strikes, current_price):
                    chain = c
                    logger.info(f"Using chain from exchange {c.exchange}")
                    break
        
        # Absolutely last resort: use chain with most strikes
        if not chain:
            chain = max(chains, key=lambda c: len(c.strikes) if c.strikes else 0)
            logger.warning(f"No chain with nearby strikes found, using {chain.exchange} with {len(chain.strikes) if chain.strikes else 0} strikes")
        
        # Filter expirations
        if expiry:
            target_expiry = expiry.strftime('%Y%m%d')
            expirations = [target_expiry] if target_expiry in chain.expirations else []
        else:
            # Get nearest 3 expirations
            expirations = sorted(chain.expirations)[:3]
        
        if not expirations:
            logger.warning(f"No matching expirations for {underlying}")
            return []
        
        # Filter strikes
        strikes = sorted(chain.strikes)
        if strike_range:
            strikes = [s for s in strikes if strike_range[0] <= s <= strike_range[1]]
        
        # Limit to reasonable ATM range if no range specified
        if not strike_range:
            # Use the current_price we already fetched
            atm = round(current_price)
            # Limit to ±5 strikes from ATM for speed (11 strikes total)
            strikes = sorted([s for s in strikes if abs(s - atm) <= 5])
            logger.debug(f"Using {len(strikes)} strikes near ATM {atm}: {strikes[:5]}...")
        
        # Limit to first 2 expirations for speed
        expirations = expirations[:2]
        logger.debug(f"Using {len(expirations)} expirations: {expirations}")
        
        total_contracts = len(expirations) * len(strikes) * 2
        logger.info(f"Requesting data for {total_contracts} option contracts...")
        
        # Build option contracts in batches to stay under IBKR's
        # simultaneous market data line limit (100 for paper accounts)
        BATCH_SIZE = 40  # Conservative: leaves room for SPY + other subs
        option_quotes = []
        qualify_failures = 0
        data_failures = 0
        contracts_processed = 0
        
        # Build flat list of (expiry, strike, right) tuples
        all_requests = [
            (exp, strike, right)
            for exp in expirations
            for strike in strikes
            for right in ['C', 'P']
        ]
        
        for batch_start in range(0, len(all_requests), BATCH_SIZE):
            batch = all_requests[batch_start:batch_start + BATCH_SIZE]
            batch_tickers = []  # Track tickers for explicit cleanup
            
            for exp, strike, right in batch:
                contracts_processed += 1
                try:
                    contract = Option(
                        underlying,
                        exp,
                        strike,
                        right,
                        'SMART',
                        tradingClass=underlying,
                    )
                    
                    # Qualify and request data
                    qualified = await self._ib.qualifyContractsAsync(contract)
                    if not qualified:
                        qualify_failures += 1
                        continue
                    
                    # Don't use generic ticks with snapshot - causes error 321 with delayed data
                    ticker = self._ib.reqMktData(contract, '', snapshot=True)
                    batch_tickers.append((ticker, contract, exp, strike, right))
                    
                except Exception as e:
                    data_failures += 1
                    logger.debug(f"Error requesting {underlying} {exp} {strike}{right}: {e}")
            
            # Wait for batch data (allow up to 0.3s per batch)
            for _ in range(6):
                await asyncio.sleep(0.05)
            
            # Process batch results
            for ticker, contract, exp, strike, right in batch_tickers:
                try:
                    # Check if we got valid price data
                    has_price = (ticker.bid and ticker.bid > 0) or \
                                (ticker.ask and ticker.ask > 0) or \
                                (ticker.last and ticker.last > 0)
                    
                    if not has_price:
                        data_failures += 1
                        continue
                    
                    # Build OptionQuote
                    exp_date = datetime.strptime(exp, '%Y%m%d')
                    
                    opt_quote = OptionQuote(
                        symbol=f"{underlying}{exp}{right}{int(strike*1000):08d}",
                        underlying=underlying,
                        option_type=OptionType.CALL if right == 'C' else OptionType.PUT,
                        strike=strike,
                        expiry=exp_date,
                        bid=safe_price(ticker.bid, default=0.0),
                        ask=safe_price(ticker.ask, default=0.0),
                        last=safe_price(ticker.last, default=0.0),
                        volume=safe_int(ticker.volume),
                        open_interest=0,  # Not in snapshot
                        timestamp=datetime.now(),
                        delta=safe_float(ticker.modelGreeks.delta) if ticker.modelGreeks else 0.0,
                        gamma=safe_float(ticker.modelGreeks.gamma) if ticker.modelGreeks else 0.0,
                        vega=safe_float(ticker.modelGreeks.vega) if ticker.modelGreeks else 0.0,
                        theta=safe_float(ticker.modelGreeks.theta) if ticker.modelGreeks else 0.0,
                        iv=safe_float(ticker.modelGreeks.impliedVol) if ticker.modelGreeks else 0.0,
                    )
                    
                    # Validate Greeks and prices
                    opt_quote = validate_option_quote(opt_quote)
                    option_quotes.append(opt_quote)
                    
                except Exception as e:
                    data_failures += 1
                    logger.debug(f"Failed to get option {underlying} {exp} {strike} {right}: {e}")
            
            # Snapshots auto-cancel after data is received, but we need to
            # wait between batches to let them clear from IBKR's line count
            if batch_start + BATCH_SIZE < len(all_requests):
                await asyncio.sleep(0.5)
        
        logger.info(
            f"Retrieved {len(option_quotes)} option quotes for {underlying} "
            f"(qualify_fail={qualify_failures}, data_fail={data_failures}, processed={contracts_processed})"
        )
        return option_quotes
    
    async def get_option_quote(
        self,
        underlying: str,
        expiry: datetime,
        strike: float,
        option_type: OptionType,
    ) -> OptionQuote:
        """Get quote for a specific option contract."""
        if not self.is_connected():
            raise BrokerConnectionError("Not connected to IBKR")
        
        right = 'C' if option_type == OptionType.CALL else 'P'
        exp_str = expiry.strftime('%Y%m%d')
        
        contract = Option(
            underlying,
            exp_str,
            strike,
            right,
            'SMART',
            tradingClass=underlying,
        )
        
        await self._ib.qualifyContractsAsync(contract)
        
        # Don't use generic ticks with snapshot - causes error 321 with delayed data
        ticker = self._ib.reqMktData(contract, '', snapshot=True)
        
        # Wait for data
        for _ in range(30):
            await asyncio.sleep(0.1)
            if ticker.modelGreeks:
                break
        
        # Don't cancel - snapshots auto-cancel after data received
        
        opt_quote = OptionQuote(
            symbol=f"{underlying}{exp_str}{right}{int(strike*1000):08d}",
            underlying=underlying,
            option_type=option_type,
            strike=strike,
            expiry=expiry,
            bid=safe_price(ticker.bid, default=0.0),
            ask=safe_price(ticker.ask, default=0.0),
            last=safe_price(ticker.last, default=0.0),
            volume=safe_int(ticker.volume),
            open_interest=0,
            timestamp=datetime.now(),
            delta=safe_float(ticker.modelGreeks.delta) if ticker.modelGreeks else 0.0,
            gamma=safe_float(ticker.modelGreeks.gamma) if ticker.modelGreeks else 0.0,
            vega=safe_float(ticker.modelGreeks.vega) if ticker.modelGreeks else 0.0,
            theta=safe_float(ticker.modelGreeks.theta) if ticker.modelGreeks else 0.0,
            iv=safe_float(ticker.modelGreeks.impliedVol) if ticker.modelGreeks else 0.0,
        )
        
        return validate_option_quote(opt_quote)
    
    async def subscribe_options(
        self,
        underlying: str,
        expiry: Optional[datetime] = None,
    ) -> None:
        """Subscribe to real-time options chain updates."""
        # For now, just log that this is called
        # Full implementation would maintain streaming subscriptions
        logger.info(f"Options subscription requested for {underlying} (expiry={expiry})")
        # In a full implementation, we'd maintain persistent subscriptions
        # and emit updates via _emit_option_chain
    
    # -------------------------------------------------------------------------
    # Position Data
    # -------------------------------------------------------------------------
    
    async def get_positions(self) -> Dict[str, Position]:
        """Get current positions."""
        if not self.is_connected():
            raise BrokerConnectionError("Not connected to IBKR")
        
        # Request positions (will trigger _on_position callbacks)
        ib_positions = self._ib.positions()
        
        positions = {}
        for pos in ib_positions:
            symbol = pos.contract.symbol
            
            position = Position(
                symbol=symbol,
                quantity=int(pos.position),
                avg_cost=pos.avgCost,
                market_value=0.0,
                unrealized_pnl=0.0,
                realized_pnl=0.0,
                is_option=isinstance(pos.contract, Option),
            )
            
            if position.is_option:
                position.option_type = OptionType.CALL if pos.contract.right == 'C' else OptionType.PUT
                position.strike = pos.contract.strike
                position.expiry = datetime.strptime(
                    pos.contract.lastTradeDateOrContractMonth, '%Y%m%d'
                )
                position.multiplier = int(pos.contract.multiplier or 100)
            
            positions[symbol] = position
        
        self._positions_cache = positions
        return positions
    
    async def get_account_info(self) -> dict:
        """Get account information."""
        if not self.is_connected():
            raise BrokerConnectionError("Not connected to IBKR")
        
        account_values = self._ib.accountValues()
        
        # Extract key values
        info = {}
        for av in account_values:
            if av.tag in ['TotalCashValue', 'BuyingPower', 'NetLiquidation', 
                          'GrossPositionValue', 'MaintMarginReq', 
                          'RealizedPnL', 'UnrealizedPnL', 'DailyPnL']:
                info[av.tag] = float(av.value)
        
        return info
    
    # -------------------------------------------------------------------------
    # P&L Tracking
    # -------------------------------------------------------------------------
    
    async def subscribe_pnl(self, account: str = '') -> None:
        """
        Subscribe to account P&L updates.
        
        IBKR provides real-time P&L via reqPnL which is more accurate
        than calculating it ourselves.
        
        Args:
            account: Account ID (empty string = use first managed account)
        """
        if not self.is_connected():
            raise BrokerConnectionError("Not connected to IBKR")
        
        # Cancel existing subscription if any
        if self._active_pnl is not None:
            try:
                self._ib.cancelPnL(self._active_pnl)
            except Exception:
                pass
        
        # Get account ID if not specified
        if not account:
            accounts = self._ib.managedAccounts()
            if accounts:
                account = accounts[0]
                logger.debug(f"Using first managed account: {account}")
            else:
                logger.warning("No managed accounts found, P&L subscription may fail")
        
        # Request P&L subscription
        pnl = self._ib.reqPnL(account)
        self._active_pnl = pnl
        logger.info(f"Subscribed to P&L updates (account={account})")
    
    def get_pnl(self) -> dict:
        """
        Get current P&L values from subscription.
        
        Returns:
            Dict with daily_pnl, unrealized_pnl, realized_pnl
        """
        if self._active_pnl is None:
            return {
                'daily_pnl': 0.0,
                'unrealized_pnl': 0.0,
                'realized_pnl': 0.0,
            }
        
        import math
        pnl = self._active_pnl

        def _safe_pnl(val, default=0.0):
            if val is None or not isinstance(val, (int, float)) or math.isnan(val) or math.isinf(val):
                return default
            return float(val)

        return {
            'daily_pnl': _safe_pnl(getattr(pnl, 'dailyPnL', None)),
            'unrealized_pnl': _safe_pnl(getattr(pnl, 'unrealizedPnL', None)),
            'realized_pnl': _safe_pnl(getattr(pnl, 'realizedPnL', None)),
        }
    
    async def cancel_pnl(self) -> None:
        """Cancel P&L subscription."""
        if self._active_pnl is not None:
            try:
                self._ib.cancelPnL(self._active_pnl)
                logger.info("Cancelled P&L subscription")
            except Exception as e:
                logger.debug(f"Error cancelling P&L subscription: {e}")
            finally:
                self._active_pnl = None
    
    # -------------------------------------------------------------------------
    # Historical Data
    # -------------------------------------------------------------------------
    
    async def get_historical_bars(
        self,
        symbol: str,
        duration: str,
        bar_size: str,
        end_time: Optional[datetime] = None,
    ) -> List[dict]:
        """Get historical price bars."""
        if not self.is_connected():
            raise BrokerConnectionError("Not connected to IBKR")
        
        contract = Stock(symbol, 'SMART', 'USD')
        await self._ib.qualifyContractsAsync(contract)
        
        end_dt = end_time or datetime.now()
        
        bars = await self._ib.reqHistoricalDataAsync(
            contract,
            endDateTime=end_dt,
            durationStr=duration,
            barSizeSetting=bar_size,
            whatToShow='TRADES',
            useRTH=True,  # Regular trading hours
            formatDate=1,
        )
        
        return [
            {
                'time': bar.date,
                'open': bar.open,
                'high': bar.high,
                'low': bar.low,
                'close': bar.close,
                'volume': bar.volume,
            }
            for bar in bars
        ]

