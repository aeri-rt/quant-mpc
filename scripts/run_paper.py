#!/usr/bin/env python3
"""
Paper Trading Script

Full integration of the MPC-based gamma scalping strategy with IBKR.

This script:
1. Connects to IBKR paper trading account
2. Fetches SPY spot and options data
3. Initializes the GammaScalpingStrategy
4. Runs the hedging loop at configurable intervals
5. Logs all decisions and optionally executes trades

Usage:
    # Dry run (no trades executed)
    python scripts/run_paper.py --dry-run
    
    # Paper trading with execution
    python scripts/run_paper.py --execute
    
    # Custom interval (seconds)
    python scripts/run_paper.py --interval 30

Prerequisites:
    - TWS or IB Gateway running
    - Paper trading account connected
    - API enabled with socket port 7497
"""

import asyncio
import argparse
import signal
from datetime import datetime, timedelta, time
from typing import Optional
import pytz

# Ensure we can import from src
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import IBKRDataProvider, DataProviderConfig, OptionsChainManager
from src.strategy import GammaScalpingStrategy, GammaScalpingConfig, PositionManager, PositionManagerConfig, PositionAction
from src.models.greeks import OptionPosition, EquityPosition, PositionType
from src.core.types import OptionType, OrderType, OrderStatus
from src.core.exceptions import BrokerConnectionError, BrokerOrderError, QuantMPCError
from src.core.state_machine import TradingStateMachine, TradingState, StateEvent
from src.control.mpc_controller import MPCConfig
from src.execution import OrderManager, OptionOrderRequest
from src.risk import RiskManager, RiskConfig
from src.utils.logging import setup_logging, LogConfig, get_logger
from src.utils.config import load_config, get_broker_config, get_strategy_config, get_risk_config, get_mpc_config
from src.utils.metrics import MetricsTracker
from src.data.trade_store import TradeStore

# Setup logging
log_config = LogConfig(
    log_dir=Path(__file__).parent.parent / "logs",
    console_level="INFO",
    file_level="DEBUG",
)
setup_logging(log_config)
logger = get_logger("paper_trading")

# US Market hours (Eastern Time)
ET = pytz.timezone('US/Eastern')
MARKET_OPEN = time(9, 30)   # 9:30 AM ET
MARKET_CLOSE = time(16, 0)  # 4:00 PM ET


def is_market_open() -> bool:
    """Check if US equity market is currently open."""
    now_et = datetime.now(ET)
    
    # Check if weekday (0=Monday, 4=Friday)
    if now_et.weekday() > 4:
        return False
    
    # Check if within trading hours
    current_time = now_et.time()
    return MARKET_OPEN <= current_time <= MARKET_CLOSE


class PaperTradingRunner:
    """
    Manages the paper trading loop.
    
    Connects data provider to strategy and executes hedging decisions.
    """
    
    def __init__(
        self,
        port: int = 7497,
        client_id: int = 10,
        target_dte: int = 30,
        position_size: int = 10,
        dry_run: bool = True,
        interval: float = 60.0,
    ):
        """
        Initialize paper trading runner.
        
        Args:
            port: IBKR socket port (7497 for paper)
            client_id: IBKR client ID
            target_dte: Target DTE for options
            position_size: Number of contracts per leg
            dry_run: If True, don't execute trades
            interval: Seconds between hedging cycles
        """
        self.port = port
        self.client_id = client_id
        self.target_dte = target_dte
        self.position_size = position_size
        self.dry_run = dry_run
        self.interval = interval
        
        self._state_machine = TradingStateMachine()
        self._provider: Optional[IBKRDataProvider] = None
        self._chain_mgr: Optional[OptionsChainManager] = None
        self._strategy: Optional[GammaScalpingStrategy] = None
        self._order_mgr: Optional[OrderManager] = None
        self._risk_mgr: Optional[RiskManager] = None
        self._metrics: Optional[MetricsTracker] = None
        self._position_mgr: Optional[PositionManager] = None
        self._trade_store: Optional[TradeStore] = None
        self._session_id: str = ""
        
        # Current positions (for paper trading simulation)
        self._equity_position: int = 0
        self._option_positions: list = []
        
        # Roll cooldown: suppress repeated roll signals within this window
        self._last_roll_time: Optional[datetime] = None
        self._roll_cooldown_seconds: float = 300.0  # 5 minutes
    
    async def initialize(self) -> None:
        """Initialize all components and connect to IBKR."""
        logger.info("Initializing paper trading runner...")
        
        # Register state machine callback for logging
        self._state_machine.on_state_change(
            lambda old, new, event: logger.info(
                f"State: {old.value} -> {new.value} (event={event.value})"
            )
        )
        
        # Load configuration from YAML files
        app_config = load_config()
        broker_config = app_config.broker
        strategy_config = app_config.strategy
        risk_config_yaml = app_config.risk
        
        logger.info(f"Loaded config: mode={app_config.mode}, symbol={app_config.instrument.symbol}")
        
        # Data provider - use config values or CLI overrides
        config = DataProviderConfig(
            port=self.port if self.port != 7497 else broker_config.port,
            client_id=self.client_id if self.client_id != 10 else broker_config.client_id,
            use_delayed_data=True,  # Set False if you have market data subscription
        )
        self._provider = IBKRDataProvider(config)
        
        # Options chain manager
        self._chain_mgr = OptionsChainManager("SPY")
        
        # Strategy configuration
        yaml_strategy_config = strategy_config  # Preserve YAML config reference
        
        # Load MPC config from YAML and create MPCConfig
        mpc_params = get_mpc_config()
        mpc_config = MPCConfig.from_yaml_config(mpc_params)
        logger.info(
            f"MPC config loaded: horizon={mpc_config.n_horizon}, "
            f"q_max={mpc_config.q_max}, u_max={mpc_config.u_max}, "
            f"w_delta={mpc_config.w_delta}"
        )
        
        gamma_config = GammaScalpingConfig(
            mpc_horizon=mpc_config.n_horizon,
            mpc_q_max=mpc_config.q_max,
            mpc_u_max=mpc_config.u_max,
            min_trade_size=yaml_strategy_config.hedging.min_hedge_size,
        )
        self._strategy = GammaScalpingStrategy(gamma_config, mpc_config=mpc_config)
        
        # Transition: INITIALIZING -> CONNECTING
        self._state_machine.trigger(StateEvent.INIT_COMPLETE, "Config and components loaded")
        
        # Connect to IBKR
        logger.info(f"Connecting to IBKR at port {self.port}...")
        await self._provider.connect()
        logger.info("✓ Connected to IBKR")
        
        # Transition: CONNECTING -> POSITIONING
        self._state_machine.trigger(StateEvent.CONNECTED, "IBKR connected")
        
        # Subscribe to streaming quotes for SPY (more efficient than snapshots)
        await self._provider.subscribe_quotes(["SPY"])
        logger.info("✓ Subscribed to SPY streaming quotes")
        
        # Subscribe to P&L updates for accurate P&L tracking
        try:
            await self._provider.subscribe_pnl()
            logger.info("✓ Subscribed to P&L updates")
        except Exception as e:
            logger.warning(f"Could not subscribe to P&L updates: {e}")
        
        # Initialize trade store for persistence
        self._trade_store = TradeStore(Path(__file__).parent.parent / "data" / "trades.db")
        self._session_id = self._trade_store.start_session(
            mode="paper" if self.dry_run else "paper_execute",
            initial_position=self._equity_position,
        )
        logger.info(f"✓ Trade store initialized (session={self._session_id})")
        
        # Check last session for stale state warning
        last_session = self._trade_store.get_last_session()
        if last_session and last_session.id != self._session_id and last_session.status == "running":
            logger.warning(
                f"Previous session {last_session.id} was not cleanly stopped "
                f"(status={last_session.status}). Positions may be stale."
            )
        
        # Initialize order manager with the IB connection and persistence
        self._order_mgr = OrderManager(
            self._provider._ib,
            trade_store=self._trade_store,
            session_id=self._session_id,
        )
        logger.info("✓ Order manager initialized")
        
        # Initialize risk manager - use YAML config with CLI overrides
        risk_config = RiskConfig(
            max_equity_position=int(risk_config_yaml.max_delta),
            max_single_trade=yaml_strategy_config.hedging.max_delta_exposure,
            max_daily_loss=risk_config_yaml.max_loss_daily,
            max_portfolio_delta=risk_config_yaml.max_delta,
            max_portfolio_gamma=risk_config_yaml.max_gamma,
            enforce_trading_hours=False,  # Allow paper trading anytime
        )
        self._risk_mgr = RiskManager(risk_config)
        
        # Register kill switch callback
        self._risk_mgr.on_kill_switch(self.stop)
        logger.info("✓ Risk manager initialized")
        
        # Initialize metrics tracker
        self._metrics = MetricsTracker()
        logger.info("✓ Metrics tracker initialized")
        
        # Initialize position manager for rolling/exit logic
        position_config = PositionManagerConfig(
            roll_dte=yaml_strategy_config.instrument_selection.roll_dte,
            roll_to_dte=yaml_strategy_config.instrument_selection.roll_to_dte,
            take_profit_pct=yaml_strategy_config.take_profit,
            stop_loss_pct=yaml_strategy_config.stop_loss,
            enable_rolling=True,
            enable_pnl_exits=True,
        )
        self._position_mgr = PositionManager(position_config)
        logger.info("✓ Position manager initialized")
        
        # Fetch initial data
        await self._fetch_market_data()
        
        logger.info("✓ Initialization complete")
    
    async def _fetch_market_data(self) -> None:
        """Fetch current market data."""
        # Get SPY quote
        spy_quote = await self._provider.get_quote("SPY")
        self._chain_mgr.update_spot(spy_quote.mid)
        
        logger.info(f"SPY: ${spy_quote.mid:.2f}")
        
        # Get options chain (±$10 keeps us under IBKR's 100 line limit)
        options = await self._provider.get_options_chain(
            underlying="SPY",
            strike_range=(spy_quote.mid - 10, spy_quote.mid + 10),
        )
        
        if options:
            self._chain_mgr.update_chain(options)
            stats = self._chain_mgr.get_stats()
            logger.info(f"Options: {stats['total_contracts']} contracts loaded")
    
    async def _recover_existing_positions(self) -> bool:
        """
        Check IBKR for existing positions and recover them instead of
        creating a new straddle. Prevents doubled exposure on restart.
        
        Returns:
            True if existing positions were recovered, False if none found
        """
        try:
            ibkr_positions = await self._provider.get_positions()
        except Exception as e:
            logger.warning(f"Could not check IBKR positions: {e}")
            return False
        
        if not ibkr_positions:
            logger.info("No existing IBKR positions found")
            return False
        
        # Separate SPY equity and option positions
        spy_equity = ibkr_positions.get("SPY")
        option_positions = [
            pos for pos in ibkr_positions.values()
            if pos.is_option and pos.symbol == "SPY" and pos.quantity != 0
        ]
        
        if not option_positions:
            # We may have an equity hedge position but no options -- not a recovery scenario
            if spy_equity and spy_equity.quantity != 0:
                logger.info(f"Found SPY equity position ({spy_equity.quantity} shares) but no options")
                self._equity_position = spy_equity.quantity
            return False
        
        logger.info(f"Recovering {len(option_positions)} existing option positions from IBKR")
        
        spot = self._chain_mgr._spot_price if self._chain_mgr._spot_price > 0 else 500.0
        
        # Build OptionPosition objects from IBKR data
        recovered_positions = []
        for ibkr_pos in option_positions:
            pos_type = PositionType.CALL if ibkr_pos.option_type == OptionType.CALL else PositionType.PUT
            
            # IBKR reports position in contracts (quantity is already in contracts)
            qty = ibkr_pos.quantity
            
            opt_pos = OptionPosition(
                symbol=f"SPY_{ibkr_pos.expiry.strftime('%Y%m%d')}_{ibkr_pos.option_type.value}_{ibkr_pos.strike:.0f}",
                position_type=pos_type,
                strike=ibkr_pos.strike,
                expiry=ibkr_pos.expiry,
                quantity=qty,
                multiplier=ibkr_pos.multiplier,
                current_price=ibkr_pos.avg_cost / ibkr_pos.multiplier if ibkr_pos.avg_cost > 0 else 0.0,
                underlying_price=spot,
                iv=0.20,  # Will be refreshed from IBKR Greeks
            )
            recovered_positions.append(opt_pos)
            
            logger.info(
                f"  Recovered: {ibkr_pos.option_type.value} ${ibkr_pos.strike:.0f} "
                f"exp {ibkr_pos.expiry.strftime('%Y-%m-%d')} x{qty} "
                f"@ ${ibkr_pos.avg_cost:.2f}/contract"
            )
            
            # Set entry price in position manager
            if self._position_mgr:
                entry = ibkr_pos.avg_cost / ibkr_pos.multiplier if ibkr_pos.avg_cost > 0 else 0.0
                self._position_mgr.set_entry_price(opt_pos.symbol, entry)
        
        self._option_positions = recovered_positions
        
        # Recover equity position
        if spy_equity:
            self._equity_position = spy_equity.quantity
            logger.info(f"  Recovered SPY equity: {self._equity_position} shares")
        
        # Update strategy
        self._strategy.set_option_positions(recovered_positions)
        self._strategy.set_equity_position("SPY", self._equity_position, spot)
        
        rec = self._strategy.update(spot_price=spot)
        logger.info(
            f"Recovered position: Δ_opt={rec.portfolio_delta:.1f}, "
            f"Γ={rec.portfolio_gamma:.2f}, "
            f"Net Δ={rec.net_delta:.1f}"
        )
        
        return True
    
    async def _setup_initial_position(self) -> None:
        """
        Set up positions for trading.
        
        First checks for existing IBKR positions (recovery on restart).
        If none exist, creates a new ATM straddle.
        """
        # Step 1: Try to recover existing positions
        recovered = await self._recover_existing_positions()
        if recovered:
            logger.info("Using recovered positions -- skipping new straddle creation")
            return
        
        # Step 2: No existing positions -- create new straddle
        # Get ATM straddle quotes
        atm_call, atm_put = self._chain_mgr.get_atm_straddle(
            target_dte=self.target_dte,
        )
        
        if not atm_call or not atm_put:
            logger.error("Could not find ATM straddle")
            return
        
        strike = atm_call.strike
        expiry = atm_call.expiry
        spot = self._chain_mgr._spot_price
        
        logger.info(
            f"Setting up new ATM straddle: Strike ${strike:.0f}, "
            f"Expiry {expiry.strftime('%Y-%m-%d')}"
        )
        
        # Decide whether to place real orders
        should_trade_options = not self.dry_run and is_market_open()
        
        if should_trade_options:
            # Place real option orders via IBKR
            logger.info(f"Placing option orders: {self.position_size} contracts each leg")
            
            try:
                call_order, put_order = await self._order_mgr.submit_straddle_orders(
                    underlying="SPY",
                    expiry=expiry,
                    strike=strike,
                    quantity=self.position_size,
                    use_limit=False,  # Market orders for faster fill
                )
                
                # Wait for fills (30 second timeout)
                orders = [call_order, put_order]
                filled = await self._order_mgr.wait_for_fills(orders, timeout=30.0)
                
                if not filled:
                    logger.error("Option orders did not fill in time - proceeding with simulation")
                    should_trade_options = False
                else:
                    logger.info(
                        f"Straddle filled: "
                        f"Call @ ${call_order.avg_fill_price:.2f}, "
                        f"Put @ ${put_order.avg_fill_price:.2f}"
                    )
                    
            except Exception as e:
                logger.error(f"Failed to place option orders: {e}")
                should_trade_options = False
        else:
            reason = "dry-run mode" if self.dry_run else "market closed"
            logger.info(f"Simulating option positions ({reason})")
        
        # Create internal position objects for Greeks calculation
        positions = self._chain_mgr.to_option_positions([
            (atm_call, self.position_size),
            (atm_put, self.position_size),
        ], spot_price=spot)
        
        self._option_positions = positions
        
        # Record entry prices for P&L tracking
        if self._position_mgr:
            for pos in positions:
                entry_price = pos.current_price if pos.current_price > 0 else (
                    atm_call.mid if pos.is_call else atm_put.mid
                )
                self._position_mgr.set_entry_price(pos.symbol, entry_price)
                logger.debug(f"Entry recorded: {pos.symbol} @ ${entry_price:.2f}")
        
        # Update strategy with positions
        self._strategy.set_option_positions(positions)
        self._strategy.set_equity_position("SPY", self._equity_position, spot)
        
        # Calculate initial Greeks
        rec = self._strategy.update(spot_price=spot)
        
        logger.info(
            f"Initial position: Δ_opt={rec.portfolio_delta:.1f}, "
            f"Γ={rec.portfolio_gamma:.2f}, "
            f"Net Δ={rec.net_delta:.1f}"
        )
    
    async def _refresh_ibkr_greeks(self) -> None:
        """Refresh Greeks from IBKR for all option positions."""
        option_quotes = {}
        
        for pos in self._option_positions:
            try:
                # Parse position details
                from src.core.types import OptionType as OT
                opt_type = OT.CALL if pos.is_call else OT.PUT
                
                quote = await self._provider.get_option_quote(
                    underlying="SPY",
                    expiry=pos.expiry,
                    strike=pos.strike,
                    option_type=opt_type,
                )
                option_quotes[pos.symbol] = quote
                
            except Exception as e:
                logger.debug(f"Failed to get IBKR quote for {pos.symbol}: {e}")
        
        # Update strategy with IBKR Greeks
        if option_quotes:
            updated = self._strategy.update_greeks_from_quotes(option_quotes)
            if updated > 0:
                logger.debug(f"Refreshed {updated} positions with IBKR Greeks")
    
    async def _reconcile_positions(self) -> bool:
        """
        Reconcile internal position state with IBKR.
        
        Compares internal equity position count with actual IBKR positions
        and logs any discrepancies.
        
        Returns:
            True if positions match, False if discrepancy found
        """
        try:
            ibkr_positions = await self._provider.get_positions()
        except Exception as e:
            logger.warning(f"Failed to fetch IBKR positions for reconciliation: {e}")
            return True  # Assume OK if we can't check
        
        # Check SPY equity position
        spy_position = ibkr_positions.get("SPY")
        ibkr_qty = spy_position.quantity if spy_position else 0
        
        if ibkr_qty != self._equity_position:
            discrepancy = ibkr_qty - self._equity_position
            logger.warning(
                f"[RECONCILIATION] SPY position mismatch! "
                f"Internal={self._equity_position}, IBKR={ibkr_qty}, "
                f"Discrepancy={discrepancy:+d} shares"
            )
            
            # Auto-correct internal state to match IBKR
            logger.info(f"Correcting internal position to match IBKR: {ibkr_qty}")
            self._equity_position = ibkr_qty
            self._strategy.set_equity_position("SPY", self._equity_position, 0)
            return False
        
        # Check option positions
        for pos in self._option_positions:
            # Options have complex symbols, check by underlying + strike + expiry
            matching_ibkr = None
            for symbol, ibkr_pos in ibkr_positions.items():
                if (ibkr_pos.is_option and 
                    ibkr_pos.strike == pos.strike and
                    ibkr_pos.expiry and ibkr_pos.expiry.date() == pos.expiry.date()):
                    matching_ibkr = ibkr_pos
                    break
            
            if matching_ibkr:
                if matching_ibkr.quantity != pos.quantity * pos.multiplier / 100:  # Convert to contracts
                    logger.warning(
                        f"[RECONCILIATION] Option position mismatch for {pos.symbol}! "
                        f"Internal={pos.quantity} contracts, IBKR={matching_ibkr.quantity}"
                    )
        
        logger.debug("Position reconciliation passed")
        return True
    
    async def _execute_exit(self, signal) -> None:
        """
        Execute an exit signal -- flatten all positions.
        
        Args:
            signal: PositionSignal with EXIT_PROFIT or EXIT_LOSS action
        """
        reason = signal.reason
        logger.info(f"Executing EXIT: {reason}")
        
        if not self.dry_run and is_market_open():
            # Cancel any open orders first
            if self._order_mgr:
                await self._order_mgr.cancel_all_orders()
            
            # Flatten equity position
            if self._equity_position != 0:
                try:
                    order = await self._order_mgr.submit_hedge_order(
                        symbol="SPY",
                        quantity=-self._equity_position,  # Opposite to flatten
                        use_limit=False,
                    )
                    await asyncio.sleep(1.0)
                    logger.info(f"Flatten order submitted: {-self._equity_position} SPY")
                except Exception as e:
                    logger.error(f"Failed to flatten equity: {e}")
            
            # Close option positions (submit closing orders)
            for pos in self._option_positions:
                try:
                    opt_type = OptionType.CALL if pos.is_call else OptionType.PUT
                    close_request = OptionOrderRequest(
                        underlying="SPY",
                        expiry=pos.expiry,
                        strike=pos.strike,
                        option_type=opt_type,
                        quantity=-pos.quantity,  # Opposite to close
                        order_type=OrderType.MARKET,
                        strategy="gamma_scalping",
                        reason=f"Exit: {reason}",
                    )
                    await self._order_mgr.submit_option_order(close_request)
                    logger.info(f"Close order submitted for {pos.symbol}")
                except Exception as e:
                    logger.error(f"Failed to close {pos.symbol}: {e}")
        else:
            logger.info(f"[DRY RUN / MARKET CLOSED] Would exit all positions")
        
        # Clear internal state
        self._equity_position = 0
        self._option_positions = []
        if self._position_mgr:
            self._position_mgr.clear_position()
        
        # Stop the trading loop
        self.stop()
    
    async def _execute_roll(self, spot: float, signal) -> None:
        """
        Execute a roll signal -- close current straddle and open a new one.
        
        Args:
            spot: Current spot price
            signal: PositionSignal with ROLL action
        """
        target_dte = signal.target_dte if hasattr(signal, 'target_dte') and signal.target_dte else self.target_dte
        logger.info(f"Executing ROLL to DTE={target_dte}: {signal.reason}")
        
        if not self.dry_run and is_market_open():
            # Step 1: Close current option positions
            for pos in self._option_positions:
                try:
                    opt_type = OptionType.CALL if pos.is_call else OptionType.PUT
                    close_request = OptionOrderRequest(
                        underlying="SPY",
                        expiry=pos.expiry,
                        strike=pos.strike,
                        option_type=opt_type,
                        quantity=-pos.quantity,
                        order_type=OrderType.MARKET,
                        strategy="gamma_scalping",
                        reason=f"Roll close: {signal.reason}",
                    )
                    await self._order_mgr.submit_option_order(close_request)
                except Exception as e:
                    logger.error(f"Failed to close {pos.symbol} for roll: {e}")
            
            # Wait for close fills
            await asyncio.sleep(2.0)
            
            # Step 2: Refresh market data and open new straddle
            await self._fetch_market_data()
            
            atm_call, atm_put = self._chain_mgr.get_atm_straddle(target_dte=target_dte)
            if atm_call and atm_put:
                try:
                    call_order, put_order = await self._order_mgr.submit_straddle_orders(
                        underlying="SPY",
                        expiry=atm_call.expiry,
                        strike=atm_call.strike,
                        quantity=self.position_size,
                        use_limit=False,
                    )
                    filled = await self._order_mgr.wait_for_fills([call_order, put_order], timeout=30.0)
                    if filled:
                        logger.info(
                            f"Rolled to new straddle: ${atm_call.strike:.0f} "
                            f"exp {atm_call.expiry.strftime('%Y-%m-%d')}"
                        )
                    else:
                        logger.error("Roll straddle orders did not fill")
                except Exception as e:
                    logger.error(f"Failed to open new straddle for roll: {e}")
            else:
                logger.error("Could not find ATM straddle for roll")
        else:
            logger.info(f"[DRY RUN / MARKET CLOSED] Would roll to DTE={target_dte}")
        
        # Step 3: Rebuild internal positions from market data
        await self._fetch_market_data()
        atm_call, atm_put = self._chain_mgr.get_atm_straddle(target_dte=target_dte)
        
        if atm_call and atm_put:
            positions = self._chain_mgr.to_option_positions([
                (atm_call, self.position_size),
                (atm_put, self.position_size),
            ], spot_price=spot)
            
            self._option_positions = positions
            
            # Update position manager with new entry prices
            if self._position_mgr:
                self._position_mgr.clear_all_positions()
                for pos in positions:
                    entry = pos.current_price if pos.current_price > 0 else (
                        atm_call.mid if pos.is_call else atm_put.mid
                    )
                    self._position_mgr.set_entry_price(pos.symbol, entry)
            
            # Update strategy
            self._strategy.set_option_positions(positions)
            self._strategy.set_equity_position("SPY", self._equity_position, spot)
            
            logger.info(f"Roll complete: new straddle at ${atm_call.strike:.0f}")
        else:
            logger.error("Could not rebuild positions after roll")
    
    async def run(self) -> None:
        """Run the paper trading loop."""
        # Setup initial position
        await self._setup_initial_position()
        
        # Initial position reconciliation
        await self._reconcile_positions()
        
        # Transition: POSITIONING -> RUNNING
        self._state_machine.trigger(StateEvent.POSITIONS_SET, "Positions established")
        
        logger.info(f"Starting paper trading loop (interval={self.interval}s, dry_run={self.dry_run})")
        print()
        print("="*60)
        print("PAPER TRADING ACTIVE")
        print("="*60)
        print("Press Ctrl+C to stop")
        print()
        
        iteration = 0
        while self._state_machine.can_trade:
            try:
                iteration += 1
                
                # Fetch latest data (prefer streaming, fallback to snapshot)
                spy_quote = self._provider.get_streaming_quote("SPY")
                if spy_quote is None or spy_quote.mid <= 0:
                    # Fallback to snapshot if streaming not ready
                    spy_quote = await self._provider.get_quote("SPY")
                
                spot = spy_quote.mid
                self._chain_mgr.update_spot(spot)
                
                # Update option prices and IVs (simplified - just update underlying)
                for pos in self._option_positions:
                    pos.underlying_price = spot
                
                # Refresh IBKR Greeks every 5 ticks (reduces API calls)
                if iteration % 5 == 1 and self._option_positions:
                    await self._refresh_ibkr_greeks()
                
                # Periodic position reconciliation (every 60 ticks / ~1 hour)
                if iteration % 60 == 0:
                    await self._reconcile_positions()
                
                # Check positions for rolling/exit signals (every 10 ticks)
                if iteration % 10 == 1 and self._option_positions and self._position_mgr:
                    position_signals = self._position_mgr.check_positions(self._option_positions)
                    for signal in position_signals:
                        if signal.action == PositionAction.ROLL:
                            # Skip if within roll cooldown
                            if self._last_roll_time:
                                elapsed = (datetime.now() - self._last_roll_time).total_seconds()
                                if elapsed < self._roll_cooldown_seconds:
                                    logger.debug(
                                        f"Roll signal suppressed (cooldown: {elapsed:.0f}s / "
                                        f"{self._roll_cooldown_seconds:.0f}s)"
                                    )
                                    continue
                            logger.warning(
                                f"[ROLL SIGNAL] {signal.position.symbol}: {signal.reason} "
                                f"(target DTE={signal.target_dte})"
                            )
                            self._last_roll_time = datetime.now()
                            await self._execute_roll(spot, signal)
                        elif signal.action in (PositionAction.EXIT_PROFIT, PositionAction.EXIT_LOSS):
                            logger.warning(
                                f"[EXIT SIGNAL] {signal.position.symbol}: {signal.reason} "
                                f"(P&L={signal.pnl_pct:.1%})"
                            )
                            await self._execute_exit(signal)
                            # After exit, stop the loop
                            break
                
                # Get hedge recommendation
                rec = self._strategy.update(spot_price=spot)
                
                # Record Greeks snapshot
                self._metrics.record_greeks(
                    spot_price=spot,
                    delta=rec.portfolio_delta,
                    gamma=rec.portfolio_gamma,
                    equity_position=self._equity_position,
                )
                
                # Log status
                logger.info(
                    f"[{iteration}] S=${spot:.2f} | "
                    f"Δ_opt={rec.portfolio_delta:.0f} | "
                    f"q={rec.equity_position:.0f} | "
                    f"Net Δ={rec.net_delta:.0f} | "
                    f"→ {rec.trade_direction} {abs(rec.trade_quantity):.0f}"
                )
                
                # Execute trade if not dry run
                if rec.trade_quantity != 0:
                    # Validate through risk manager
                    risk_check = self._risk_mgr.validate_trade(
                        symbol="SPY",
                        quantity=rec.trade_quantity,
                        current_position=self._equity_position,
                        current_delta=rec.net_delta,
                        current_gamma=rec.portfolio_gamma,
                    )
                    
                    if not risk_check.passed:
                        logger.warning(
                            f"[RISK BLOCKED] {risk_check.result.value}: {risk_check.message}",
                            original_qty=rec.trade_quantity,
                        )
                        continue
                    
                    # Use adjusted quantity from risk manager
                    trade_qty = risk_check.adjusted_quantity
                    
                    if self.dry_run:
                        logger.info(f"[DRY RUN] Would execute: {rec.trade_direction} {abs(trade_qty):.0f} SPY")
                        # Simulate execution
                        self._equity_position += int(trade_qty)
                        self._strategy.execute_hedge(trade_qty)
                        
                        # Record trade in metrics
                        self._metrics.record_trade(
                            symbol="SPY",
                            side=rec.trade_direction,
                            quantity=abs(int(trade_qty)),
                            price=spot,
                            expected_price=spot,  # No slippage in dry run
                        )
                    elif not is_market_open():
                        # Don't submit orders outside market hours
                        logger.warning(f"[MARKET CLOSED] Skipping order: {rec.trade_direction} {abs(trade_qty):.0f} SPY")
                    else:
                        # Execute via OrderManager
                        try:
                            order = await self._order_mgr.submit_hedge_order(
                                symbol="SPY",
                                quantity=trade_qty,
                                use_limit=False,  # Market orders for simplicity
                            )
                            
                            # Wait briefly for fill (paper trading fills instantly)
                            await asyncio.sleep(0.5)
                            
                            if order.status.value == "FILLED":
                                # Preserve sign from trade_qty (filled_quantity is always positive)
                                signed_fill = trade_qty if abs(trade_qty) == order.filled_quantity else \
                                              (1 if trade_qty > 0 else -1) * order.filled_quantity
                                self._equity_position = int(self._equity_position + signed_fill)
                                self._strategy.execute_hedge(signed_fill)
                                logger.info(
                                    f"✓ Order filled: {order.filled_quantity} @ ${order.avg_fill_price:.2f}"
                                )
                                
                                # Record trade in metrics
                                self._metrics.record_trade(
                                    symbol="SPY",
                                    side=rec.trade_direction,
                                    quantity=abs(order.filled_quantity),
                                    price=order.avg_fill_price,
                                    expected_price=spot,
                                )
                            else:
                                logger.warning(f"Order status: {order.status.value}")
                                # Still update position for paper trading simulation
                                self._equity_position += int(trade_qty)
                                self._strategy.execute_hedge(trade_qty)
                                
                        except Exception as e:
                            logger.error(f"Order execution failed: {e}")
                            # Fallback to simulation
                            self._equity_position += int(trade_qty)
                            self._strategy.execute_hedge(trade_qty)
                
                # Wait for next iteration (check state every second for fast shutdown)
                for _ in range(int(self.interval)):
                    if not self._state_machine.can_trade:
                        break
                    await asyncio.sleep(1.0)
                
            except asyncio.CancelledError:
                break
            except BrokerConnectionError as e:
                logger.error(f"Broker connection lost: {e}")
                self._state_machine.trigger(StateEvent.CONNECTION_LOST, str(e))
                await asyncio.sleep(10)  # Longer pause for connection issues
            except BrokerOrderError as e:
                logger.error(f"Order error: {e}")
                self._state_machine.trigger(StateEvent.RECOVERABLE_ERROR, str(e))
                await asyncio.sleep(5)
            except QuantMPCError as e:
                logger.error(f"System error: {e}")
                self._state_machine.trigger(StateEvent.RECOVERABLE_ERROR, str(e))
                await asyncio.sleep(5)
            except Exception as e:
                logger.error(f"Unexpected error in trading loop: {e}")
                import traceback
                traceback.print_exc()
                self._state_machine.trigger(StateEvent.FATAL_ERROR, str(e))
                await asyncio.sleep(5)
        
        logger.info("Paper trading loop stopped")
    
    def stop(self) -> None:
        """Signal the loop to stop."""
        self._state_machine.trigger(StateEvent.STOP, "Stop requested")
    
    async def shutdown(self) -> None:
        """Clean up resources."""
        logger.info("Shutting down...")
        
        # Ensure we're in STOPPING or STOPPED state
        if self._state_machine.state == TradingState.RUNNING:
            self._state_machine.trigger(StateEvent.STOP, "Shutdown initiated")
        if self._state_machine.state == TradingState.STOPPING:
            self._state_machine.trigger(StateEvent.STOP, "Shutdown complete")
        
        # Get final P&L from IBKR before disconnecting
        ibkr_pnl = None
        if self._provider and self._provider.is_connected():
            try:
                ibkr_pnl = self._provider.get_pnl()
            except Exception as e:
                logger.debug(f"Could not get final P&L: {e}")
        
        # Cancel any open orders
        if self._order_mgr:
            await self._order_mgr.cancel_all_orders()
        
        if self._provider:
            await self._provider.disconnect()
        
        # Print final summary
        print()
        print("="*60)
        print("SESSION SUMMARY")
        print("="*60)
        if self._strategy:
            state = self._strategy.get_state_summary()
            print(f"Final equity position: {state['equity_position']} shares")
            print(f"Final net delta: {state['net_delta']:.1f}")
            print(f"Total ticks processed: {state['tick_count']}")
        
        if ibkr_pnl:
            print(f"IBKR P&L:")
            print(f"  Daily P&L: ${ibkr_pnl.get('daily_pnl', 0):.2f}")
            print(f"  Unrealized: ${ibkr_pnl.get('unrealized_pnl', 0):.2f}")
            print(f"  Realized: ${ibkr_pnl.get('realized_pnl', 0):.2f}")
        
        if self._order_mgr:
            stats = self._order_mgr.get_session_stats()
            print(f"\nOrder Statistics:")
            print(f"  Total orders: {stats['total_orders']}")
            print(f"  Filled: {stats['filled_orders']}")
            print(f"  Shares bought: {stats['total_bought']}")
            print(f"  Shares sold: {stats['total_sold']}")
            print(f"  Total commission: ${stats['total_commission']:.2f}")
        
        if self._risk_mgr:
            risk_status = self._risk_mgr.get_status()
            print(f"\nRisk Status:")
            print(f"  Kill switch: {'ACTIVE' if risk_status['kill_switch_active'] else 'inactive'}")
            print(f"  Daily P&L: ${risk_status['daily_realized_pnl']:.2f}")
            print(f"  Trades last hour: {risk_status['trades_last_hour']}")
        
        # End TradeStore session
        if self._trade_store and self._session_id:
            realized_pnl = ibkr_pnl.get('realized_pnl', 0.0) if ibkr_pnl else 0.0
            session_status = "completed" if self._state_machine.state == TradingState.STOPPED else "error"
            try:
                self._trade_store.end_session(
                    session_id=self._session_id,
                    final_position=self._equity_position,
                    realized_pnl=realized_pnl,
                    status=session_status,
                )
                print(f"\nSession {self._session_id} persisted to TradeStore ({session_status})")
            except Exception as e:
                logger.warning(f"Could not end TradeStore session: {e}")
        
        # Log and save session metrics
        if self._metrics:
            self._metrics.log_session_summary()
            try:
                saved_path = self._metrics.save_to_file()
                print(f"\nSession data saved to: {saved_path}")
            except Exception as e:
                logger.warning(f"Could not save session data: {e}")
        print()


async def main_async(args):
    """Async main function."""
    runner = PaperTradingRunner(
        port=7497 if not args.live else 7496,
        client_id=args.client_id,
        target_dte=args.dte,
        position_size=args.contracts,
        dry_run=args.dry_run,
        interval=args.interval,
    )
    
    # Handle shutdown signals
    loop = asyncio.get_event_loop()
    
    def signal_handler():
        logger.info("Received shutdown signal")
        runner.stop()
    
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, signal_handler)
    
    try:
        await runner.initialize()
        await runner.run()
    finally:
        await runner.shutdown()


def main():
    parser = argparse.ArgumentParser(
        description="Paper trading with MPC gamma scalping strategy"
    )
    parser.add_argument(
        "--dry-run", 
        action="store_true", 
        default=True,
        help="Don't execute trades, just log recommendations (default: True)"
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually execute trades (overrides --dry-run)"
    )
    parser.add_argument(
        "--live", 
        action="store_true", 
        help="Use live trading port (7496) - USE WITH CAUTION"
    )
    parser.add_argument(
        "--interval", 
        type=float, 
        default=60.0,
        help="Seconds between hedging cycles (default: 60)"
    )
    parser.add_argument(
        "--dte", 
        type=int, 
        default=30,
        help="Target days to expiration for options (default: 30)"
    )
    parser.add_argument(
        "--contracts", 
        type=int, 
        default=10,
        help="Number of option contracts per leg (default: 10)"
    )
    parser.add_argument(
        "--client-id", 
        type=int, 
        default=10,
        help="IBKR client ID (default: 10)"
    )
    
    args = parser.parse_args()
    
    # --execute overrides --dry-run
    if args.execute:
        args.dry_run = False
    
    # Safety check for live trading
    if args.live and not args.dry_run:
        confirm = input("⚠️  LIVE TRADING with execution enabled! Type 'CONFIRM' to proceed: ")
        if confirm != "CONFIRM":
            print("Aborted.")
            return
    
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
