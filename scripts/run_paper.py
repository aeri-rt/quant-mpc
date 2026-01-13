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
    python scripts/run_paper.py
    
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
from datetime import datetime, timedelta
from typing import Optional

# Ensure we can import from src
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import IBKRDataProvider, DataProviderConfig, OptionsChainManager
from src.strategy import GammaScalpingStrategy, GammaScalpingConfig
from src.models.greeks import OptionPosition, EquityPosition, PositionType
from src.core.types import OptionType
from src.execution import OrderManager
from src.risk import RiskManager, RiskConfig
from src.utils.logging import setup_logging, LogConfig, get_logger
from src.utils.config import load_config, get_broker_config, get_strategy_config, get_risk_config
from src.utils.metrics import MetricsTracker

# Setup logging
log_config = LogConfig(
    log_dir=Path(__file__).parent.parent / "logs",
    console_level="INFO",
    file_level="DEBUG",
)
setup_logging(log_config)
logger = get_logger("paper_trading")


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
        
        self._running = False
        self._provider: Optional[IBKRDataProvider] = None
        self._chain_mgr: Optional[OptionsChainManager] = None
        self._strategy: Optional[GammaScalpingStrategy] = None
        self._order_mgr: Optional[OrderManager] = None
        self._risk_mgr: Optional[RiskManager] = None
        self._metrics: Optional[MetricsTracker] = None
        
        # Current positions (for paper trading simulation)
        self._equity_position: int = 0
        self._option_positions: list = []
    
    async def initialize(self) -> None:
        """Initialize all components and connect to IBKR."""
        logger.info("Initializing paper trading runner...")
        
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
        
        # Strategy
        strategy_config = GammaScalpingConfig(
            mpc_horizon=20,
            mpc_q_max=500,
            mpc_u_max=100,
            min_trade_size=5,  # Minimum 5 shares to trade
        )
        self._strategy = GammaScalpingStrategy(strategy_config)
        
        # Connect to IBKR
        logger.info(f"Connecting to IBKR at port {self.port}...")
        await self._provider.connect()
        logger.info("✓ Connected to IBKR")
        
        # Initialize order manager with the IB connection
        self._order_mgr = OrderManager(self._provider._ib)
        logger.info("✓ Order manager initialized")
        
        # Initialize risk manager - use YAML config with CLI overrides
        risk_config = RiskConfig(
            max_equity_position=int(risk_config_yaml.max_delta),
            max_single_trade=strategy_config.hedging.max_delta_exposure,
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
        
        # Fetch initial data
        await self._fetch_market_data()
        
        logger.info("✓ Initialization complete")
    
    async def _fetch_market_data(self) -> None:
        """Fetch current market data."""
        # Get SPY quote
        spy_quote = await self._provider.get_quote("SPY")
        self._chain_mgr.update_spot(spy_quote.mid)
        
        logger.info(f"SPY: ${spy_quote.mid:.2f}")
        
        # Get options chain
        options = await self._provider.get_options_chain(
            underlying="SPY",
            strike_range=(spy_quote.mid - 20, spy_quote.mid + 20),
        )
        
        if options:
            self._chain_mgr.update_chain(options)
            stats = self._chain_mgr.get_stats()
            logger.info(f"Options: {stats['total_contracts']} contracts loaded")
    
    async def _setup_initial_position(self) -> None:
        """Set up initial ATM straddle position (simulated for paper trading)."""
        # Get ATM straddle
        atm_call, atm_put = self._chain_mgr.get_atm_straddle(
            target_dte=self.target_dte,
        )
        
        if not atm_call or not atm_put:
            logger.error("Could not find ATM straddle")
            return
        
        logger.info(
            f"Setting up ATM straddle: Strike ${atm_call.strike:.0f}, "
            f"Expiry {atm_call.expiry.strftime('%Y-%m-%d')}"
        )
        
        # Convert to option positions
        spot = self._chain_mgr._spot_price
        
        positions = self._chain_mgr.to_option_positions([
            (atm_call, self.position_size),
            (atm_put, self.position_size),
        ], spot_price=spot)
        
        self._option_positions = positions
        
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
    
    async def run(self) -> None:
        """Run the paper trading loop."""
        self._running = True
        
        # Setup initial position
        await self._setup_initial_position()
        
        logger.info(f"Starting paper trading loop (interval={self.interval}s, dry_run={self.dry_run})")
        print()
        print("="*60)
        print("PAPER TRADING ACTIVE")
        print("="*60)
        print("Press Ctrl+C to stop")
        print()
        
        iteration = 0
        while self._running:
            try:
                iteration += 1
                
                # Fetch latest data
                spy_quote = await self._provider.get_quote("SPY")
                spot = spy_quote.mid
                self._chain_mgr.update_spot(spot)
                
                # Update option prices and IVs (simplified - just update underlying)
                for pos in self._option_positions:
                    pos.underlying_price = spot
                
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
                                self._equity_position = int(self._equity_position + order.filled_quantity)
                                self._strategy.execute_hedge(order.filled_quantity)
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
                
                # Wait for next iteration
                await asyncio.sleep(self.interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                import traceback
                traceback.print_exc()
                await asyncio.sleep(5)  # Brief pause on error
        
        logger.info("Paper trading loop stopped")
    
    def stop(self) -> None:
        """Signal the loop to stop."""
        self._running = False
    
    async def shutdown(self) -> None:
        """Clean up resources."""
        logger.info("Shutting down...")
        
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
