"""
Risk Manager for the Quant-MPC trading system.

Provides pre-trade validation and risk controls including:
- Position limits (max equity position)
- Trade size limits (max single trade)
- Loss limits (daily P&L threshold)
- Greeks limits (max delta/gamma exposure)
- Kill switch (manual emergency stop)

All recommendations pass through risk checks before execution.
"""

import math
from dataclasses import dataclass, field
from datetime import datetime, date
from enum import Enum
from typing import Optional, List, Dict, Callable
import threading

from src.utils.logging import get_logger, log_metrics

logger = get_logger("risk")


# =============================================================================
# Risk Check Results
# =============================================================================

class RiskCheckResult(Enum):
    """Result of a risk check."""
    PASS = "PASS"
    FAIL_POSITION_LIMIT = "FAIL_POSITION_LIMIT"
    FAIL_TRADE_SIZE = "FAIL_TRADE_SIZE"
    FAIL_LOSS_LIMIT = "FAIL_LOSS_LIMIT"
    FAIL_DELTA_LIMIT = "FAIL_DELTA_LIMIT"
    FAIL_GAMMA_LIMIT = "FAIL_GAMMA_LIMIT"
    FAIL_KILL_SWITCH = "FAIL_KILL_SWITCH"
    FAIL_NOT_TRADING_HOURS = "FAIL_NOT_TRADING_HOURS"


@dataclass
class RiskCheckResponse:
    """Response from risk validation."""
    
    passed: bool
    result: RiskCheckResult
    message: str
    
    # Original request (if modified)
    original_quantity: float = 0.0
    adjusted_quantity: float = 0.0
    
    # Timestamp
    timestamp: datetime = field(default_factory=datetime.now)
    
    def __repr__(self) -> str:
        return f"RiskCheck({self.result.value}: {self.message})"


# =============================================================================
# Risk Configuration
# =============================================================================

@dataclass
class RiskConfig:
    """Configuration for risk limits."""
    
    # Position limits
    max_equity_position: int = 500         # Max shares (long or short)
    max_single_trade: int = 100            # Max shares per trade
    
    # Loss limits
    max_daily_loss: float = 1000.0         # Max daily realized loss ($)
    max_unrealized_loss: float = 2000.0    # Max unrealized loss ($)
    
    # Greeks limits
    max_portfolio_delta: float = 500.0     # Max absolute delta
    max_portfolio_gamma: float = 100.0     # Max absolute gamma
    
    # Trading hours (Eastern Time)
    trading_start_hour: int = 9
    trading_start_minute: int = 30
    trading_end_hour: int = 16
    trading_end_minute: int = 0
    enforce_trading_hours: bool = False    # Set True for live trading
    
    # Circuit breakers
    max_trades_per_minute: int = 10
    max_trades_per_hour: int = 100
    
    # Allow position reduction even when limits exceeded
    allow_risk_reducing_trades: bool = True


# =============================================================================
# Risk Manager
# =============================================================================

class RiskManager:
    """
    Risk manager for pre-trade validation.
    
    Usage:
        risk_mgr = RiskManager(RiskConfig())
        
        # Validate a trade
        response = risk_mgr.validate_trade(
            symbol="SPY",
            quantity=100,
            current_position=200,
            current_delta=150.0,
        )
        
        if response.passed:
            execute_trade(response.adjusted_quantity)
        else:
            logger.warning(f"Trade rejected: {response.message}")
    """
    
    def __init__(self, config: Optional[RiskConfig] = None):
        """Initialize risk manager."""
        self.config = config or RiskConfig()
        
        # State tracking
        self._kill_switch_active = False
        self._daily_realized_pnl = 0.0
        self._unrealized_pnl = 0.0
        self._last_reset_date: Optional[date] = None
        
        # Trade rate tracking
        self._recent_trades: List[datetime] = []
        
        # Lock for thread safety
        self._lock = threading.Lock()
        
        # Callbacks for kill switch events
        self._kill_switch_callbacks: List[Callable[[], None]] = []
        
        logger.info(
            "RiskManager initialized",
            max_position=self.config.max_equity_position,
            max_trade=self.config.max_single_trade,
            max_daily_loss=self.config.max_daily_loss,
        )
    
    # -------------------------------------------------------------------------
    # Trade Validation
    # -------------------------------------------------------------------------
    
    def validate_trade(
        self,
        symbol: str,
        quantity: float,
        current_position: int = 0,
        current_delta: float = 0.0,
        current_gamma: float = 0.0,
    ) -> RiskCheckResponse:
        """
        Validate a proposed trade against risk limits.
        
        Args:
            symbol: Symbol to trade
            quantity: Proposed quantity (+ buy, - sell)
            current_position: Current equity position
            current_delta: Current portfolio delta
            current_gamma: Current portfolio gamma
            
        Returns:
            RiskCheckResponse indicating pass/fail and any adjustments
        """
        with self._lock:
            # NaN/Inf guard on all numeric inputs
            def _sanitize(val: float, name: str) -> float:
                if math.isnan(val) or math.isinf(val):
                    logger.warning(f"Risk input {name} is NaN/Inf, treating as 0.0")
                    return 0.0
                return val
            
            quantity = _sanitize(float(quantity), "quantity")
            current_delta = _sanitize(float(current_delta), "current_delta")
            current_gamma = _sanitize(float(current_gamma), "current_gamma")
            
            # Reset daily stats if new day
            self._maybe_reset_daily_stats()
            
            # Check kill switch first
            if self._kill_switch_active:
                return RiskCheckResponse(
                    passed=False,
                    result=RiskCheckResult.FAIL_KILL_SWITCH,
                    message="Kill switch is active - all trading halted",
                    original_quantity=quantity,
                    adjusted_quantity=0.0,
                )
            
            # Check trading hours
            if self.config.enforce_trading_hours and not self._is_trading_hours():
                return RiskCheckResponse(
                    passed=False,
                    result=RiskCheckResult.FAIL_NOT_TRADING_HOURS,
                    message="Outside trading hours",
                    original_quantity=quantity,
                    adjusted_quantity=0.0,
                )
            
            # Check loss limits
            if self._daily_realized_pnl <= -self.config.max_daily_loss:
                return RiskCheckResponse(
                    passed=False,
                    result=RiskCheckResult.FAIL_LOSS_LIMIT,
                    message=f"Daily loss limit reached: ${-self._daily_realized_pnl:.2f}",
                    original_quantity=quantity,
                    adjusted_quantity=0.0,
                )
            
            # Check if trade is risk-reducing
            is_risk_reducing = self._is_risk_reducing(quantity, current_position, current_delta)
            
            # Adjust quantity for position limit
            adjusted_quantity = self._adjust_for_position_limit(
                quantity, current_position, is_risk_reducing
            )
            
            if adjusted_quantity == 0 and quantity != 0:
                return RiskCheckResponse(
                    passed=False,
                    result=RiskCheckResult.FAIL_POSITION_LIMIT,
                    message=f"Position limit {self.config.max_equity_position} would be exceeded",
                    original_quantity=quantity,
                    adjusted_quantity=0.0,
                )
            
            # Adjust for trade size limit
            adjusted_quantity = self._adjust_for_trade_size(adjusted_quantity)
            
            # Check delta limits (only for risk-increasing trades)
            if not is_risk_reducing:
                new_delta = current_delta + adjusted_quantity
                if abs(new_delta) > self.config.max_portfolio_delta:
                    return RiskCheckResponse(
                        passed=False,
                        result=RiskCheckResult.FAIL_DELTA_LIMIT,
                        message=f"Delta limit {self.config.max_portfolio_delta} would be exceeded",
                        original_quantity=quantity,
                        adjusted_quantity=0.0,
                    )
            
            # Check gamma limits (only for risk-increasing trades)
            if not is_risk_reducing and self.config.max_portfolio_gamma > 0:
                if abs(current_gamma) > self.config.max_portfolio_gamma:
                    return RiskCheckResponse(
                        passed=False,
                        result=RiskCheckResult.FAIL_GAMMA_LIMIT,
                        message=f"Gamma limit {self.config.max_portfolio_gamma} exceeded (current={current_gamma:.2f})",
                        original_quantity=quantity,
                        adjusted_quantity=0.0,
                    )
            
            # Check trade rate limits
            if not self._check_trade_rate():
                return RiskCheckResponse(
                    passed=False,
                    result=RiskCheckResult.FAIL_TRADE_SIZE,
                    message="Trade rate limit exceeded",
                    original_quantity=quantity,
                    adjusted_quantity=0.0,
                )
            
            # Record this trade for rate limiting
            self._recent_trades.append(datetime.now())
            
            # Log the validation
            logger.debug(
                f"Trade validated: {symbol} qty={adjusted_quantity} (orig={quantity})",
                symbol=symbol,
                original_quantity=quantity,
                adjusted_quantity=adjusted_quantity,
                current_position=current_position,
                is_risk_reducing=is_risk_reducing,
            )
            
            return RiskCheckResponse(
                passed=True,
                result=RiskCheckResult.PASS,
                message="Trade approved",
                original_quantity=quantity,
                adjusted_quantity=adjusted_quantity,
            )
    
    def _is_risk_reducing(
        self,
        quantity: float,
        current_position: int,
        current_delta: float,
    ) -> bool:
        """
        Check if a trade reduces risk.
        
        A trade is risk-reducing if it:
        - Reduces absolute position size
        - Reduces absolute delta exposure
        """
        new_position = current_position + quantity
        new_delta = current_delta + quantity
        
        # Position-reducing
        if abs(new_position) < abs(current_position):
            return True
        
        # Delta-reducing
        if abs(new_delta) < abs(current_delta):
            return True
        
        return False
    
    def _adjust_for_position_limit(
        self,
        quantity: float,
        current_position: int,
        is_risk_reducing: bool,
    ) -> float:
        """Adjust quantity to stay within position limits."""
        max_pos = self.config.max_equity_position
        
        # Always allow risk-reducing trades
        if is_risk_reducing and self.config.allow_risk_reducing_trades:
            return quantity
        
        new_position = current_position + quantity
        
        # Check if new position exceeds limit
        if abs(new_position) > max_pos:
            # Calculate max allowed quantity
            if quantity > 0:
                # Buying - cap at max - current
                max_buy = max_pos - current_position
                return max(0, min(quantity, max_buy))
            else:
                # Selling - cap at -max - current
                max_sell = -max_pos - current_position
                return min(0, max(quantity, max_sell))
        
        return quantity
    
    def _adjust_for_trade_size(self, quantity: float) -> float:
        """Adjust quantity to stay within single trade limit."""
        max_trade = self.config.max_single_trade
        
        if abs(quantity) > max_trade:
            # Reduce to max allowed, maintaining sign
            return max_trade if quantity > 0 else -max_trade
        
        return quantity
    
    def _check_trade_rate(self) -> bool:
        """Check if trade rate limits are exceeded."""
        now = datetime.now()
        
        # Count recent trades
        trades_last_minute = sum(1 for t in self._recent_trades if (now - t).total_seconds() < 60)
        trades_last_hour = sum(1 for t in self._recent_trades if (now - t).total_seconds() < 3600)
        
        if trades_last_minute >= self.config.max_trades_per_minute:
            logger.warning(f"Trade rate limit: {trades_last_minute} trades in last minute")
            return False
        
        if trades_last_hour >= self.config.max_trades_per_hour:
            logger.warning(f"Trade rate limit: {trades_last_hour} trades in last hour")
            return False
        
        # Clean up old entries
        self._recent_trades = [t for t in self._recent_trades if (now - t).total_seconds() < 3600]
        
        return True
    
    def _is_trading_hours(self) -> bool:
        """Check if current time is within trading hours."""
        now = datetime.now()
        
        trading_start = now.replace(
            hour=self.config.trading_start_hour,
            minute=self.config.trading_start_minute,
            second=0,
            microsecond=0,
        )
        trading_end = now.replace(
            hour=self.config.trading_end_hour,
            minute=self.config.trading_end_minute,
            second=0,
            microsecond=0,
        )
        
        # Check if weekday (0=Monday, 6=Sunday)
        if now.weekday() >= 5:
            return False
        
        return trading_start <= now <= trading_end
    
    def _maybe_reset_daily_stats(self) -> None:
        """Reset daily stats if it's a new trading day."""
        today = date.today()
        
        if self._last_reset_date != today:
            logger.info(
                "Resetting daily risk stats",
                previous_date=self._last_reset_date,
                new_date=today,
                previous_daily_pnl=self._daily_realized_pnl,
            )
            self._daily_realized_pnl = 0.0
            self._last_reset_date = today
    
    # -------------------------------------------------------------------------
    # P&L Tracking
    # -------------------------------------------------------------------------
    
    def record_pnl(self, realized: float = 0.0, unrealized: float = 0.0) -> None:
        """
        Record P&L for risk tracking.
        
        Args:
            realized: Realized P&L from this trade
            unrealized: Current unrealized P&L
        """
        with self._lock:
            self._maybe_reset_daily_stats()
            
            self._daily_realized_pnl += realized
            self._unrealized_pnl = unrealized
            
            # Check if loss limits triggered
            if self._daily_realized_pnl <= -self.config.max_daily_loss:
                logger.error(
                    f"Daily loss limit triggered: ${-self._daily_realized_pnl:.2f}",
                    daily_pnl=self._daily_realized_pnl,
                )
                self.activate_kill_switch("Daily loss limit reached")
            
            if self._unrealized_pnl <= -self.config.max_unrealized_loss:
                logger.error(
                    f"Unrealized loss limit triggered: ${-self._unrealized_pnl:.2f}",
                    unrealized_pnl=self._unrealized_pnl,
                )
                self.activate_kill_switch("Unrealized loss limit reached")
    
    # -------------------------------------------------------------------------
    # Kill Switch
    # -------------------------------------------------------------------------
    
    def activate_kill_switch(self, reason: str = "Manual activation") -> None:
        """
        Activate the kill switch to halt all trading.
        
        Args:
            reason: Reason for activation
        """
        with self._lock:
            if not self._kill_switch_active:
                self._kill_switch_active = True
                logger.critical(
                    f"KILL SWITCH ACTIVATED: {reason}",
                    reason=reason,
                    daily_pnl=self._daily_realized_pnl,
                    unrealized_pnl=self._unrealized_pnl,
                )
                
                # Notify callbacks
                for callback in self._kill_switch_callbacks:
                    try:
                        callback()
                    except Exception as e:
                        logger.error(f"Kill switch callback error: {e}")
    
    def deactivate_kill_switch(self, confirm: str = "") -> bool:
        """
        Deactivate the kill switch.
        
        Args:
            confirm: Must be "CONFIRM" to deactivate
            
        Returns:
            True if deactivated
        """
        if confirm != "CONFIRM":
            logger.warning("Kill switch deactivation requires 'CONFIRM'")
            return False
        
        with self._lock:
            self._kill_switch_active = False
            logger.warning("Kill switch deactivated - trading resumed")
            return True
    
    @property
    def is_kill_switch_active(self) -> bool:
        """Check if kill switch is active."""
        return self._kill_switch_active
    
    def on_kill_switch(self, callback: Callable[[], None]) -> None:
        """Register callback for kill switch activation."""
        self._kill_switch_callbacks.append(callback)
    
    # -------------------------------------------------------------------------
    # Status and Metrics
    # -------------------------------------------------------------------------
    
    def get_status(self) -> Dict:
        """Get current risk manager status."""
        with self._lock:
            return {
                "kill_switch_active": self._kill_switch_active,
                "daily_realized_pnl": self._daily_realized_pnl,
                "unrealized_pnl": self._unrealized_pnl,
                "daily_loss_remaining": self.config.max_daily_loss + self._daily_realized_pnl,
                "trades_last_minute": sum(
                    1 for t in self._recent_trades 
                    if (datetime.now() - t).total_seconds() < 60
                ),
                "trades_last_hour": sum(
                    1 for t in self._recent_trades 
                    if (datetime.now() - t).total_seconds() < 3600
                ),
                "is_trading_hours": self._is_trading_hours(),
            }
    
    def log_risk_metrics(self, portfolio_delta: float = 0.0, portfolio_gamma: float = 0.0) -> None:
        """Log current risk metrics."""
        status = self.get_status()
        
        log_metrics(
            realized_pnl=self._daily_realized_pnl,
            unrealized_pnl=self._unrealized_pnl,
            portfolio_delta=portfolio_delta,
            portfolio_gamma=portfolio_gamma,
            n_trades=status["trades_last_hour"],
        )
