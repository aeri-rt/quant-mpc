"""
Position Manager for option rolling and exit logic.

Monitors option positions and determines when to:
- Roll positions approaching expiration
- Exit positions based on P&L thresholds
- Adjust positions based on delta/gamma targets

This module handles the higher-level position management decisions,
while the MPC controller handles the tactical hedge execution.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Tuple
from enum import Enum

from src.models.greeks import OptionPosition, PositionType
from src.utils.logging import get_logger

logger = get_logger("position_manager")


class PositionAction(str, Enum):
    """Action to take on a position."""
    HOLD = "HOLD"
    ROLL = "ROLL"           # Close and open new position at different expiry
    EXIT_PROFIT = "EXIT_PROFIT"   # Close due to take profit
    EXIT_LOSS = "EXIT_LOSS"       # Close due to stop loss
    EXIT_DTE = "EXIT_DTE"         # Close due to approaching expiration
    ADJUST = "ADJUST"       # Adjust position size


@dataclass
class PositionSignal:
    """Signal for position action."""
    position: OptionPosition
    action: PositionAction
    reason: str
    target_dte: Optional[int] = None  # For ROLL: target DTE for new position
    pnl_pct: Optional[float] = None   # P&L percentage at decision time
    

@dataclass
class PositionManagerConfig:
    """Configuration for position management rules."""
    
    # Rolling rules
    roll_dte: int = 7               # Roll when DTE falls below this
    roll_to_dte: int = 30           # Roll into position with this DTE
    enable_rolling: bool = True
    
    # P&L exit rules
    take_profit_pct: float = 0.50   # Exit at 50% profit
    stop_loss_pct: float = -0.30    # Exit at 30% loss
    enable_pnl_exits: bool = True
    
    # Position sizing
    max_gamma_exposure: float = 100.0  # Max portfolio gamma
    rebalance_threshold: float = 0.20  # Rebalance if delta moves 20% from target
    
    # Timing
    min_hold_period_hours: float = 4.0  # Minimum time before considering exit


class PositionManager:
    """
    Manages option position lifecycle including rolling and exits.
    
    Usage:
        config = PositionManagerConfig(roll_dte=7, take_profit_pct=0.50)
        manager = PositionManager(config)
        
        # Check positions
        signals = manager.check_positions(positions)
        
        for signal in signals:
            if signal.action == PositionAction.ROLL:
                # Execute roll logic
            elif signal.action == PositionAction.EXIT_PROFIT:
                # Close position
    """
    
    def __init__(self, config: Optional[PositionManagerConfig] = None):
        """Initialize position manager."""
        self.config = config or PositionManagerConfig()
        
        # Track entry prices for P&L calculation
        self._entry_prices: dict = {}  # symbol -> entry_price
        self._entry_times: dict = {}   # symbol -> entry_datetime
        
        logger.info(
            "PositionManager initialized",
            roll_dte=self.config.roll_dte,
            take_profit=f"{self.config.take_profit_pct:.0%}",
            stop_loss=f"{self.config.stop_loss_pct:.0%}",
        )
    
    def set_entry_price(self, symbol: str, price: float) -> None:
        """
        Record entry price for a position.
        
        Args:
            symbol: Option symbol
            price: Entry price (per contract)
        """
        self._entry_prices[symbol] = price
        self._entry_times[symbol] = datetime.now()
        logger.debug(f"Entry recorded for {symbol} @ ${price:.2f}")
    
    def get_pnl_pct(self, position: OptionPosition) -> Optional[float]:
        """
        Calculate P&L percentage for a position.
        
        Args:
            position: Option position
            
        Returns:
            P&L as decimal (0.10 = 10% profit), or None if no entry price
        """
        if position.symbol not in self._entry_prices:
            return None
        
        entry = self._entry_prices[position.symbol]
        if entry <= 0:
            return None
        
        return (position.current_price - entry) / entry
    
    def _check_min_hold_period(self, symbol: str) -> bool:
        """Check if minimum hold period has passed."""
        if symbol not in self._entry_times:
            return True  # No entry time recorded, allow exit
        
        entry_time = self._entry_times[symbol]
        hours_held = (datetime.now() - entry_time).total_seconds() / 3600
        return hours_held >= self.config.min_hold_period_hours
    
    def should_roll(self, position: OptionPosition) -> Tuple[bool, str]:
        """
        Check if position should be rolled due to approaching expiration.
        
        Args:
            position: Option position to check
            
        Returns:
            (should_roll, reason)
        """
        if not self.config.enable_rolling:
            return False, ""
        
        dte = position.dte
        
        if dte <= self.config.roll_dte:
            return True, f"DTE={dte} below threshold {self.config.roll_dte}"
        
        return False, ""
    
    def should_exit(self, position: OptionPosition) -> Tuple[bool, PositionAction, str]:
        """
        Check if position should be exited based on P&L.
        
        Args:
            position: Option position to check
            
        Returns:
            (should_exit, action, reason)
        """
        if not self.config.enable_pnl_exits:
            return False, PositionAction.HOLD, ""
        
        # Check minimum hold period
        if not self._check_min_hold_period(position.symbol):
            return False, PositionAction.HOLD, "Min hold period not met"
        
        pnl_pct = self.get_pnl_pct(position)
        if pnl_pct is None:
            return False, PositionAction.HOLD, "No entry price recorded"
        
        # Check take profit
        if pnl_pct >= self.config.take_profit_pct:
            return True, PositionAction.EXIT_PROFIT, f"Take profit: {pnl_pct:.1%}"
        
        # Check stop loss
        if pnl_pct <= self.config.stop_loss_pct:
            return True, PositionAction.EXIT_LOSS, f"Stop loss: {pnl_pct:.1%}"
        
        return False, PositionAction.HOLD, ""
    
    def check_positions(
        self,
        positions: List[OptionPosition],
    ) -> List[PositionSignal]:
        """
        Check all positions and return action signals.
        
        Args:
            positions: List of current option positions
            
        Returns:
            List of PositionSignal for positions requiring action
        """
        signals = []
        
        for pos in positions:
            # Check for rolling first (higher priority)
            should_roll, roll_reason = self.should_roll(pos)
            if should_roll:
                signals.append(PositionSignal(
                    position=pos,
                    action=PositionAction.ROLL,
                    reason=roll_reason,
                    target_dte=self.config.roll_to_dte,
                    pnl_pct=self.get_pnl_pct(pos),
                ))
                continue
            
            # Check for P&L exit
            should_exit, exit_action, exit_reason = self.should_exit(pos)
            if should_exit:
                signals.append(PositionSignal(
                    position=pos,
                    action=exit_action,
                    reason=exit_reason,
                    pnl_pct=self.get_pnl_pct(pos),
                ))
        
        return signals
    
    def check_straddle(
        self,
        call: OptionPosition,
        put: OptionPosition,
    ) -> List[PositionSignal]:
        """
        Check a straddle pair and return action signals.
        
        For straddles, we typically roll both legs together.
        
        Args:
            call: Call leg of straddle
            put: Put leg of straddle
            
        Returns:
            List of PositionSignal (may include both legs)
        """
        signals = []
        
        # Check DTE (same for both legs)
        should_roll, roll_reason = self.should_roll(call)
        if should_roll:
            # Roll both legs together
            signals.extend([
                PositionSignal(
                    position=call,
                    action=PositionAction.ROLL,
                    reason=f"Straddle roll: {roll_reason}",
                    target_dte=self.config.roll_to_dte,
                    pnl_pct=self.get_pnl_pct(call),
                ),
                PositionSignal(
                    position=put,
                    action=PositionAction.ROLL,
                    reason=f"Straddle roll: {roll_reason}",
                    target_dte=self.config.roll_to_dte,
                    pnl_pct=self.get_pnl_pct(put),
                ),
            ])
            return signals
        
        # Check combined P&L for straddle
        call_pnl = self.get_pnl_pct(call)
        put_pnl = self.get_pnl_pct(put)
        
        if call_pnl is not None and put_pnl is not None:
            # Use average P&L for straddle decision
            avg_pnl = (call_pnl + put_pnl) / 2
            
            # Check take profit
            if avg_pnl >= self.config.take_profit_pct:
                if self._check_min_hold_period(call.symbol):
                    signals.extend([
                        PositionSignal(
                            position=call,
                            action=PositionAction.EXIT_PROFIT,
                            reason=f"Straddle take profit: {avg_pnl:.1%}",
                            pnl_pct=call_pnl,
                        ),
                        PositionSignal(
                            position=put,
                            action=PositionAction.EXIT_PROFIT,
                            reason=f"Straddle take profit: {avg_pnl:.1%}",
                            pnl_pct=put_pnl,
                        ),
                    ])
                    return signals
            
            # Check stop loss
            if avg_pnl <= self.config.stop_loss_pct:
                if self._check_min_hold_period(call.symbol):
                    signals.extend([
                        PositionSignal(
                            position=call,
                            action=PositionAction.EXIT_LOSS,
                            reason=f"Straddle stop loss: {avg_pnl:.1%}",
                            pnl_pct=call_pnl,
                        ),
                        PositionSignal(
                            position=put,
                            action=PositionAction.EXIT_LOSS,
                            reason=f"Straddle stop loss: {avg_pnl:.1%}",
                            pnl_pct=put_pnl,
                        ),
                    ])
        
        return signals
    
    def clear_position(self, symbol: str) -> None:
        """Clear entry data for a closed position."""
        self._entry_prices.pop(symbol, None)
        self._entry_times.pop(symbol, None)
