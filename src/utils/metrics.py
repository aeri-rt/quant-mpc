"""
Metrics Tracker for Quant-MPC.

Provides real-time metrics tracking and session summaries including:
- Trade metrics (P&L, slippage, execution quality)
- Strategy metrics (hedge ratio, delta exposure)
- Session summaries (daily P&L, trade counts)

Usage:
    from src.utils.metrics import MetricsTracker
    
    metrics = MetricsTracker()
    
    # Record a trade
    metrics.record_trade(
        symbol="SPY",
        side="BUY",
        quantity=100,
        price=500.0,
        expected_price=499.95,
    )
    
    # Record Greeks snapshot
    metrics.record_greeks(delta=50.0, gamma=10.0)
    
    # Get session summary
    summary = metrics.get_session_summary()
"""

from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from typing import List, Dict, Optional
import numpy as np
import json
from pathlib import Path

from src.utils.logging import get_logger, log_metrics

logger = get_logger("metrics")


# =============================================================================
# Trade Record
# =============================================================================

@dataclass
class TradeRecord:
    """Record of a single trade."""
    timestamp: datetime
    symbol: str
    side: str  # BUY or SELL
    quantity: int
    price: float
    expected_price: float = 0.0
    commission: float = 0.0
    
    # Computed
    notional: float = field(init=False)
    slippage: float = field(init=False)
    slippage_bps: float = field(init=False)
    
    def __post_init__(self):
        self.notional = abs(self.quantity) * self.price
        
        if self.expected_price > 0:
            self.slippage = self.price - self.expected_price
            self.slippage_bps = (self.slippage / self.expected_price) * 10000
        else:
            self.slippage = 0.0
            self.slippage_bps = 0.0


@dataclass
class GreeksSnapshot:
    """Snapshot of portfolio Greeks."""
    timestamp: datetime
    spot_price: float
    delta: float
    gamma: float
    theta: float = 0.0
    vega: float = 0.0
    equity_position: int = 0
    net_delta: float = 0.0


@dataclass
class PnLRecord:
    """P&L record."""
    timestamp: datetime
    realized_pnl: float
    unrealized_pnl: float
    total_pnl: float
    position_value: float = 0.0


# =============================================================================
# Metrics Tracker
# =============================================================================

class MetricsTracker:
    """
    Tracks and computes trading metrics.
    
    Maintains history of:
    - All trades with execution quality metrics
    - Greeks snapshots over time
    - P&L evolution
    """
    
    def __init__(self, session_date: Optional[date] = None):
        """
        Initialize metrics tracker.
        
        Args:
            session_date: Date for this session (defaults to today)
        """
        self.session_date = session_date or date.today()
        self.session_start = datetime.now()
        
        # Trade history
        self._trades: List[TradeRecord] = []
        
        # Greeks history
        self._greeks_history: List[GreeksSnapshot] = []
        
        # P&L history
        self._pnl_history: List[PnLRecord] = []
        
        # Running totals
        self._total_bought = 0
        self._total_sold = 0
        self._total_commission = 0.0
        # Note: P&L tracking delegated to IBKR for accuracy
        
        # For execution quality
        self._initial_position_value = 0.0
        
        logger.info(
            "MetricsTracker initialized",
            session_date=self.session_date.isoformat(),
        )
    
    # -------------------------------------------------------------------------
    # Recording
    # -------------------------------------------------------------------------
    
    def record_trade(
        self,
        symbol: str,
        side: str,
        quantity: int,
        price: float,
        expected_price: float = 0.0,
        commission: float = 0.0,
    ) -> TradeRecord:
        """
        Record a trade execution.
        
        Args:
            symbol: Symbol traded
            side: BUY or SELL
            quantity: Shares traded (always positive)
            price: Execution price
            expected_price: Expected/target price for slippage calc
            commission: Commission paid
            
        Returns:
            TradeRecord for this trade
        """
        record = TradeRecord(
            timestamp=datetime.now(),
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=price,
            expected_price=expected_price,
            commission=commission,
        )
        
        self._trades.append(record)
        
        # Update running totals
        if side == "BUY":
            self._total_bought += quantity
        else:
            self._total_sold += quantity
        
        # Note: Realized P&L calculation is complex with short positions
        # IBKR tracks this accurately - we just track trade volume here
        # P&L in session summary will show 0 (check IBKR for actual P&L)
        self._total_commission += commission
        
        logger.debug(
            f"Trade recorded: {side} {quantity} {symbol} @ ${price:.2f}",
            slippage_bps=record.slippage_bps,
            commission=commission,
        )
        
        return record
    
    def record_greeks(
        self,
        spot_price: float,
        delta: float,
        gamma: float,
        theta: float = 0.0,
        vega: float = 0.0,
        equity_position: int = 0,
    ) -> None:
        """Record a Greeks snapshot."""
        net_delta = delta + equity_position
        
        snapshot = GreeksSnapshot(
            timestamp=datetime.now(),
            spot_price=spot_price,
            delta=delta,
            gamma=gamma,
            theta=theta,
            vega=vega,
            equity_position=equity_position,
            net_delta=net_delta,
        )
        
        self._greeks_history.append(snapshot)
    
    def record_pnl(
        self,
        realized_pnl: float,
        unrealized_pnl: float,
        position_value: float = 0.0,
    ) -> None:
        """Record a P&L snapshot."""
        record = PnLRecord(
            timestamp=datetime.now(),
            realized_pnl=realized_pnl,
            unrealized_pnl=unrealized_pnl,
            total_pnl=realized_pnl + unrealized_pnl,
            position_value=position_value,
        )
        
        self._pnl_history.append(record)
    
    # -------------------------------------------------------------------------
    # Trade Metrics
    # -------------------------------------------------------------------------
    
    def get_trade_metrics(self) -> Dict:
        """Get trade execution metrics."""
        if not self._trades:
            return {
                "n_trades": 0,
                "total_volume": 0,
                "total_notional": 0.0,
                "total_commission": 0.0,
                "avg_trade_size": 0.0,
                "avg_slippage_bps": 0.0,
            }
        
        volumes = [t.quantity for t in self._trades]
        notionals = [t.notional for t in self._trades]
        slippages = [t.slippage_bps for t in self._trades if t.expected_price > 0]
        
        return {
            "n_trades": len(self._trades),
            "total_volume": sum(volumes),
            "total_notional": sum(notionals),
            "total_commission": self._total_commission,
            "avg_trade_size": np.mean(volumes),
            "max_trade_size": max(volumes),
            "avg_slippage_bps": np.mean(slippages) if slippages else 0.0,
            "max_slippage_bps": max(slippages) if slippages else 0.0,
            "total_bought": self._total_bought,
            "total_sold": self._total_sold,
            "net_traded": self._total_bought - self._total_sold,
        }
    
    # -------------------------------------------------------------------------
    # Greeks Metrics
    # -------------------------------------------------------------------------
    
    def get_greeks_metrics(self) -> Dict:
        """Get Greeks distribution metrics."""
        if not self._greeks_history:
            return {
                "n_snapshots": 0,
                "avg_delta": 0.0,
                "max_abs_delta": 0.0,
                "avg_gamma": 0.0,
                "pct_time_hedged": 0.0,
            }
        
        deltas = [g.net_delta for g in self._greeks_history]
        gammas = [g.gamma for g in self._greeks_history]
        
        return {
            "n_snapshots": len(self._greeks_history),
            "avg_delta": np.mean(deltas),
            "std_delta": np.std(deltas),
            "max_abs_delta": max(abs(d) for d in deltas),
            "min_delta": min(deltas),
            "max_delta": max(deltas),
            "avg_gamma": np.mean(gammas),
            "pct_time_hedged": np.mean([1 if abs(d) < 50 else 0 for d in deltas]) * 100,
        }
    
    # -------------------------------------------------------------------------
    # P&L Metrics
    # -------------------------------------------------------------------------
    
    def get_pnl_metrics(self) -> Dict:
        """Get P&L metrics. Note: Actual P&L tracked by IBKR."""
        if not self._pnl_history:
            return {
                "realized_pnl": 0.0,  # Check IBKR for actual P&L
                "unrealized_pnl": 0.0,
                "total_pnl": 0.0,
                "max_drawdown": 0.0,
            }
        
        total_pnls = [p.total_pnl for p in self._pnl_history]
        
        # Calculate max drawdown
        peak = total_pnls[0]
        max_drawdown = 0.0
        for pnl in total_pnls:
            if pnl > peak:
                peak = pnl
            drawdown = peak - pnl
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        
        latest = self._pnl_history[-1]
        
        return {
            "realized_pnl": latest.realized_pnl,
            "unrealized_pnl": latest.unrealized_pnl,
            "total_pnl": latest.total_pnl,
            "max_drawdown": max_drawdown,
            "peak_pnl": max(total_pnls),
            "trough_pnl": min(total_pnls),
        }
    
    # -------------------------------------------------------------------------
    # Session Summary
    # -------------------------------------------------------------------------
    
    def get_session_summary(self) -> Dict:
        """Get complete session summary."""
        duration = datetime.now() - self.session_start
        
        trade_metrics = self.get_trade_metrics()
        greeks_metrics = self.get_greeks_metrics()
        pnl_metrics = self.get_pnl_metrics()
        
        summary = {
            "session_date": self.session_date.isoformat(),
            "session_start": self.session_start.isoformat(),
            "duration_minutes": duration.total_seconds() / 60,
            
            # Trade summary
            "trades": trade_metrics,
            
            # Greeks summary
            "greeks": greeks_metrics,
            
            # P&L summary
            "pnl": pnl_metrics,
            
            # Costs
            "total_costs": self._total_commission,
            "net_pnl_after_costs": pnl_metrics["total_pnl"] - self._total_commission,
        }
        
        return summary
    
    def log_session_summary(self) -> None:
        """Log the session summary."""
        summary = self.get_session_summary()
        
        # Build summary string (avoid loguru multiline issues)
        summary_text = (
            f"Session {summary['session_date']} | "
            f"Duration: {summary['duration_minutes']:.1f}min | "
            f"Trades: {summary['trades']['n_trades']} | "
            f"Volume: {summary['trades']['total_volume']} | "
            f"Avg Δ: {summary['greeks']['avg_delta']:.1f} | "
            f"P&L: ${summary['pnl']['total_pnl']:,.2f}"
        )
        logger.info(summary_text, event="session_summary")
        
        # Also log to metrics
        log_metrics(
            realized_pnl=summary['pnl']['realized_pnl'],
            unrealized_pnl=summary['pnl']['unrealized_pnl'],
            portfolio_delta=summary['greeks']['avg_delta'],
            n_trades=summary['trades']['n_trades'],
            total_volume=summary['trades']['total_volume'],
            total_commission=summary['total_costs'],
            max_drawdown=summary['pnl']['max_drawdown'],
        )
    
    def save_to_file(self, filepath: Optional[Path] = None) -> Path:
        """
        Save session data to JSON file.
        
        Args:
            filepath: Output path (defaults to logs/session_YYYY-MM-DD.json)
            
        Returns:
            Path to saved file
        """
        if filepath is None:
            filepath = Path("logs") / f"session_{self.session_date.isoformat()}.json"
        
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "summary": self.get_session_summary(),
            "trades": [
                {
                    "timestamp": t.timestamp.isoformat(),
                    "symbol": t.symbol,
                    "side": t.side,
                    "quantity": t.quantity,
                    "price": t.price,
                    "slippage_bps": t.slippage_bps,
                    "commission": t.commission,
                }
                for t in self._trades
            ],
            "greeks_history": [
                {
                    "timestamp": g.timestamp.isoformat(),
                    "spot_price": g.spot_price,
                    "delta": g.delta,
                    "gamma": g.gamma,
                    "net_delta": g.net_delta,
                }
                for g in self._greeks_history[-100:]  # Last 100 snapshots
            ],
        }
        
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Session data saved to {filepath}")
        return filepath


# =============================================================================
# Convenience Functions
# =============================================================================

_global_tracker: Optional[MetricsTracker] = None


def get_metrics_tracker() -> MetricsTracker:
    """Get or create the global metrics tracker."""
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = MetricsTracker()
    return _global_tracker


def reset_metrics_tracker() -> MetricsTracker:
    """Reset the global metrics tracker."""
    global _global_tracker
    _global_tracker = MetricsTracker()
    return _global_tracker
