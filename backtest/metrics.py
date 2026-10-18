"""
Backtest metrics collection and analytics.

Replaces the live-trading ``MetricsTracker`` with a simulation-clock-aware
implementation that computes backtest-specific statistics (Sharpe, drawdown,
P&L decomposition, etc.).
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.models.greeks import OptionPosition
from backtest.pnl_engine import PnLSnapshot


# =========================================================================
# Data structures
# =========================================================================

@dataclass
class BarRecord:
    """Single-bar metrics snapshot."""
    timestamp: datetime
    spot: float
    net_delta: float
    portfolio_gamma: float
    equity_qty: int
    trade_quantity: int       # shares traded this bar (0 if no trade)
    trade_direction: str      # "BUY" / "SELL" / "" (from HedgeRecommendation)
    total_pnl: float
    equity_pnl: float
    option_unrealized: float
    costs: float
    mpc_status: str


@dataclass
class BacktestResult:
    """Complete results from a backtest run, passed to the report generator."""
    config: object                            # BacktestConfig
    bars: pd.DataFrame                        # Input price data
    records: List[BarRecord]                  # Per-bar metric snapshots
    pnl_engine: object                        # PnLEngine (for equity curve / decomp)
    total_bars: int = 0
    warmup_bars: int = 0
    num_rolls: int = 0


# =========================================================================
# Metrics collector
# =========================================================================

class BacktestMetrics:
    """
    Simulation-clock-aware metrics collection.

    Accepts explicit timestamps (never calls datetime.now()).
    """

    def __init__(self) -> None:
        self._records: List[BarRecord] = []

    # ------------------------------------------------------------------
    def record(
        self,
        timestamp: datetime,
        spot: float,
        rec,                        # HedgeRecommendation from strategy.update()
        snap: PnLSnapshot,
        option_positions: List[OptionPosition],
    ) -> None:
        """Record a single bar's metrics."""
        net_delta = getattr(rec, "net_delta", 0.0)
        port_gamma = getattr(rec, "portfolio_gamma", 0.0)
        trade_qty = int(getattr(rec, "trade_quantity", 0))
        direction = getattr(rec, "trade_direction", "")
        if hasattr(direction, "value"):
            direction = direction.value
        mpc_status = getattr(rec, "mpc_status", "")
        if hasattr(mpc_status, "value"):
            mpc_status = mpc_status.value

        self._records.append(BarRecord(
            timestamp=timestamp,
            spot=spot,
            net_delta=net_delta,
            portfolio_gamma=port_gamma,
            equity_qty=snap.equity_position_qty,
            trade_quantity=trade_qty,
            trade_direction=str(direction),
            total_pnl=snap.total_pnl,
            equity_pnl=snap.equity_realized_pnl,
            option_unrealized=snap.option_unrealized_pnl,
            costs=snap.total_costs,
            mpc_status=str(mpc_status),
        ))

    # ------------------------------------------------------------------
    # Analytics
    # ------------------------------------------------------------------

    @property
    def records(self) -> List[BarRecord]:
        return list(self._records)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert bar records to a DataFrame indexed by timestamp."""
        if not self._records:
            return pd.DataFrame()
        rows = [
            {
                "timestamp": r.timestamp,
                "spot": r.spot,
                "net_delta": r.net_delta,
                "portfolio_gamma": r.portfolio_gamma,
                "equity_qty": r.equity_qty,
                "trade_quantity": r.trade_quantity,
                "trade_direction": r.trade_direction,
                "total_pnl": r.total_pnl,
                "equity_pnl": r.equity_pnl,
                "option_unrealized": r.option_unrealized,
                "costs": r.costs,
                "mpc_status": r.mpc_status,
            }
            for r in self._records
        ]
        df = pd.DataFrame(rows)
        df.set_index("timestamp", inplace=True)
        return df

    def total_pnl(self) -> float:
        if not self._records:
            return 0.0
        return self._records[-1].total_pnl

    def sharpe_ratio(self, annual_factor: float = 252.0) -> float:
        """
        Annualised Sharpe ratio from bar-over-bar P&L changes.

        Uses daily aggregation: we group bar-level P&L into daily
        returns to avoid over-counting intraday bars.
        """
        df = self.to_dataframe()
        if df.empty or len(df) < 2:
            return 0.0

        daily = df["total_pnl"].resample("D").last().dropna()
        daily_returns = daily.diff().dropna()
        if daily_returns.empty or daily_returns.std() == 0:
            return 0.0
        return float(
            daily_returns.mean() / daily_returns.std() * np.sqrt(annual_factor)
        )

    def max_drawdown(self) -> float:
        """Maximum drawdown in dollar terms."""
        if not self._records:
            return 0.0
        pnls = np.array([r.total_pnl for r in self._records])
        running_max = np.maximum.accumulate(pnls)
        drawdowns = running_max - pnls
        return float(np.max(drawdowns))

    def max_drawdown_pct(self, initial_capital: float = 100_000.0) -> float:
        """Maximum drawdown as percentage of (initial_capital + peak P&L)."""
        if not self._records:
            return 0.0
        pnls = np.array([r.total_pnl for r in self._records])
        peaks = np.maximum.accumulate(pnls + initial_capital)
        drawdowns = peaks - (pnls + initial_capital)
        pct_dd = drawdowns / np.where(peaks > 0, peaks, 1.0)
        return float(np.max(pct_dd))

    def avg_abs_delta(self) -> float:
        """Average absolute portfolio delta."""
        if not self._records:
            return 0.0
        return float(np.mean([abs(r.net_delta) for r in self._records]))

    def max_abs_delta(self) -> float:
        """Maximum absolute portfolio delta."""
        if not self._records:
            return 0.0
        return float(np.max([abs(r.net_delta) for r in self._records]))

    def pct_time_hedged(self, threshold: float = 50.0) -> float:
        """Percentage of bars where |delta| < threshold."""
        if not self._records:
            return 0.0
        hedged = sum(1 for r in self._records if abs(r.net_delta) < threshold)
        return hedged / len(self._records) * 100.0

    def total_trades(self) -> int:
        """Total number of equity hedge trades."""
        return sum(1 for r in self._records if r.trade_quantity != 0)

    def avg_trade_size(self) -> float:
        """Average absolute trade size (shares)."""
        sizes = [abs(r.trade_quantity) for r in self._records if r.trade_quantity != 0]
        return float(np.mean(sizes)) if sizes else 0.0
