"""
P&L engine for backtesting.

This is the **centrepiece** of the backtest system.  The live system
delegates P&L tracking to IBKR; this module replaces that with a
fully self-contained accounting engine that tracks:

- Equity hedge realised P&L (cost-basis per trade)
- Option realised P&L (entry value vs exit value)
- Option unrealised P&L (mark-to-market vs entry value)
- Transaction costs (commissions + slippage)
- P&L decomposition (gamma P&L, theta decay, costs)
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd

from src.models.greeks import OptionPosition


# =========================================================================
# Data structures
# =========================================================================

@dataclass
class PnLSnapshot:
    """Point-in-time P&L state."""
    timestamp: datetime

    # Equity hedge
    equity_realized_pnl: float = 0.0
    equity_position_qty: int = 0
    equity_cost_basis: float = 0.0       # avg cost per share (signed: long > 0)
    equity_market_value: float = 0.0

    # Options
    option_realized_pnl: float = 0.0
    option_entry_value: float = 0.0      # total $ paid to open current straddle
    option_market_value: float = 0.0     # current mark-to-market

    # Costs
    total_commissions: float = 0.0
    total_slippage: float = 0.0

    # Derived
    @property
    def option_unrealized_pnl(self) -> float:
        return self.option_market_value - self.option_entry_value

    @property
    def total_costs(self) -> float:
        return self.total_commissions + self.total_slippage

    @property
    def total_pnl(self) -> float:
        return (
            self.equity_realized_pnl
            + self.option_realized_pnl
            + self.option_unrealized_pnl
            - self.total_costs
        )

    @property
    def equity_unrealized_pnl(self) -> float:
        if self.equity_position_qty == 0:
            return 0.0
        return self.equity_market_value - (self.equity_cost_basis * self.equity_position_qty)


# =========================================================================
# P&L Engine
# =========================================================================

class PnLEngine:
    """
    Self-contained P&L accounting engine.

    Tracks cost basis, realised gains, unrealised mark-to-market,
    and transaction costs across both equity hedges and option positions.
    """

    def __init__(self, initial_capital: float = 100_000.0):
        self._initial_capital = initial_capital

        # Equity hedge book-keeping (FIFO cost basis)
        self._equity_qty: int = 0
        self._equity_cost_basis: float = 0.0   # weighted avg cost per share
        self._equity_realized: float = 0.0

        # Option book-keeping
        self._option_entry_value: float = 0.0  # total $ of current open straddle
        self._option_realized: float = 0.0
        self._option_market_value: float = 0.0

        # Costs
        self._total_commissions: float = 0.0
        self._total_slippage: float = 0.0

        # Time-series for equity curve & decomposition
        self._equity_curve: List[Tuple[datetime, float]] = []
        self._snapshots: List[PnLSnapshot] = []

        # Trade-level records (for decomposition / debugging)
        self._equity_trades: List[dict] = []
        self._option_trades: List[dict] = []

        # For theta / gamma decomposition tracking
        self._prev_option_value: Optional[float] = None
        self._prev_spot: Optional[float] = None
        self._cum_gamma_pnl: float = 0.0
        self._cum_theta_pnl: float = 0.0

    # ------------------------------------------------------------------
    # Equity hedge trades
    # ------------------------------------------------------------------

    def record_equity_trade(
        self,
        quantity: int,
        fill_price: float,
        commission: float,
        slippage: float,
        timestamp: datetime,
    ) -> None:
        """
        Record an equity hedge trade.

        Uses weighted-average cost basis.  When the trade reduces the
        position, realised P&L is booked.  When the trade extends or
        opens, cost basis is adjusted.
        """
        self._total_commissions += commission
        self._total_slippage += slippage

        self._equity_trades.append({
            "timestamp": timestamp,
            "quantity": quantity,
            "fill_price": fill_price,
            "commission": commission,
        })

        old_qty = self._equity_qty
        new_qty = old_qty + quantity

        if old_qty == 0:
            # Opening a fresh position
            self._equity_cost_basis = fill_price
        elif _same_sign(old_qty, quantity):
            # Adding to position -- adjust avg cost
            total_cost = self._equity_cost_basis * old_qty + fill_price * quantity
            self._equity_cost_basis = total_cost / new_qty
        else:
            # Reducing / flipping position -- realise P&L on the closed portion
            closed_qty = min(abs(quantity), abs(old_qty))
            if old_qty > 0:
                # Was long, selling some
                realised = (fill_price - self._equity_cost_basis) * closed_qty
            else:
                # Was short, buying some
                realised = (self._equity_cost_basis - fill_price) * closed_qty
            self._equity_realized += realised

            if abs(quantity) > abs(old_qty):
                # Flipped direction -- remainder opens at new price
                self._equity_cost_basis = fill_price
            # else: cost basis unchanged for remaining shares

        self._equity_qty = new_qty

    # ------------------------------------------------------------------
    # Option trades
    # ------------------------------------------------------------------

    def record_option_entry(
        self,
        positions: List[OptionPosition],
        total_cost: float,
        commission: float,
        timestamp: datetime,
    ) -> None:
        """
        Record opening a new straddle (or any set of option positions).

        Args:
            positions: The option positions opened.
            total_cost: Total $ paid (fill price * qty * multiplier), positive.
            commission: Commission for the trade.
            timestamp: Simulation time.
        """
        self._option_entry_value = total_cost
        self._option_market_value = total_cost  # at entry, MtM = cost
        self._total_commissions += commission

        self._option_trades.append({
            "timestamp": timestamp,
            "type": "entry",
            "cost": total_cost,
            "commission": commission,
        })

        # Reset decomposition baseline
        self._prev_option_value = total_cost
        self._prev_spot = None

    def record_option_exit(
        self,
        positions: List[OptionPosition],
        total_proceeds: float,
        commission: float,
        timestamp: datetime,
    ) -> None:
        """
        Record closing an option straddle.

        Args:
            positions: Positions being closed.
            total_proceeds: Absolute $ received (positive number).
            commission: Commission for the trade.
            timestamp: Simulation time.
        """
        # total_proceeds comes from fill_sim.option_fill(is_entry=False)
        # which returns negative (cash inflow); we take absolute value
        proceeds_abs = abs(total_proceeds)
        realised = proceeds_abs - self._option_entry_value
        self._option_realized += realised
        self._total_commissions += commission

        self._option_trades.append({
            "timestamp": timestamp,
            "type": "exit",
            "proceeds": proceeds_abs,
            "realized": realised,
            "commission": commission,
        })

        # Reset option state
        self._option_entry_value = 0.0
        self._option_market_value = 0.0
        self._prev_option_value = None

    # ------------------------------------------------------------------
    # Mark-to-market
    # ------------------------------------------------------------------

    def mark_to_market(
        self,
        option_value: float,
        equity_value: float,
        spot: float,
        timestamp: datetime,
        portfolio_gamma: float = 0.0,
    ) -> None:
        """
        Update mark-to-market values and record to equity curve.

        Also updates the theta/gamma decomposition estimate.

        Args:
            option_value: Current total option MtM (price * qty * multiplier).
            equity_value: Current equity position value (qty * spot).
            spot: Current spot price.
            timestamp: Simulation time.
            portfolio_gamma: Dollar gamma of the portfolio (for decomposition).
        """
        self._option_market_value = option_value

        # -- Theta / Gamma decomposition --
        if self._prev_option_value is not None and self._prev_spot is not None:
            ds = spot - self._prev_spot
            # Gamma P&L ≈ 0.5 * portfolio_gamma * ds^2
            gamma_pnl_bar = 0.5 * portfolio_gamma * ds * ds
            # Theta P&L = actual option change - gamma P&L component
            option_change = option_value - self._prev_option_value
            theta_pnl_bar = option_change - gamma_pnl_bar
            self._cum_gamma_pnl += gamma_pnl_bar
            self._cum_theta_pnl += theta_pnl_bar

        self._prev_option_value = option_value
        self._prev_spot = spot

        # Record equity curve point
        snap = self.snapshot(timestamp)
        total = snap.total_pnl
        self._equity_curve.append((timestamp, total))
        self._snapshots.append(snap)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def snapshot(self, timestamp: Optional[datetime] = None) -> PnLSnapshot:
        """Return a point-in-time snapshot of all P&L components."""
        ts = timestamp or datetime.now()
        spot = self._prev_spot or 0.0
        return PnLSnapshot(
            timestamp=ts,
            equity_realized_pnl=self._equity_realized,
            equity_position_qty=self._equity_qty,
            equity_cost_basis=self._equity_cost_basis,
            equity_market_value=self._equity_qty * spot,
            option_realized_pnl=self._option_realized,
            option_entry_value=self._option_entry_value,
            option_market_value=self._option_market_value,
            total_commissions=self._total_commissions,
            total_slippage=self._total_slippage,
        )

    def get_equity_curve(self) -> pd.Series:
        """Timestamped total P&L series."""
        if not self._equity_curve:
            return pd.Series(dtype=float)
        ts, vals = zip(*self._equity_curve)
        return pd.Series(vals, index=pd.DatetimeIndex(ts), name="total_pnl")

    def get_decomposition(self) -> Dict[str, float]:
        """
        P&L decomposition.

        Returns dict with keys:
        - equity_pnl: Realised equity hedge P&L
        - option_pnl: Realised + unrealised option P&L
        - gamma_pnl: Cumulative estimated gamma P&L
        - theta_pnl: Cumulative estimated theta decay
        - costs: Total commissions + slippage
        """
        return {
            "equity_pnl": self._equity_realized,
            "option_pnl": self._option_realized + (self._option_market_value - self._option_entry_value),
            "gamma_pnl": self._cum_gamma_pnl,
            "theta_pnl": self._cum_theta_pnl,
            "costs": self._total_commissions + self._total_slippage,
        }

    @property
    def equity_qty(self) -> int:
        return self._equity_qty

    @property
    def total_trades(self) -> int:
        return len(self._equity_trades)

    @property
    def option_rolls(self) -> int:
        return sum(1 for t in self._option_trades if t["type"] == "exit")


# =========================================================================
# Helpers
# =========================================================================

def _same_sign(a: int, b: int) -> bool:
    """True if a and b have the same sign (both positive or both negative)."""
    return (a > 0 and b > 0) or (a < 0 and b < 0)
