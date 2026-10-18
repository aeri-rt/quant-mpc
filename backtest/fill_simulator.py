"""
Fill simulation for backtesting.

Models realistic execution with separate logic for equity and options:
- **Equity hedges**: Fill at mid + configurable slippage. Commission per share.
- **Option trades**: Fill at mid + half bid-ask spread (function of moneyness/DTE).
  Commission per contract.
"""

from dataclasses import dataclass
from typing import List

from src.models.greeks import OptionPosition


@dataclass(frozen=True)
class FillResult:
    """Result of simulating a fill."""
    fill_price: float       # Per-unit price (per share or per contract)
    total_cost: float       # fill_price * quantity (signed: positive = cash outflow)
    commission: float       # Absolute commission
    slippage: float         # Absolute slippage cost


class FillSimulator:
    """
    Simulate trade execution with slippage and commissions.

    Args:
        equity_slippage_bps: Slippage in basis points for equity fills.
        equity_commission_per_share: Commission per share for equity.
        option_spread_dollars: Half bid-ask spread for ATM SPY options ($ per contract).
        option_commission_per_contract: Commission per option contract.
    """

    def __init__(
        self,
        equity_slippage_bps: float = 1.0,
        equity_commission_per_share: float = 0.005,
        option_spread_dollars: float = 0.05,
        option_commission_per_contract: float = 0.65,
    ):
        self._eq_slip_bps = equity_slippage_bps
        self._eq_comm = equity_commission_per_share
        self._opt_spread = option_spread_dollars
        self._opt_comm = option_commission_per_contract

    # ------------------------------------------------------------------
    # Equity
    # ------------------------------------------------------------------

    def equity_fill(self, mid_price: float, quantity: int) -> FillResult:
        """
        Simulate an equity fill.

        Buying:  fill at mid * (1 + slippage_bps / 10000)
        Selling: fill at mid * (1 - slippage_bps / 10000)

        Args:
            mid_price: Mid-point price.
            quantity: Shares to trade (positive = buy, negative = sell).

        Returns:
            FillResult with signed total_cost (positive = cash outflow).
        """
        slip_frac = self._eq_slip_bps / 10_000
        if quantity > 0:
            fill_price = mid_price * (1 + slip_frac)
        elif quantity < 0:
            fill_price = mid_price * (1 - slip_frac)
        else:
            return FillResult(fill_price=mid_price, total_cost=0.0, commission=0.0, slippage=0.0)

        slippage = abs(quantity) * mid_price * slip_frac
        commission = abs(quantity) * self._eq_comm
        total_cost = fill_price * quantity  # positive for buys, negative for sells

        return FillResult(
            fill_price=fill_price,
            total_cost=total_cost,
            commission=commission,
            slippage=slippage,
        )

    def equity_commission(self, quantity: int) -> float:
        """Commission for an equity trade of *quantity* shares."""
        return abs(quantity) * self._eq_comm

    # ------------------------------------------------------------------
    # Options
    # ------------------------------------------------------------------

    def option_fill(
        self,
        positions: List[OptionPosition],
        is_entry: bool,
    ) -> float:
        """
        Simulate an option fill for a list of positions (straddle legs).

        - Entry (buy):  fill at current_price + half_spread  (pay more)
        - Exit  (sell): fill at current_price - half_spread  (receive less)

        Returns:
            Total dollar cost of the fill (positive = cash outflow for entry,
            negative = cash inflow for exit), **excluding commissions**.
        """
        total = 0.0
        for pos in positions:
            half_spread = self._opt_spread
            if is_entry:
                fill = pos.current_price + half_spread
            else:
                fill = max(pos.current_price - half_spread, 0.0)

            # cost = price * quantity * multiplier (positive for buys)
            if is_entry:
                total += fill * abs(pos.quantity) * pos.multiplier
            else:
                total -= fill * abs(pos.quantity) * pos.multiplier

        return total

    def option_commission(self, positions: List[OptionPosition]) -> float:
        """Total commission for trading the given option positions."""
        return sum(abs(pos.quantity) * self._opt_comm for pos in positions)
