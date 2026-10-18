"""The one cost model, used by band, QP, gate, risk and SimBroker. All $; `n` is a SIGNED share
count (negative = sell).

Fill convention (single; book and sim_broker follow it): a fill executes at bid/ask, so the
half-spread is inside the fill price. `commission()` + `reg_fees()` are charged on the fill
(`Fill.commission`). `per_share_rate()` and `option_cost()` are planning inputs only (band k,
QP c_side, gate) and are never used to charge a fill, so slippage is never counted twice.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.config import Costs


def commission(n: int, c: Costs) -> float:
    """IBKR Pro Tiered: max(minimum, per-share * |n|); 0 shares -> 0.0."""
    return max(c.min_commission, c.commission_per_share * abs(n)) if n else 0.0


def reg_fees(n: int, price: float, c: Costs) -> float:
    """SEC Section 31 fee + FINRA TAF (capped); sells only."""
    if n >= 0:
        return 0.0
    s = abs(n)
    return c.sec_fee_rate * price * s + min(c.finra_taf_per_share * s, c.finra_taf_cap)


def per_share_rate(price: float, side: int, c: Costs) -> float:
    """$/share one-way, no minimum; side +1 buy, -1 sell."""
    r = c.commission_per_share + c.half_spread
    return r + (c.sec_fee_rate * price + c.finra_taf_per_share) if side < 0 else r


def option_half_spread(dte: float, c: Costs) -> float:
    """First (max_dte, half_spread) row with max_dte >= dte, else the last row."""
    rows = c.option_half_spread_by_dte
    return next((hs for max_dte, hs in rows if max_dte >= dte), rows[-1][1])


def option_cost(n_contracts: int, dte: float, c: Costs, legs: int = 2) -> float:
    """One option order (straddle = 2 legs): commission with minimum + fees + spread crossed."""
    k = legs * abs(n_contracts)
    if not k:
        return 0.0
    return (
        max(c.option_min_commission, c.option_commission * k)
        + c.option_fees_per_contract * k
        + 100 * option_half_spread(dte, c) * k
    )
