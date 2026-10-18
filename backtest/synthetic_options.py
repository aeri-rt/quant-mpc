"""
Synthetic option position factory for backtesting.

Creates and reprices stateful OptionPosition objects using
BlackScholesNumeric.  Positions persist across bars -- only their
market values and Greeks are updated each tick.
"""

from datetime import datetime, timedelta
from typing import List, Tuple

from src.models.black_scholes import BlackScholesNumeric
from src.models.greeks import OptionPosition, PositionType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def compute_T(expiry: datetime, sim_time: datetime) -> float:
    """Time to expiry in years, bypassing OptionPosition.time_to_expiry
    which uses datetime.now()."""
    dt = (expiry - sim_time).total_seconds()
    return max(dt / (365.25 * 24 * 3600), 1e-8)


def compute_dte(expiry: datetime, sim_time: datetime) -> int:
    """Integer days to expiry, bypassing OptionPosition.dte."""
    dt = (expiry - sim_time).total_seconds()
    return max(int(dt / 86400), 0)


def nearest_strike(spot: float, tick: float = 1.0) -> float:
    """Round *spot* to the nearest strike on the given tick grid."""
    return round(spot / tick) * tick


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_straddle(
    spot: float,
    iv: float,
    sim_time: datetime,
    target_dte: int,
    quantity: int,
    risk_free_rate: float = 0.045,
    dividend_yield: float = 0.0106,
) -> List[OptionPosition]:
    """
    Create an ATM straddle (call + put) at the nearest $1 strike.

    The two OptionPosition objects are priced immediately using
    BlackScholesNumeric so ``current_price`` and ``greeks`` are populated.

    Args:
        spot: Current underlying price.
        iv: Annualised implied volatility (decimal).
        sim_time: Current simulation timestamp.
        target_dte: Desired days to expiry.
        quantity: Contracts per leg (positive = long).
        risk_free_rate: Risk-free rate for BS pricing.
        dividend_yield: Continuous dividend yield.

    Returns:
        [call_position, put_position]
    """
    strike = nearest_strike(spot)
    expiry = sim_time + timedelta(days=target_dte)
    T = compute_T(expiry, sim_time)

    call_price = BlackScholesNumeric.call_price(spot, strike, T, risk_free_rate, iv, dividend_yield)
    put_price = BlackScholesNumeric.put_price(spot, strike, T, risk_free_rate, iv, dividend_yield)

    call_greeks = BlackScholesNumeric.call_greeks(spot, strike, T, risk_free_rate, iv, dividend_yield)
    put_greeks = BlackScholesNumeric.put_greeks(spot, strike, T, risk_free_rate, iv, dividend_yield)

    exp_str = expiry.strftime("%Y%m%d")

    call_pos = OptionPosition(
        symbol=f"SPY{exp_str}C{int(strike * 1000):08d}",
        position_type=PositionType.CALL,
        strike=strike,
        expiry=expiry,
        quantity=quantity,
        multiplier=100,
        current_price=call_price,
        underlying_price=spot,
        iv=iv,
        greeks=call_greeks,
        greeks_source="calculated",
    )

    put_pos = OptionPosition(
        symbol=f"SPY{exp_str}P{int(strike * 1000):08d}",
        position_type=PositionType.PUT,
        strike=strike,
        expiry=expiry,
        quantity=quantity,
        multiplier=100,
        current_price=put_price,
        underlying_price=spot,
        iv=iv,
        greeks=put_greeks,
        greeks_source="calculated",
    )

    return [call_pos, put_pos]


def reprice_positions(
    positions: List[OptionPosition],
    spot: float,
    iv: float,
    sim_time: datetime,
    risk_free_rate: float = 0.045,
    dividend_yield: float = 0.0106,
) -> None:
    """
    Reprice existing option positions in-place for a new bar.

    Updates ``current_price``, ``underlying_price``, ``iv``, and ``greeks``
    on each position.  T is computed from (expiry - sim_time), not
    datetime.now().
    """
    for pos in positions:
        T = compute_T(pos.expiry, sim_time)
        K = pos.strike

        if pos.is_call:
            pos.current_price = BlackScholesNumeric.call_price(
                spot, K, T, risk_free_rate, iv, dividend_yield
            )
            pos.greeks = BlackScholesNumeric.call_greeks(
                spot, K, T, risk_free_rate, iv, dividend_yield
            )
        else:
            pos.current_price = BlackScholesNumeric.put_price(
                spot, K, T, risk_free_rate, iv, dividend_yield
            )
            pos.greeks = BlackScholesNumeric.put_greeks(
                spot, K, T, risk_free_rate, iv, dividend_yield
            )

        pos.underlying_price = spot
        pos.iv = iv
        pos.greeks_source = "calculated"
