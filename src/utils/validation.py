"""
Data validation utilities for market data.

Provides validation and sanitization for quotes, option quotes, and prices
to prevent NaN, zero, or otherwise invalid data from propagating through
the system and causing silent errors or crashes.
"""

import math
from datetime import datetime
from typing import Optional

from src.core.types import Quote, OptionQuote
from src.utils.logging import get_logger

logger = get_logger("validation")


class ValidationError(Exception):
    """Raised when data fails critical validation."""
    pass


def _is_valid_float(val: float) -> bool:
    """Check if a float is finite and not NaN."""
    return isinstance(val, (int, float)) and not math.isnan(val) and not math.isinf(val)


def validate_spot_price(price: float, label: str = "spot") -> float:
    """
    Validate a spot price value.

    Args:
        price: The price to validate
        label: Label for log messages

    Returns:
        The validated price

    Raises:
        ValidationError: If price is NaN, infinite, or non-positive
    """
    if not _is_valid_float(price):
        raise ValidationError(f"{label} price is invalid (NaN/Inf): {price}")
    if price <= 0:
        raise ValidationError(f"{label} price is non-positive: {price}")
    return price


def validate_quote(quote: Quote, symbol: str = "unknown") -> Quote:
    """
    Validate a Quote, clamping or rejecting invalid fields.

    Checks:
    - bid/ask/last are not NaN or Inf
    - bid <= ask (if both are positive)
    - At least one price (bid, ask, or last) is positive

    Since Quote is frozen, returns a new Quote with sanitized values
    if any field needs correction.

    Args:
        quote: The Quote to validate
        symbol: Symbol for log context

    Returns:
        A validated Quote (may be the same object if already valid)

    Raises:
        ValidationError: If no usable price data exists
    """
    bid = quote.bid
    ask = quote.ask
    last = quote.last

    needs_fix = False

    # Sanitize NaN/Inf
    if not _is_valid_float(bid) or bid < 0:
        logger.debug(f"[{symbol}] Invalid bid={bid}, clamping to 0.0")
        bid = 0.0
        needs_fix = True

    if not _is_valid_float(ask) or ask < 0:
        logger.debug(f"[{symbol}] Invalid ask={ask}, clamping to 0.0")
        ask = 0.0
        needs_fix = True

    if not _is_valid_float(last) or last < 0:
        logger.debug(f"[{symbol}] Invalid last={last}, clamping to 0.0")
        last = 0.0
        needs_fix = True

    # bid > ask with both positive -> swap (likely stale data)
    if bid > 0 and ask > 0 and bid > ask:
        # Use DEBUG for small-value quotes (OTM options with stale/delayed data)
        if bid < 1.00:
            logger.debug(f"[{symbol}] Inverted spread: bid={bid} > ask={ask}, swapping")
        else:
            logger.warning(f"[{symbol}] Inverted spread: bid={bid} > ask={ask}, swapping")
        bid, ask = ask, bid
        needs_fix = True

    # At least one price must be usable
    best_price = max(bid, ask, last)
    if best_price <= 0:
        raise ValidationError(f"[{symbol}] No usable price data: bid={bid}, ask={ask}, last={last}")

    # Fill zero prices from best available
    if bid <= 0 and ask > 0:
        bid = ask
        needs_fix = True
    elif ask <= 0 and bid > 0:
        ask = bid
        needs_fix = True
    elif bid <= 0 and ask <= 0:
        bid = last
        ask = last
        needs_fix = True

    if last <= 0:
        last = (bid + ask) / 2
        needs_fix = True

    if not needs_fix:
        return quote

    # Sanitize sizes
    bid_size = quote.bid_size if _is_valid_float(float(quote.bid_size)) else 0
    ask_size = quote.ask_size if _is_valid_float(float(quote.ask_size)) else 0
    last_size = quote.last_size if _is_valid_float(float(quote.last_size)) else 0

    return Quote(
        bid=bid,
        ask=ask,
        bid_size=bid_size,
        ask_size=ask_size,
        last=last,
        last_size=last_size,
        timestamp=quote.timestamp,
    )


def _safe_greek(val: float, name: str, low: float, high: float, symbol: str = "") -> float:
    """Clamp a Greek to a valid range, replacing NaN/Inf with 0.0."""
    if not _is_valid_float(val):
        logger.debug(f"[{symbol}] {name} is NaN/Inf, defaulting to 0.0")
        return 0.0
    if val < low or val > high:
        clamped = max(low, min(val, high))
        logger.debug(f"[{symbol}] {name}={val:.6f} out of range [{low}, {high}], clamped to {clamped:.6f}")
        return clamped
    return val


def validate_option_quote(oq: OptionQuote) -> OptionQuote:
    """
    Validate an OptionQuote, sanitizing Greeks and prices.

    Checks:
    - bid/ask/last are not NaN (clamp to 0.0)
    - delta is in [-1, 1]
    - gamma >= 0
    - iv is in [0, 5.0]  (500% is extreme but possible)
    - theta and vega are finite

    Since OptionQuote is frozen, returns a new object if corrections are needed.

    Args:
        oq: The OptionQuote to validate

    Returns:
        A validated OptionQuote
    """
    symbol = oq.symbol
    needs_fix = False

    # Validate prices
    bid = oq.bid
    ask = oq.ask
    last = oq.last

    if not _is_valid_float(bid) or bid < 0:
        bid = 0.0
        needs_fix = True
    if not _is_valid_float(ask) or ask < 0:
        ask = 0.0
        needs_fix = True
    if not _is_valid_float(last) or last < 0:
        last = 0.0
        needs_fix = True

    # Validate and clamp Greeks
    delta = _safe_greek(oq.delta, "delta", -1.0, 1.0, symbol)
    gamma = _safe_greek(oq.gamma, "gamma", 0.0, 100.0, symbol)
    vega = _safe_greek(oq.vega, "vega", -100.0, 100.0, symbol)
    theta = _safe_greek(oq.theta, "theta", -100.0, 100.0, symbol)
    iv = _safe_greek(oq.iv, "iv", 0.0, 5.0, symbol)

    # Volume and OI
    volume = oq.volume if _is_valid_float(float(oq.volume)) and oq.volume >= 0 else 0
    open_interest = oq.open_interest if _is_valid_float(float(oq.open_interest)) and oq.open_interest >= 0 else 0

    if (delta != oq.delta or gamma != oq.gamma or vega != oq.vega or
            theta != oq.theta or iv != oq.iv or bid != oq.bid or
            ask != oq.ask or last != oq.last or volume != oq.volume or
            open_interest != oq.open_interest):
        needs_fix = True

    if not needs_fix:
        return oq

    return OptionQuote(
        symbol=oq.symbol,
        underlying=oq.underlying,
        option_type=oq.option_type,
        strike=oq.strike,
        expiry=oq.expiry,
        bid=bid,
        ask=ask,
        last=last,
        volume=volume,
        open_interest=open_interest,
        timestamp=oq.timestamp,
        delta=delta,
        gamma=gamma,
        vega=vega,
        theta=theta,
        iv=iv,
    )
