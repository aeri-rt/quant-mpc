"""Broker-boundary validation: a typed value or None, never a repair.

No bid/ask swap, no back-fill from the close, no stamping of the wall clock, no clamping of a
Greek into range. Anything the broker sends that is not usable as-is becomes None here, and the
caller decides what "no data" means for this tick.
"""

import math
from datetime import datetime

from src.clock import aware
from src.types import IbkrGreeks, OptionQuote, Quote

STOCK_MAX_SPREAD = 0.005  # (ask - bid) / mid
OPTION_MAX_SPREAD = 0.10
IB_NOT_COMPUTED = -2.0  # ib_async leaks this sentinel for vega/theta (wrapper.py)


def _finite(x) -> bool:
    try:
        return math.isfinite(x)
    except TypeError:  # None or not a number
        return False


def _utc(ts: datetime | None) -> datetime | None:
    """ts in UTC; None when missing or naive (a naive broker time is unusable, not fixable)."""
    try:
        return aware(ts) if ts is not None else None
    except ValueError:
        return None


def _ranged(x, lo: float, hi: float, open_ends: bool = False) -> float | None:
    """x when finite and inside [lo, hi] ((lo, hi) when open_ends); otherwise None, never clamped."""
    if not _finite(x):
        return None
    return x if (lo < x < hi if open_ends else lo <= x <= hi) else None


def _computed(x) -> float | None:
    """IBKR vega/theta as sent: None for None, -2 or non-finite. Units are Book's job."""
    return x if _finite(x) and x != IB_NOT_COMPUTED else None


def quote_age_s(ts: datetime | None, now: datetime) -> float | None:
    """Seconds since ts as seen by our clock, floored at 0; None without ts."""
    ts = _utc(ts)
    if ts is None:
        return None
    return max(0.0, (aware(now) - ts).total_seconds())


def fresh_quote(
    bid,
    ask,
    last,
    ts: datetime | None,
    market_data_type: int,
    now: datetime,
    max_age_s: float,
    require_live: bool,
) -> Quote | None:
    """Stock quote usable for a decision, else None. `last` is informational: mid when absent."""
    ts = _utc(ts)
    age = quote_age_s(ts, now)
    if age is None or age > max_age_s or (require_live and market_data_type != 1):
        return None
    if not (_finite(bid) and _finite(ask) and 0 < bid < ask):
        return None
    if (ask - bid) / (0.5 * (bid + ask)) > STOCK_MAX_SPREAD:
        return None
    if not (_finite(last) and last > 0):
        last = 0.5 * (bid + ask)
    return Quote(bid, ask, last, ts, market_data_type)


def option_quote(
    con_id: int,
    right: str,
    strike: float,
    expiry: datetime,
    bid,
    ask,
    iv,
    delta,
    ts: datetime | None,
    market_data_type: int,
    now: datetime,
    max_age_s: float,
) -> OptionQuote | None:
    """Two-sided, fresh option quote, else None. iv/delta drop to None when -2, NaN or out of range."""
    ts = _utc(ts)
    age = quote_age_s(ts, now)
    if age is None or age > max_age_s:
        return None
    if not (_finite(bid) and _finite(ask) and 0 < bid <= ask):
        return None
    if (ask - bid) / (0.5 * (bid + ask)) > OPTION_MAX_SPREAD:
        return None
    return OptionQuote(
        con_id,
        right,
        strike,
        expiry,
        bid,
        ask,
        _ranged(iv, 0, 5, open_ends=True),
        _ranged(delta, -1, 1),
        ts,
        market_data_type,
    )


def ibkr_greeks(delta, gamma, vega, theta, iv, und_price, ts: datetime | None) -> IbkrGreeks | None:
    """modelGreeks worth using, else None. -2 and NaN are "not computed", out of range is garbage."""
    ts = _utc(ts)
    d = _ranged(delta, -1, 1)
    g = _ranged(gamma, 0, 10)
    v = _ranged(iv, 0, 5, open_ends=True)
    if ts is None or d is None or g is None or v is None:
        return None
    if not (_finite(und_price) and und_price > 0):
        return None
    return IbkrGreeks(d, g, _computed(vega), _computed(theta), v, und_price, ts)
