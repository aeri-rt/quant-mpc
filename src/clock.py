"""Single time base: tz-aware datetimes, XNYS sessions, and the project's only wall-clock call.

UTC internally, America/New_York for session logic; naive datetimes raise.
T (option pricing) is calendar years; dt (hedging) is trading years (see dt_years).
"""

import math
from datetime import UTC, date, datetime, timedelta
from zoneinfo import ZoneInfo

import exchange_calendars as xc
import pandas as pd

NY = ZoneInfo("America/New_York")
TRADING_SECONDS_PER_YEAR = 252 * 6.5 * 3600  # 5_896_800
XNYS = xc.get_calendar("XNYS")  # created once at import (~0.3 s)


class Clock:
    def now(self) -> datetime:
        return datetime.now(UTC)  # the only wall-clock read in src/


class SimClock(Clock):
    def __init__(self, t: datetime):
        self._t = aware(t)

    def now(self) -> datetime:
        return self._t

    def set(self, t: datetime) -> None:
        self._t = aware(t)

    def advance(self, seconds: float) -> None:
        self._t += timedelta(seconds=seconds)


def aware(t: datetime) -> datetime:
    """Return t in UTC; raise on naive input."""
    if t.tzinfo is None or t.utcoffset() is None:
        raise ValueError("naive datetime")
    return t.astimezone(UTC)


def ny(t: datetime) -> datetime:
    return aware(t).astimezone(NY)


def expiry_at(d: date) -> datetime:
    """Option expiry instant: 16:00 America/New_York of the listed date."""
    return datetime(d.year, d.month, d.day, 16, 0, tzinfo=NY)


def T_years(expiry: datetime, now: datetime) -> float:
    """Calendar years to expiry, floored at 0 (option pricing / theta)."""
    return max(0.0, (aware(expiry) - aware(now)).total_seconds() / (365 * 86400))


def dt_years(seconds: float) -> float:
    """Trading years in `seconds` (hedging dt, RV, QP)."""
    return seconds / TRADING_SECONDS_PER_YEAR


def session_date(now: datetime) -> date:
    return ny(now).date()


def is_session(d: date) -> bool:
    return bool(XNYS.is_session(pd.Timestamp(d)))


def is_open(now: datetime) -> bool:
    """True inside regular hours of a session (honours holidays and 13:00 early closes)."""
    return bool(XNYS.is_open_on_minute(pd.Timestamp(aware(now))))


def session_open(now: datetime) -> datetime | None:
    d = session_date(now)
    return XNYS.session_open(pd.Timestamp(d)).to_pydatetime() if is_session(d) else None


def session_close(now: datetime) -> datetime | None:
    d = session_date(now)
    return XNYS.session_close(pd.Timestamp(d)).to_pydatetime() if is_session(d) else None


def seconds_to_close(now: datetime) -> float:
    """Seconds until today's close while open; math.inf when closed."""
    return (session_close(now) - aware(now)).total_seconds() if is_open(now) else math.inf


def steps_to_close(now: datetime, step_s: float) -> int:
    """Number of `step_s` ticks left in the session (>= 1); 1 when closed."""
    s = seconds_to_close(now)
    return max(1, math.ceil(s / step_s)) if math.isfinite(s) else 1


def next_session_date(d: date) -> date:
    """First session strictly after d (d itself need not be a session)."""
    return XNYS.date_to_session(pd.Timestamp(d) + pd.Timedelta(days=1), "next").date()


def monthly_expiry(y: int, m: int) -> date:
    """Monthly option expiry: the 3rd Friday, or the session before it when that is a holiday."""
    d = date(y, m, 15)
    d += timedelta(days=(4 - d.weekday()) % 7)
    return XNYS.date_to_session(pd.Timestamp(d), "previous").date()


def pick_expiry(
    cands: list[date], today: date, target_dte: float, roll_dte: int, prefer_monthly: bool
) -> date | None:
    """Expiry for a new straddle (live chain and backtest share it): the monthly within +-5 days of
    today + target_dte when prefer_monthly, else the nearest candidate (earlier on a tie); never
    one with fewer than roll_dte + 1 days left. None when no candidate qualifies."""
    ok = sorted(d for d in cands if (d - today).days >= roll_dte + 1)
    if not ok:
        return None
    target = today + timedelta(days=round(target_dte))
    if prefer_monthly:
        for x in (target - timedelta(days=5), target + timedelta(days=5)):  # the <= 2 months
            f = monthly_expiry(x.year, x.month)
            if f in ok and abs((f - target).days) <= 5:
                return f
    return min(ok, key=lambda d: abs((d - target).days))
