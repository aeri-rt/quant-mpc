import math
import re
from datetime import date, datetime, timedelta
from pathlib import Path

import pytest

from src.clock import (
    NY,
    TRADING_SECONDS_PER_YEAR,
    UTC,
    Clock,
    SimClock,
    T_years,
    aware,
    dt_years,
    expiry_at,
    is_open,
    is_session,
    monthly_expiry,
    next_session_date,
    pick_expiry,
    seconds_to_close,
    session_close,
    session_open,
    steps_to_close,
)

SAT = datetime(2026, 10, 17, 12, 0, tzinfo=NY)
MON_1500 = datetime(2026, 10, 19, 15, 0, tzinfo=NY)


def test_expiry_at_is_16_ny():
    assert expiry_at(date(2026, 12, 18)) == datetime(2026, 12, 18, 21, 0, tzinfo=UTC)


def test_thanksgiving_and_early_close():
    assert is_session(date(2026, 11, 26)) is False
    fri = datetime(2026, 11, 27, 10, 0, tzinfo=NY)
    assert session_close(fri) == datetime(2026, 11, 27, 13, 0, tzinfo=NY)
    assert session_open(fri) == datetime(2026, 11, 27, 9, 30, tzinfo=NY)
    assert is_open(datetime(2026, 11, 27, 14, 0, tzinfo=NY)) is False
    assert is_open(datetime(2026, 11, 27, 12, 0, tzinfo=NY)) is True
    assert session_close(SAT) is None and session_open(SAT) is None


def test_T_years():
    exp = datetime(2026, 12, 18, 21, 0, tzinfo=UTC)
    assert T_years(exp, exp) == 0.0
    assert T_years(exp, exp - timedelta(days=365)) == pytest.approx(1.0, abs=1e-6)
    assert T_years(exp, exp + timedelta(days=1)) == 0.0


def test_dt_years():
    assert TRADING_SECONDS_PER_YEAR == 5_896_800
    assert dt_years(60) == 60 / 5_896_800


@pytest.mark.parametrize("fn", [aware, is_open, lambda t: T_years(t, t)])
def test_naive_rejected(fn):
    with pytest.raises(ValueError, match="naive"):
        fn(datetime(2026, 10, 19, 15, 0))  # noqa: DTZ001


def test_seconds_to_close():
    assert seconds_to_close(SAT) is math.inf
    assert seconds_to_close(MON_1500) == pytest.approx(3600)


def test_steps_and_sessions():
    assert steps_to_close(MON_1500, 60) == 60
    assert steps_to_close(SAT, 60) == 1
    assert next_session_date(date(2026, 10, 17)) == date(2026, 10, 19)
    assert next_session_date(date(2026, 11, 25)) == date(2026, 11, 27)


def test_clocks(sim_clock):
    t0 = sim_clock.now()
    sim_clock.advance(60)
    assert sim_clock.now() - t0 == timedelta(seconds=60)
    assert sim_clock.now().tzinfo is UTC
    wall = Clock().now()
    assert wall.tzinfo is UTC and wall.utcoffset() == timedelta(0)
    with pytest.raises(ValueError, match="naive"):
        SimClock(datetime(2026, 3, 10, 10, 0))  # noqa: DTZ001


def test_monthly_expiry_holiday_is_session_before():
    assert monthly_expiry(2026, 11) == date(2026, 11, 20)
    assert monthly_expiry(2025, 4) == date(2025, 4, 17)  # Good Friday Apr 18
    assert monthly_expiry(2026, 6) == date(2026, 6, 18)  # Juneteenth Jun 19


def test_pick_expiry():
    today = date(2026, 10, 18)
    days = [today + timedelta(days=k) for k in (5, 8, 29, 30, 31, 33, 40)]  # +33 = Nov 20 monthly
    assert pick_expiry(days, today, 30, 7, True) == date(2026, 11, 20)  # monthly within 5 d
    assert pick_expiry(days, today, 30, 7, False) == today + timedelta(days=30)  # nearest
    assert pick_expiry(days, today, 40, 7, True) == today + timedelta(days=40)  # monthly 7 d off
    assert pick_expiry(days, today, 1, 7, True) == today + timedelta(days=8)  # >= roll_dte + 1
    assert pick_expiry(days[:1], today, 30, 7, True) is None
    tie = [today + timedelta(days=k) for k in (28, 32)]
    assert pick_expiry(tie, today, 30, 7, False) == today + timedelta(days=28)  # earlier on a tie


def test_only_wall_clock_read_is_clock_py():
    """No wall-clock read outside src/clock.py (ruff DTZ only catches the tz-less ones). The
    pattern is escaped so this file never matches itself; clock.py must match, or it is blind."""
    pat = re.compile(
        r"datetime\.(now|utcnow|today)\(|\btime\.time\(|\bdate\.today\(|Timestamp\.(now|today)\("
    )
    root = Path(__file__).resolve().parents[1]
    hits = {
        p.relative_to(root).as_posix()
        for d in ("src", "backtest", "scripts")
        for p in (root / d).rglob("*.py")
        if any(pat.search(line) for line in p.read_text(encoding="utf-8").splitlines())
    }
    assert hits == {"src/clock.py"}
