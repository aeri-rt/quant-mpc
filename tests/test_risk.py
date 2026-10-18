import inspect
import time
from datetime import datetime, timedelta

import pytest
from loguru import logger

import src.risk
from src.clock import NY, SimClock
from src.risk import Risk
from src.types import Account

T10 = datetime(2026, 3, 10, 10, 0, tzinfo=NY)  # Tuesday, regular session


def acct_at(nlv=100_000, dtr=-1, now=T10):
    return Account(
        nlv=nlv,
        excess_liquidity=60_000,
        buying_power=200_000,
        daily_pnl=None,
        day_trades_remaining=dtr,
        ts=now,
    )


@pytest.fixture
def risk(live_cfg):
    return Risk(live_cfg.risk, live_cfg.loop, execute=True)


@pytest.fixture
def acct():
    return acct_at()


def test_loss_persists_three_ticks_then_halts_once(risk, acct, sim_clock):
    logged = []
    sink = logger.add(logged.append, level="CRITICAL")
    try:
        t0 = time.perf_counter()
        for i in range(3):
            risk.on_pnl(-2500, -2600, acct, sim_clock.now())
            assert risk.halted is (i == 2)
        elapsed = time.perf_counter() - t0
        assert elapsed < 0.010, f"{elapsed * 1e3:.1f} ms"
        reason = risk.halt_reason
        risk.on_pnl(-2500, -2600, acct, sim_clock.now())
    finally:
        logger.remove(sink)
    assert risk.halt_reason == reason == "daily loss -2600 < -2000"
    assert len(logged) == 1 and "HALT daily loss" in logged[0]


def test_open_grace(live_cfg, acct):
    clk = SimClock(datetime(2026, 3, 10, 9, 31, tzinfo=NY))
    r = Risk(live_cfg.risk, live_cfg.loop, execute=True)
    for _ in range(3):
        r.on_pnl(-2500, -2600, acct, clk.now())
        clk.advance(60)
    assert not r.halted and r._loss_streak == 3
    clk.set(datetime(2026, 3, 10, 9, 36, tzinfo=NY))
    r = Risk(live_cfg.risk, live_cfg.loop, execute=True)
    for _ in range(3):
        r.on_pnl(-2500, -2600, acct, clk.now())
    assert r.halted


def test_streak_resets_on_recovery(risk, acct):
    for pnl in (-2500, 0, -2500, -2500):
        risk.on_pnl(pnl, pnl, acct, T10)
    assert not risk.halted and risk._loss_streak == 2


def test_broker_none_uses_book_alone_and_both_none_noop(risk, acct):
    for _ in range(3):
        risk.on_pnl(None, -2600, acct, T10)
    assert risk.halted
    r2 = Risk(risk.cfg, risk.loop, execute=True)
    for _ in range(3):
        r2.on_pnl(None, None, acct, T10)
    assert not r2.halted and r2._loss_streak == 0


def test_check_hedge(risk, live_cfg, acct):
    ok = {"acct": acct, "quote_ok": True}
    assert risk.check_hedge(-50, 60, 0, 5, **ok) == (-50, "ok")
    assert risk.check_hedge(10, 60, 0, 5, **ok) == (0, "increases_exposure")
    assert risk.check_hedge(-30, 700, 640, 5, **ok) == (-30, "ok")  # reduces, |shares| > 650
    assert risk.check_hedge(20, -10, 640, 5, **ok) == (0, "book_bound")
    assert risk.check_hedge(-50, 60, 0, 5, acct, quote_ok=False) == (0, "delayed")
    assert risk.check_hedge(-50, 60, 0, 5, None, quote_ok=True) == (0, "no_account")
    assert risk.check_hedge(0, 60, 0, 5, **ok) == (0, "none")
    assert risk.check_hedge(-50, 60, 0, 5, acct_at(20_000), True) == (0, "min_nlv")
    assert risk.check_hedge(-50, 60, 0, 5, acct_at(20_000, dtr=0), True) == (0, "min_nlv")
    assert risk.check_hedge(-50, 60, 0, 5, acct_at(30_000, dtr=0), True) == (-50, "ok")
    assert risk.check_hedge(-50, 60, 0, 5, acct_at(24_000, dtr=0), True) == (0, "min_nlv")
    low = Risk(live_cfg.risk.model_copy(update={"min_nlv": 0}), live_cfg.loop, execute=True)
    assert low.check_hedge(-50, 60, 0, 5, acct_at(24_000, dtr=0), True) == (0, "pdt")
    dry = Risk(live_cfg.risk, live_cfg.loop, execute=False)
    assert dry.check_hedge(-50, 60, 0, 5, None, quote_ok=False) == (-50, "ok")
    risk.halt("test")
    assert risk.check_hedge(-50, 60, 0, 5, **ok) == (0, "halted")


def test_size_contracts(risk, live_cfg, acct):
    assert risk.size_contracts(28.0, acct, None, 10) == (5, "ok")  # 0.15*100k/2800 = 5.36
    assert risk.size_contracts(28.0, acct, 2, 10) == (2, "ok")
    assert risk.size_contracts(28.0, acct, None, 3) == (3, "ok")
    assert risk.size_contracts(28.0, acct_at(20_000), None, 10) == (0, "min_nlv")
    assert risk.size_contracts(0.0, acct, None, 10) == (0, "no_premium")
    assert risk.size_contracts(28.0, None, None, 10) == (0, "no_account")
    low = Risk(live_cfg.risk.model_copy(update={"min_nlv": 0}), live_cfg.loop, execute=True)
    assert low.size_contracts(28.0, acct_at(10_000), None, 10) == (0, "premium_frac")


def test_check_straddle(risk, acct):
    assert risk.check_straddle(5, 28.0, acct, 10_000, 10) == (True, "ok")
    assert risk.check_straddle(5, 28.0, acct, None, 10) == (True, "ok")
    assert risk.check_straddle(5, 28.0, acct, 40_000, 10) == (False, "margin")
    assert risk.check_straddle(11, 28.0, acct, 10_000, 10) == (False, "contracts")
    assert risk.check_straddle(0, 28.0, acct, 10_000, 10) == (False, "contracts")
    assert risk.check_straddle(5, 100.0, acct, 10_000, 10) == (False, "premium_frac")
    assert risk.check_straddle(5, 28.0, acct_at(20_000), 10_000, 10) == (False, "min_nlv")
    assert risk.check_straddle(5, 28.0, None, 10_000, 10) == (False, "no_account")
    risk.halt("test")
    assert risk.check_straddle(5, 28.0, acct, 10_000, 10) == (False, "halted")


def test_reject_storm(risk, live_cfg):
    for i in range(3):
        risk.record_reject(T10 + timedelta(minutes=i))
    assert risk.halted and risk.halt_reason == "reject storm"
    r2 = Risk(live_cfg.risk, live_cfg.loop, execute=True)
    for i in range(3):
        r2.record_reject(T10 + timedelta(minutes=7.5 * i))  # 0, 7.5, 15 min
    assert not r2.halted and len(r2._rejects) == 2


def test_check_stale(risk, sim_clock):
    assert risk.check_stale(5, sim_clock.now()) == "fresh"
    assert risk.check_stale(20, sim_clock.now()) == "stale"
    sim_clock.advance(300)
    assert risk.check_stale(None, sim_clock.now()) == "stale"
    sim_clock.advance(301)
    assert risk.check_stale(20, sim_clock.now()) == "halt"
    assert risk.halted and risk.halt_reason == "stale quote"


def test_stale_recovery_resets_timer(risk, sim_clock):
    risk.check_stale(20, sim_clock.now())
    sim_clock.advance(500)
    assert risk.check_stale(1, sim_clock.now()) == "fresh"
    assert risk._stale_since is None
    sim_clock.advance(500)
    assert risk.check_stale(20, sim_clock.now()) == "stale"
    assert not risk.halted


def test_stale_never_halts_outside_the_session(risk):
    """No SPY ticks after hours or on a holiday is not a fault: no halt, the timer restarts."""
    evening = datetime(2026, 3, 10, 20, 0, tzinfo=NY)
    for m in range(0, 120, 10):  # two hours > stale_halt_s, no quote at all
        assert risk.check_stale(None, evening + timedelta(minutes=m)) == "stale"
    assert not risk.halted and risk._stale_since is None
    holiday = datetime(2026, 4, 3, 11, 0, tzinfo=NY)  # Good Friday: XNYS closed
    assert risk.check_stale(None, holiday) == "stale"
    assert risk.check_stale(None, holiday + timedelta(hours=1)) == "stale" and not risk.halted
    opened = datetime(2026, 3, 11, 9, 30, tzinfo=NY)  # the next open starts a fresh timer
    assert risk.check_stale(None, opened) == "stale"
    assert risk.check_stale(None, opened + timedelta(seconds=599)) == "stale"
    assert risk.check_stale(None, opened + timedelta(seconds=601)) == "halt"


def test_naive_datetime_rejected(risk):
    with pytest.raises(ValueError):
        risk.record_reject(datetime(2026, 3, 10, 10, 0))  # noqa: DTZ001


def test_no_threading():
    assert "threading" not in inspect.getsource(src.risk)
