"""src/gate.py: edge_per_day and the once-per-session Gate. Greeks are BS straddles ×n×100,
costs are Costs() defaults, thresholds come from the real config/live.yaml via live_cfg."""

import math
from datetime import date, datetime

import pytest

from src.bs import Greeks, greeks
from src.clock import NY, SimClock, next_session_date, session_date
from src.config import Costs
from src.costs import option_cost
from src.gate import Edge, Gate, edge_per_day

C = Costs()
S, K, T, R, Q, IV = 600.0, 600.0, 30 / 365, 0.04, 0.0099, 0.15
DTE, BAND, INTERVAL = 30.0, 24.0, 60.0  # band_trade_shares = (1 − f)·H ≈ 0.5·48
N = 5
STRADDLE = (greeks(S, K, T, R, Q, IV, "C") + greeks(S, K, T, R, Q, IV, "P")).scale(N * 100)


@pytest.fixture
def cfg(live_cfg):
    return live_cfg.gate


def edge(cfg, sigma_f, g=STRADDLE, n=N, sunk=False):
    return edge_per_day(
        g, S, sigma_f, IV, C, BAND, n, DTE, cfg.holding_sessions, cfg.turnover_frac, INTERVAL, sunk
    )


def fake_edge(x):  # the held/exit branch reads only edge_day
    return Edge(x, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, IV, 0.0, 0.0)


def next_session(clock):
    """Move the clock to 10:00 NY of the next session."""
    d = next_session_date(session_date(clock.now()))
    clock.set(datetime(d.year, d.month, d.day, 10, 0, tzinfo=NY))


def test_parity_closes_on_vol_margin(cfg, sim_clock):
    e = edge(cfg, IV)
    assert abs(e.gamma_day + e.theta_tday) < 3  # only the r/q theta terms survive
    assert e.edge_day == pytest.approx(-(e.hedge_cost_day + e.option_cost_day), abs=5)
    assert e.edge_day < 0
    assert 2 < e.hedge_cost_day < 4  # c̄·shares_day·turnover_frac + F·crossings ≈ 3.1 at 2.0
    g = Gate(cfg)
    assert g.evaluate(e, sim_clock.now(), held=False) == "CLOSED"
    assert g.reason == "vol_margin"  # checked before edge_below_min
    assert g.last_edge is e


def test_small_edge_closes_on_edge_below_min(cfg, sim_clock):
    e = edge(cfg, IV + 0.02, g=STRADDLE.scale(0.3 / N), n=1)  # 0.3 contracts: edge ≈ +3
    assert 0 < e.edge_day < cfg.min_edge_usd_per_day
    g = Gate(cfg)
    assert g.evaluate(e, sim_clock.now(), held=False) == "CLOSED"
    assert g.reason == "edge_below_min"


def test_vol_premium_opens(cfg, sim_clock):
    e = edge(cfg, 0.20)
    assert e.edge_day > 150
    g = Gate(cfg)
    assert g.evaluate(e, sim_clock.now(), held=False) == "OPEN"
    assert g.reason == "gate_open"


def test_trading_day_basis(cfg):
    e = edge(cfg, 0.20)
    assert e.theta_tday == pytest.approx(STRADDLE.theta * 365 / 252)
    assert e.gamma_day == pytest.approx(0.5 * STRADDLE.gamma * S**2 * 0.20**2 / 252)
    assert e.edge_day == pytest.approx(
        e.gamma_day + e.theta_tday - e.hedge_cost_day - e.option_cost_day
    )


def test_sunk_option_cost(cfg):
    e, es = edge(cfg, IV), edge(cfg, IV, sunk=True)
    assert es.option_cost_day == 0.0
    assert e.option_cost_day == pytest.approx(2 * option_cost(N, DTE, C) / cfg.holding_sessions)
    assert es.edge_day - e.edge_day == pytest.approx(e.option_cost_day)


def test_exit_hysteresis(cfg, sim_clock):
    assert cfg.exit_confirm_sessions == 2
    g = Gate(cfg)
    for x in (-30, -30):
        state = g.evaluate(fake_edge(x), sim_clock.now(), held=True)
        next_session(sim_clock)
    assert (state, g.reason, g.exit_streak) == ("CLOSED", "exit_confirmed", 2)

    g = Gate(cfg)
    for x in (-30, -10, -30):
        state = g.evaluate(fake_edge(x), sim_clock.now(), held=True)
        next_session(sim_clock)
    assert (state, g.reason, g.exit_streak) == ("OPEN", "holding", 1)


def test_cooldown_after_exit(cfg, sim_clock):
    assert cfg.reentry_cooldown_sessions == 2
    g = Gate(cfg)
    g.evaluate(fake_edge(-30), sim_clock.now(), held=True)
    next_session(sim_clock)
    assert g.evaluate(fake_edge(-30), sim_clock.now(), held=True) == "CLOSED"  # session d
    good = edge(cfg, 0.20)
    next_session(sim_clock)  # d + 1
    assert g.evaluate(good, sim_clock.now(), held=False) == "CLOSED"
    assert g.reason == "cooldown"
    next_session(sim_clock)  # d + 2
    assert g.evaluate(good, sim_clock.now(), held=False) == "OPEN"
    assert g.exit_streak == 0


def test_force_closed(cfg, sim_clock):
    g = Gate(cfg)
    g.state, g.exit_streak = "OPEN", 1
    g.force_closed(sim_clock.now(), "roll_refused")
    # Tue 03-10 + 2 sessions
    assert (g.state, g.reason, g.exit_streak, g.cooldown_until) == (
        "CLOSED",
        "roll_refused",
        0,
        date(2026, 3, 12),
    )


def test_due_once_per_session(cfg):
    clock = SimClock(datetime(2026, 3, 10, 9, 59, tzinfo=NY))
    g = Gate(cfg)
    assert g.due(clock.now()) is False
    clock.advance(60)
    assert g.due(clock.now()) is True
    g.evaluate(edge(cfg, IV), clock.now(), held=False)
    assert g.due(clock.now()) is False
    clock.advance(3600)
    assert g.due(clock.now()) is False
    next_session(clock)
    assert g.due(clock.now()) is True


def test_roll_time_reevaluation_same_session(cfg, sim_clock):
    g = Gate(cfg)
    assert g.evaluate(edge(cfg, 0.20), sim_clock.now(), held=False) == "OPEN"
    sim_clock.advance(90 * 60)  # 11:30, same session: roll-time candidate check needs no flag
    assert g.evaluate(edge(cfg, IV), sim_clock.now(), held=False) == "CLOSED"
    assert g.reason == "vol_margin"


@pytest.mark.parametrize("bad", [math.nan, math.inf])
def test_non_finite_input(cfg, sim_clock, bad):
    e = edge(cfg, bad)
    assert e.edge_day == -math.inf and math.isnan(e.gamma_day)
    assert edge(cfg, 0.2, g=Greeks(0.0, bad, 0.0, 0.0)).edge_day == -math.inf
    g = Gate(cfg)
    assert g.evaluate(e, sim_clock.now(), held=False) == "CLOSED"
    assert g.reason == "edge_below_min"


def test_short_straddle_same_code_path(cfg):
    e = edge(cfg, 0.10, g=STRADDLE.scale(-1))
    assert e.gamma_day < 0 < e.theta_tday
    assert e.gamma_day + e.theta_tday > 0  # realised < implied: the short side's edge
    assert e.shares_day > 0  # hedge flow is |Γ|


def test_leland_a_positive_decreasing(cfg):
    a = [edge(cfg, s).leland_a for s in (0.10, 0.15, 0.20)]
    assert a[0] > a[1] > a[2] > 0
