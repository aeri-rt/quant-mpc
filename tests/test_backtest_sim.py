"""backtest/sim.py: the Runner driven per bar through SimBroker over a SyntheticMarket.

Every Runner-driven test uses 5-min synthetic bars (78/session), 5 warm-up sessions and 10
ticked ones (~780 ticks per run); the one 1-min run covers 2 ticked sessions. Runs are shared
through module-scoped fixtures so the file stays well under a minute."""

import asyncio
import math
from datetime import date, datetime, time, timedelta

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from backtest.data import synthetic_bars, synthetic_term_structure
from backtest.sim import STOCK_CON_ID, SyntheticMarket, run
from src import bs
from src.book import Book, Leg
from src.clock import NY, XNYS, SimClock, expiry_at
from src.config import Backtest, load_live
from src.costs import commission, reg_fees
from tests.conftest import FakeMarket

approx = pytest.approx
SESSIONS = XNYS.sessions_in_range("2025-01-06", "2025-03-01")
START, END = SESSIONS[0].date(), SESSIONS[14].date()  # 15 sessions: 5 warm-up + 10 ticked
WARM, BAR_S, SIGMA, IV = 5, 300, 0.16, 0.15
HEDGERS = ("band", "qp", "naive")


def _sessions(bars: pd.DataFrame) -> pd.Series:
    return bars.ts.dt.tz_convert(NY).dt.date


def _term(start: date, end: date, iv: float = IV) -> pd.DataFrame:
    return synthetic_term_structure(XNYS.sessions_in_range(start - timedelta(days=10), end), iv)


@pytest.fixture(scope="module")
def live5():
    live = load_live("config/live.yaml")
    return live.model_copy(update={"vol": live.vol.model_copy(update={"har_min_sessions": WARM})})


@pytest.fixture(scope="module")
def bt():
    return Backtest(start=START, end=END, bar_s=BAR_S)


@pytest.fixture(scope="module")
def bars():
    return synthetic_bars(START, END, BAR_S, SIGMA, 42, 600.0)


@pytest.fixture(scope="module")
def term():
    return _term(START, END)


@pytest.fixture(scope="module")
def results(bars, term, live5, bt):
    """All three hedgers on identical bars/term/costs, straddle forced on."""
    return {h: asyncio.run(run(bars, term, live5, bt, h, force_open=True)) for h in HEDGERS}


@pytest.fixture(scope="module")
def rolled(live5):
    """15-min bars over a window with a roll (entry 01-14 into the 02-13 expiry, rolled 02-07).
    initial_nlv puts floor(0.15·NLV / premium) at 8.0 straddles on the roll, so the hedgers'
    ±$1.5k P&L would split them 7/8 if sizing read the hedger's P&L."""
    s = SESSIONS[:23]
    b = synthetic_bars(s[0].date(), s[-1].date(), 900, SIGMA, 42, 600.0)
    bt15 = Backtest(start=s[0].date(), end=s[-1].date(), bar_s=900, initial_nlv=108_000)
    return {
        h: asyncio.run(run(b, _term(START, END), live5, bt15, h, force_open=True)) for h in HEDGERS
    }


@pytest.fixture(scope="module")
def minute(live5):
    """1-min bars, 2 ticked sessions, every open shifted to close_prev + 0.05 so a BUY's fill at
    open_{i+1} + h exceeds the limit (ask_i + 0.01) the runner put on the order."""
    s = SESSIONS[: WARM + 2]
    b = synthetic_bars(s[0].date(), s[-1].date(), 60, SIGMA, 42, 600.0)
    same = _sessions(b) == _sessions(b).shift(1)
    b["open"] = (b.close.shift(1) + 0.05).where(same, b.open)
    bt1 = Backtest(start=s[0].date(), end=s[-1].date(), bar_s=60)
    return b, asyncio.run(run(b, _term(START, END), live5, bt1, "band", force_open=True))


# 1. end to end
def test_end_to_end(results, bars):
    r = results["band"]
    ticked = bars[_sessions(bars) >= SESSIONS[WARM].date()]
    assert len(r.daily) == 10 and len(r.decisions) == len(ticked) == 780
    assert r.decisions.ts.tolist() == ticked.ts.tolist()
    after_first = r.decisions[_sessions(r.decisions) > SESSIONS[WARM].date()]
    assert (after_first.campaign == "held").all()
    assert math.isfinite(r.summary["total_pnl"]) and r.summary["days"] == 10
    assert not r.decisions.reason.fillna("").str.startswith("halt").any()
    assert (
        set(r.fills.con_id) > {STOCK_CON_ID}
        and r.fills.ts.between(ticked.ts.iloc[0], ticked.ts.iloc[-1]).all()
    )


# 2. attribution identity, residual small with and without costs, delta column absorbs the band
def test_attribution_identity(results):
    for r in results.values():
        d = r.daily
        lhs = d[["delta", "gamma", "theta", "vega"]].sum(axis=1) - d.hedge_spread - d.commissions
        assert (lhs + d.residual - d.pnl).abs().max() < 1e-6
        assert d.pnl.sum() == approx(r.summary["total_pnl"], abs=1e-6)
        assert d.residual.abs().sum() < 0.05 * d.theta.abs().sum()
    assert results["band"].daily.delta.abs().sum() > 0
    assert results["naive"].daily.commissions.sum() > results["band"].daily.commissions.sum()


def test_attribution_residual_zero_costs(bars, term, live5, bt):
    zero = live5.model_copy(
        update={
            "costs": live5.costs.model_copy(
                update={
                    "commission_per_share": 0.0,
                    "min_commission": 0.0,
                    "half_spread": 0.0,
                    "sec_fee_rate": 0.0,
                    "finra_taf_per_share": 0.0,
                    "option_commission": 0.0,
                    "option_min_commission": 0.0,
                    "option_fees_per_contract": 0.0,
                    "option_half_spread_by_dte": [(5, 0.0), (14, 0.0), (45, 0.0), (90, 0.0)],
                }
            )
        }
    )
    for h in ("naive", "band"):
        d = asyncio.run(run(bars, term, zero, bt, h, force_open=True)).daily
        assert d.hedge_spread.sum() == 0 and d.commissions.sum() == 0
        assert d.residual.abs().sum() < 0.05 * d.theta.abs().sum()
        assert d.theta.sum() < 0 < d.gamma.sum()
        if h == "band":
            assert d.delta.abs().sum() > 0


# 3. SimBroker + Book: no double-counted spread, exact marks
async def test_c5_pnl_regressions(sim_broker, fake_market, sim_clock, live_cfg):
    c, h, cid = live_cfg.costs, fake_market.h, fake_market.stock_con_id("SPY")
    book = Book()
    await sim_broker.place_shares(100, None, "hedge")
    sim_clock.advance(60)
    sim_broker.quote(cid, sim_clock.now(), False)  # fills at the ask 600.005
    book.apply_fill(sim_broker.fills.popleft())
    fake_market.set_spot(595.0)
    S = sim_broker.quote(cid, sim_clock.now(), False).mid
    assert S == 595.0 and book.shares == 100
    assert book.pnl(S) == approx(-100 * (600.005 - 595.0) - 0.35, abs=1e-9)
    fake_market.set_spot(600.0)
    await sim_broker.place_shares(-100, None, "hedge")
    sim_clock.advance(60)
    sim_broker.quote(cid, sim_clock.now(), False)  # fills at the bid 599.995
    book.apply_fill(sim_broker.fills.popleft())
    expected = -(2 * 100 * h + 2 * commission(100, c) + reg_fees(-100, 600.0 - h, c))
    assert book.shares == 0 and book.pnl(600.0) == approx(expected, abs=1e-9)


# 4. no lookahead: bars on day d price with day d-1 closes; a spike on d shows from d+1
async def test_prior_day_iv(bars, live5, bt):
    term = synthetic_term_structure(XNYS.sessions_in_range("2025-01-06", "2025-01-10"), IV)
    term.loc[pd.Timestamp("2025-01-07")] *= 2  # spike on day d = Jan 7
    clock = SimClock(datetime(2025, 1, 8, 10, 0, tzinfo=NY))
    m = SyntheticMarket(bars, term, live5, bt, clock)
    exp = expiry_at(date(2025, 2, 7))
    at = {dd: datetime(2025, 1, dd, 10, 0, tzinfo=NY) for dd in (7, 8, 10)}
    assert m.iv_for(exp, at[7]) == approx(IV, abs=0.01)  # d: prior day (Jan 6) is unspiked
    assert m.iv_for(exp, at[8]) == approx(2 * IV, abs=0.02)  # d+1 sees the spike
    assert m.iv_for(exp, at[10]) == approx(IV, abs=0.01)  # d+2 back (Jan 9 closed)
    m.set_bar(len(bars) - 1, "close")  # any bar: only S matters for the quote
    (q, *_) = await m.chain("SPY", 30, at[8])
    assert q.iv == approx(m.iv_for(q.expiry, at[8]))
    assert (
        m.leg_quote(q.con_id, at[8]).iv == q.iv and m.leg_greeks(q.con_id, 600.0, at[8]).iv == q.iv
    )
    assert m.leg_quote(999, at[8]) is None and m.leg_greeks(999, 600.0, at[8]) is None


def test_expiry_menu(bars, term, live5, bt):
    """Sim side of clock.pick_expiry: daily sessions, holiday monthly = the session before."""
    m = SyntheticMarket(bars, term, live5, bt, SimClock(datetime(2025, 1, 8, 10, 0, tzinfo=NY)))
    assert m._expiry(date(2025, 3, 14), 35) == date(2025, 4, 17)  # Good Friday Apr 18
    assert m._expiry(date(2025, 1, 8), 30) == date(2025, 2, 7)  # Jan 8 + 30 is a Fri
    assert m._expiry(date(2025, 1, 8), 39) == date(2025, 2, 21)  # target Sun Feb 16, +5 d
    off = live5.model_copy(
        update={"position": live5.position.model_copy(update={"prefer_monthly": False})}
    )
    m = SyntheticMarket(bars, term, off, bt, SimClock(datetime(2025, 1, 8, 10, 0, tzinfo=NY)))
    assert m._expiry(date(2025, 3, 14), 35) == date(2025, 4, 17)  # Apr 18 closed: Thu nearest
    assert m._expiry(date(2025, 1, 8), 39) == date(2025, 2, 14)  # Fri 14 / Tue 18 tie: earlier


# 5. no lookahead: a hedge decided on bar i fills at bar i+1's open +- h, limit or not
def test_fill_timing(minute, live5):
    b, r = minute
    d, f = r.decisions, r.fills
    h = live5.costs.half_spread
    opens = b.set_index("ts").open
    buys = d.index[(d.u_sent.fillna(0) > 0) & (d.reason != "close_hedge")]  # those fill at close
    assert len(buys) >= 2
    for i in buys[:3]:
        fill = f[(f.con_id == STOCK_CON_ID) & (f.ts == d.ts[i + 1])]
        assert len(fill) == 1 and fill.qty.iloc[0] == d.u_sent[i]
        price = fill.price.iloc[0]
        assert price == approx(opens[d.ts[i + 1]] + h, abs=1e-9)
        assert price > d.ask[i] + 0.01  # beyond the marketable limit: fills anyway
    assert (f[f.con_id == STOCK_CON_ID].ts.isin(d.ts)).all()
    first_bar_fills = f[f.ts == d.ts.iloc[0]]
    assert (first_bar_fills.con_id != STOCK_CON_ID).all()  # nothing pending before the first bar


# 6. live/backtest parity: one state through Runner.tick under FakeMarket and SyntheticMarket
async def test_live_and_backtest_identical_decision(runner, sim_clock, live_cfg, bt):
    now = datetime(2026, 3, 10, 10, 30, tzinfo=NY)
    S, n, shares = 600.0, 5, -137
    warm = synthetic_bars(date(2026, 3, 2), date(2026, 3, 9), BAR_S, SIGMA, 1, S)
    term = _term(date(2026, 3, 1), date(2026, 3, 31))
    bar = pd.DataFrame(
        {"ts": pd.DatetimeIndex([now]), "open": S, "high": S, "low": S, "close": S, "mid": S}
    )
    market = SyntheticMarket(bar, term, live_cfg, bt, sim_clock)
    market.set_bar(0, "close")
    chain = await market.chain("SPY", live_cfg.position.entry_dte, now)
    call, put = (next(q for q in chain if q.strike == S and q.right == x) for x in "CP")
    fake = FakeMarket(sim_clock, S=S)
    fake.leg_quotes = {call.con_id: call, put.con_id: put}
    rows = []
    for m in (fake, market):
        r = runner(m)
        await r.start()
        r.book = Book(
            legs=[
                Leg(q.con_id, q.right, q.strike, q.expiry, n, iv=q.iv, mid=q.mid, avg_cost=q.mid)
                for q in (call, put)
            ],
            shares=shares,
            premium_paid=(call.mid + put.mid) * n * 100,
        )
        r.vol.warm_from_bars(warm)
        sim_clock.set(now)
        rows.append(await r.tick())
    a, b = rows
    for k in ("band_u", "qp_u", "naive_u", "band_h", "qp_h", "delta_net", "gamma", "sigma_f"):
        assert a[k] == approx(b[k], abs=1e-9), k
    assert a["band_u"] != 0 and a["qp_u"] != 0 and a["naive_u"] != 0
    assert a["S"] == b["S"] == S and b["u_sent"] == a["u_sent"]


# 7. gate regime on the honest gate (force_open=False)
@pytest.mark.parametrize(("sigma", "opens"), [(0.25, True), (0.15, False)])
def test_gate_regime(live5, bt, term, sigma, opens):
    b = synthetic_bars(START, END, BAR_S, sigma, 7, 600.0)
    r = asyncio.run(run(b, term, live5, bt, "band"))
    d = r.daily
    assert d.gate_open.any() == opens
    if opens:
        assert d.gate_open.iloc[:3].any() and (r.decisions.campaign == "held").any()
        assert r.summary["gate_open_frac"] > 0 and (d.sigma_f > d.sigma_iv + 0.01).any()
    else:
        assert (r.decisions.campaign == "flat").all() and r.fills.empty
        assert r.summary["total_pnl"] == 0 and r.summary["gate_open_frac"] == 0


# 8. Derman-Kamal: discrete hedging error of a long ATM straddle (a test oracle, not a metric)
def derman_kamal(
    n_paths: int, n_hedges: int, sigma: float, S0: float = 600.0, T: float = 30 / 365, seed: int = 0
) -> tuple[float, float]:
    """Long ATM straddle on GBM (zero rates), re-hedged flat at n_hedges equal steps, no costs:
    (mean P&L / premium, std P&L / premium); theory std/premium ~= sqrt(pi / (4 n_hedges))."""
    rng = np.random.default_rng(seed)
    dt = T / n_hedges
    z = rng.standard_normal((n_paths, n_hedges))
    S = S0 * np.exp(np.cumsum(-0.5 * sigma**2 * dt + sigma * math.sqrt(dt) * z, axis=1))
    S = np.concatenate([np.full((n_paths, 1), S0), S], axis=1)
    tau = T - dt * np.arange(n_hedges)
    d1 = (np.log(S[:, :-1] / S0) + 0.5 * sigma**2 * tau) / (sigma * np.sqrt(tau))
    delta = 2 * norm.cdf(d1) - 1  # straddle delta
    prem = bs.straddle_premium(S0, S0, T, 0.0, 0.0, sigma)
    pnl = np.abs(S[:, -1] - S0) - prem - np.sum(delta * np.diff(S, axis=1), axis=1)
    return float(pnl.mean() / prem), float(pnl.std() / prem)


@pytest.mark.parametrize("n_hedges", [23, 140])
def test_derman_kamal(n_hedges):
    mean, std = derman_kamal(200, n_hedges, 0.15)
    theory = math.sqrt(math.pi / (4 * n_hedges))
    assert abs(mean) < 3 * std / math.sqrt(200)
    assert abs(std / theory - 1) < 0.25


# 9. band vs naive vs qp on identical bars: fewer, cheaper trades; identical gate decisions
def test_band_vs_naive(results):
    band, qp, naive = (results[h].summary for h in HEDGERS)
    assert band["trades_per_day"] < 0.5 * naive["trades_per_day"]
    assert band["hedge_cost_per_day"] < naive["hedge_cost_per_day"]
    for s in (band, naive, qp):
        assert "dk_ratio" not in s
        for k, v in s.items():
            assert math.isfinite(v), (k, v)
    assert naive["mean_abs_delta"] < band["mean_abs_delta"]
    gates = [results[h].daily.gate_open.tolist() for h in HEDGERS]
    assert gates[0] == gates[1] == gates[2]
    assert results["band"].daily.gamma.tolist() == approx(results["naive"].daily.gamma.tolist())
    assert results["band"].risk_aversion == 1e-4 and math.isnan(results["naive"].risk_aversion)


# 10. realised turnover vs the gate's assumption (fix live.yaml, not these bounds)
def test_turnover_sanity(results, live5):
    tf = live5.gate.turnover_frac
    band, naive = (results[h].summary["turnover_frac_realised"] for h in ("band", "naive"))
    assert tf / 2 <= band <= 2 * tf
    assert naive >= band


# 11. bar size drives the row count: 390 rows per 1-min session, 78 per 5-min session
def test_minute_bars_row_count(minute, results):
    _, r = minute
    assert r.decisions.groupby(_sessions(r.decisions)).size().tolist() == [390, 390]
    assert len(r.daily) == 2
    five = results["band"].decisions
    assert five.groupby(_sessions(five)).size().eq(78).all()


# 12. account(): sizing NLV is constant (hedger-independent); daily P&L rolls at a session's first call
async def test_account_constant_nlv_and_daily_roll(bars, term, live5, bt):
    d0, d1 = SESSIONS[WARM].date(), SESSIONS[WARM + 1].date()
    clock = SimClock(datetime(d0.year, d0.month, d0.day, 15, 55, tzinfo=NY))
    m = SyntheticMarket(bars, term, live5, bt, clock)
    assert (await m.account()).daily_pnl == 0.0
    m.mark(-5000.0)  # a losing session
    a = await m.account()
    assert (a.nlv, a.daily_pnl) == (bt.initial_nlv, -5000.0)
    clock.set(datetime(d1.year, d1.month, d1.day, 9, 30, tzinfo=NY))
    a = await m.account()  # first tick of the next session, before its mark()
    assert (a.nlv, a.daily_pnl) == (bt.initial_nlv, 0.0)
    m.mark(-5100.0)
    assert (await m.account()).daily_pnl == approx(-100.0)


# 13. identical option book for every hedger, through a roll (sizing never reads hedger P&L)
def test_identical_option_book_across_roll(rolled):
    cols = ["ts", "con_id", "qty"]
    opt = {
        h: r.fills[r.fills.con_id != STOCK_CON_ID][cols].reset_index(drop=True)
        for h, r in rolled.items()
    }
    assert opt["band"].con_id.nunique() >= 4  # the entry straddle and the rolled one
    for h in ("qp", "naive"):
        pd.testing.assert_frame_equal(opt[h], opt["band"])
        assert rolled[h].daily.gamma.tolist() == approx(rolled["band"].daily.gamma.tolist())


# 14. frontier axes: share-only hedge friction and the hedging-error std
def test_frontier_metrics(results, live5):
    h = live5.costs.half_spread
    for r in results.values():
        s, d, f = r.summary, r.daily, r.fills
        stk = f[f.con_id == STOCK_CON_ID]
        assert d.hedge_friction.sum() == approx(
            (h * stk.qty.abs() + stk.commission.fillna(0)).sum()
        )
        assert s["hedge_friction_per_day"] == approx(d.hedge_friction.mean())
        assert s["hedge_friction_per_day"] < s["hedge_cost_per_day"]  # no option round trips
        assert s["hedge_error_std"] == approx((d.delta + d.residual).std(ddof=1))
    band, naive = results["band"].summary, results["naive"].summary
    assert naive["hedge_error_std"] < band["hedge_error_std"]
    assert naive["hedge_friction_per_day"] > band["hedge_friction_per_day"]


# 15. the last-bar fill rule keys on the data: a session whose 15:55 bar is missing still fills
# its last order at that bar's close +- h, not at the next session's open
def test_last_bar_fill_on_gapped_data(live5):
    s = SESSIONS[: WARM + 2]
    b = synthetic_bars(s[0].date(), s[-1].date(), BAR_S, SIGMA, 42, 600.0)
    gap = s[WARM].date()
    b = b.drop(b.index[_sessions(b) == gap][-1]).reset_index(drop=True)
    bt2 = Backtest(start=s[0].date(), end=s[-1].date(), bar_s=BAR_S)
    r = asyncio.run(run(b, _term(START, END), live5, bt2, "naive", force_open=True))
    d, f, h = r.decisions, r.fills, live5.costs.half_spread
    stk = f[f.con_id == STOCK_CON_ID]
    i = d.index[_sessions(d) == gap][-1]
    assert d.ts[i].time() == time(15, 50) and d.u_sent[i] != 0
    px = d.S[i] + math.copysign(h, d.u_sent[i])
    last = stk[(stk.ts == d.ts[i]) & (stk.price - px).abs().lt(1e-9)]
    assert last.qty.tolist() == [d.u_sent[i]]
    assert stk[stk.ts == d.ts[i + 1]].empty  # nothing carried overnight
