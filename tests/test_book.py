"""Book ledger: Greeks with injected clock, average-cost P&L identity, recovery and diff."""

import math
from dataclasses import replace
from datetime import date, datetime, timedelta

import pytest
from hypothesis import given
from hypothesis import strategies as st
from scipy.stats import norm

from src import bs
from src.book import Book, Leg
from src.clock import NY, expiry_at
from src.config import GreeksCfg
from src.types import Fill, IbkrGreeks, Position

NOW = datetime(2025, 6, 2, 10, 0, tzinfo=NY)
R, Q = 0.04, 0.0099
EXP = expiry_at(date(2025, 7, 18))


def stk(exec_id, qty, price, commission=None):
    return Fill(exec_id, "o", 756733, qty, price, NOW, commission)


def opt(exec_id, con_id, qty, price, commission=1.0):
    return Fill(exec_id, "o", con_id, qty, price, NOW, commission)


def pos(con_id, qty, avg_cost, right=None, strike=None, symbol="SPY"):
    if right is None:
        return Position(con_id, symbol, "STK", qty, avg_cost)
    return Position(con_id, symbol, "OPT", qty, avg_cost, right, strike, EXP, 100)


# 1. C3 regression: straddle created at sim date, evaluated the next day through Book.greeks
def test_straddle_greeks_next_day(straddle_book):
    n, S, iv = 5, 600.0, 0.15
    book = straddle_book(n=n, S=S, iv=iv, now=NOW)
    later = NOW + timedelta(days=1)
    K, T = book.legs[0].strike, book.T(later)
    assert K == 600 and 0.07 < T < 0.09 and 28 < book.dte(later) < 30

    g = book.greeks(S, later, R, Q)
    d1 = bs.d1(S, K, T, R, Q, iv)
    gamma_ref = 2 * n * 100 * norm.pdf(d1) / (S * iv * math.sqrt(T))
    assert g.gamma > 0 and abs(g.gamma - gamma_ref) < 0.01 * gamma_ref
    assert g.vega > 0 and g.theta < 0

    # |delta| < 2 shares where the straddle is delta-neutral (d1 = 0)
    S_neutral = K * math.exp(-(R - Q + 0.5 * iv * iv) * T)
    assert abs(book.greeks(S_neutral, later, R, Q).delta) < 2
    # finite difference of delta reproduces gamma
    d_up, d_dn = book.greeks(601, later, R, Q).delta, book.greeks(599, later, R, Q).delta
    assert abs((d_up - d_dn) - 2 * g.gamma) < 0.05 * 2 * g.gamma


# 2. Cash-flow identity under random STK fill sequences
fills_ = st.lists(
    st.tuples(
        st.integers(-300, 300).filter(bool),
        st.floats(500, 700),
        st.floats(0, 1),
    ),
    min_size=1,
    max_size=20,
)


@given(fills=fills_, S_last=st.floats(500, 700))
def test_ledger_cash_flow_identity(fills, S_last):
    b = Book()
    for i, (qty, px, c) in enumerate(fills):
        b.apply_fill(stk(f"e{i}", qty, px, c))
    assert b.shares == sum(q for q, _, _ in fills)
    cash = sum(-q * px for q, px, _ in fills)
    assert abs(b.pnl(S_last) - (cash + b.shares * S_last - sum(c for *_, c in fills))) < 1e-6


# 3. Round trip at equal mids costs exactly the commissions (spread lives in the fill prices)
def test_round_trip():
    b = Book()
    b.apply_fill(stk("a", 100, 600.06, 0.35))
    b.apply_fill(stk("b", -100, 599.94, 0.35))
    assert b.shares == 0 and b.is_flat()
    assert abs(b.realized - (-12.0)) < 1e-9
    assert abs(b.pnl(600) - (-12.70)) < 1e-9


# 4. C5 regression: the hedge is marked to market
def test_hedge_mtm():
    b = Book()
    b.apply_fill(stk("a", 100, 600.0, 0.35))
    assert abs(b.pnl(595) - (-500 - 0.35)) < 1e-9
    assert b.share_avg_cost == 600.0


def test_flip_and_short_cover():
    b = Book()
    b.apply_fill(stk("a", 100, 600.0))
    b.apply_fill(stk("b", -150, 610.0))  # close 100 (+1000), open -50 @ 610
    assert b.shares == -50 and b.share_avg_cost == 610.0 and b.realized == 1000.0
    b.apply_fill(stk("c", 20, 605.0))  # cover 20 of the short (+100)
    assert b.shares == -30 and b.share_avg_cost == 610.0 and b.realized == 1100.0
    assert abs(b.pnl(605) - (1100 + (605 - 610) * -30)) < 1e-9


# 5. Option fills open legs from leg_meta and remove them at qty 0
def test_option_fills_and_straddle():
    meta_c = Leg(1001, "C", 600.0, EXP, 0, iv=0.15, mid=10.5)
    meta_p = Leg(1002, "P", 600.0, EXP, 0, iv=0.16, mid=9.5)
    b = Book()
    b.apply_fill(opt("c1", 1001, 5, 10.0), leg_meta=meta_c)
    assert b.straddle() is None and not b.is_flat()
    b.apply_fill(opt("p1", 1002, 5, 9.0), leg_meta=meta_p)
    call, put = b.straddle()
    assert (call.con_id, put.con_id) == (1001, 1002)
    assert call.avg_cost == 10.0 and call.iv == 0.15 and call.mid == 10.5 and call.qty == 5
    assert abs(b.pnl(600) - ((0.5 + 0.5) * 500 - 2.0)) < 1e-9

    b.apply_fill(opt("c2", 1001, -5, 12.0))  # close the call, leg_meta not needed
    assert b.leg(1001) is None and len(b.legs) == 1 and b.straddle() is None
    assert b.realized == (12.0 - 10.0) * 5 * 100 and b.commissions == 3.0

    # adding to an existing leg blends the average
    b.apply_fill(opt("p2", 1002, 5, 11.0))
    assert put.qty == 10 and put.avg_cost == 10.0


@pytest.mark.parametrize(
    "legs",
    [
        [Leg(1, "C", 600.0, EXP, 5), Leg(2, "C", 600.0, EXP, 5)],  # two calls
        [Leg(1, "C", 600.0, EXP, 5), Leg(2, "P", 605.0, EXP, 5)],  # strangle
        [Leg(1, "C", 600.0, EXP, 5), Leg(2, "P", 600.0, expiry_at(date(2025, 8, 15)), 5)],
        [Leg(1, "C", 600.0, EXP, 5), Leg(2, "P", 600.0, EXP, -5)],  # short put
        [Leg(1, "C", 600.0, EXP, 5)],
    ],
)
def test_not_a_straddle(legs):
    assert Book(legs=legs).straddle() is None


def test_straddle_order_is_call_put():
    put, call = Leg(2, "P", 600.0, EXP, 5), Leg(1, "C", 600.0, EXP, 5)
    assert Book(legs=[put, call]).straddle() == (call, put)


# 6. C6 regression: recovery works whatever order IBKR reports positions in
STK_POS = pos(756733, -37, 600.5)
CALL_POS = pos(1001, 5, 10.0, "C", 600.0)
PUT_POS = pos(1002, 5, 9.0, "P", 600.0)


@pytest.mark.parametrize("order", [(STK_POS, CALL_POS, PUT_POS), (CALL_POS, PUT_POS, STK_POS)])
def test_from_positions_both_orders(order):
    extra = [pos(5, 10, 400.0, symbol="QQQ"), pos(1003, 0, 1.0, "P", 610.0)]
    b = Book.from_positions({p.con_id: p for p in [*order, *extra]}, "SPY")
    assert b.shares == -37 and b.share_avg_cost == 600.5 and len(b.legs) == 2
    call, put = b.straddle()
    assert (call.con_id, put.con_id) == (1001, 1002)
    assert call.avg_cost == 10.0 and call.mult == 100 and call.expiry == EXP
    assert b.diff({p.con_id: p for p in order}, "SPY").ok


def test_from_positions_rejects_unpaired():
    with pytest.raises(ValueError, match="1002"):
        Book.from_positions({1002: PUT_POS}, "SPY")
    four = [CALL_POS, PUT_POS, pos(1003, 5, 8.0, "C", 610.0), pos(1004, 5, 7.0, "P", 610.0)]
    with pytest.raises(ValueError, match="unpaired"):
        Book.from_positions({p.con_id: p for p in four}, "SPY")
    assert Book.from_positions({}, "SPY").is_flat()


# 7. diff
def test_diff():
    b = Book.from_positions({p.con_id: p for p in (STK_POS, CALL_POS, PUT_POS)}, "SPY")
    broker = [
        pos(756733, -40, 600.5),
        pos(1001, 4, 10.0, "C", 600.0),  # qty mismatch
        pos(1003, 5, 8.0, "P", 610.0),  # extra; put 1002 missing
        pos(5, 10, 400.0, symbol="QQQ"),
    ]
    d = b.diff({p.con_id: p for p in broker}, "SPY")
    assert d.share_drift == -3
    assert d.missing_legs == [1002] and d.extra_legs == [1003]
    assert d.qty_mismatch == {1001: (5, 4)}
    assert not d.ok
    d = b.diff({1001: CALL_POS, 1002: PUT_POS}, "SPY")
    assert d.share_drift == 37 and not d.ok and not d.missing_legs


# 8. IBKR source: carry-forward delta, unit switches, staleness
def test_ibkr_source(straddle_book):
    b = straddle_book(now=NOW)
    call, _ = b.straddle()
    call.ibkr = IbkrGreeks(0.52, 0.015, None, None, 0.15, 600.0, NOW)
    g = call.greeks(602, NOW, R, Q, "ibkr", None, 120.0)
    ref = bs.greeks(602, 600, call.T(NOW), R, Q, 0.15, "C").scale(500)
    assert abs(g.delta - (0.52 + 0.015 * 2) * 500) < 1e-9
    assert g.gamma == 0.015 * 500
    assert g.vega == ref.vega and g.theta == ref.theta  # flags None -> BS at ibkr.iv

    call.ibkr = replace(call.ibkr, vega=83.4, theta=-36.5)
    cfg = GreeksCfg(source="ibkr", ibkr_vega_per_volpt=False, ibkr_theta_per_day=False)
    g = call.greeks(602, NOW, R, Q, "ibkr", cfg, 120.0)
    assert abs(g.vega - 0.834 * 500) < 1e-9 and abs(g.theta - (-0.1) * 500) < 1e-9
    cfg = GreeksCfg(source="ibkr", ibkr_vega_per_volpt=True, ibkr_theta_per_day=True)
    g = call.greeks(602, NOW, R, Q, "ibkr", cfg, 120.0)
    assert abs(g.vega - 83.4 * 500) < 1e-9 and abs(g.theta - (-36.5) * 500) < 1e-9

    stale = NOW + timedelta(seconds=200)
    g = call.greeks(602, stale, R, Q, "ibkr", None, 120.0)
    assert g.delta == bs.greeks(602, 600, call.T(stale), R, Q, 0.15, "C").delta * 500
    assert call.greeks(602, NOW, R, Q, "bs", None, 120.0).delta == pytest.approx(
        bs.greeks(602, 600, call.T(NOW), R, Q, 0.15, "C").delta * 500
    )

    # Book sums the ibkr leg with the BS put and the shares
    b.shares = -40
    total = b.greeks(602, NOW, R, Q, source="ibkr", flags=cfg)
    call_g = call.greeks(602, NOW, R, Q, "ibkr", cfg, 120.0)
    put_g = b.legs[1].greeks(602, NOW, R, Q, "ibkr", cfg, 120.0)
    assert abs(total.delta - (call_g.delta + put_g.delta - 40)) < 1e-9
    assert abs(total.vega - (call_g.vega + put_g.vega)) < 1e-9


def test_iv_fallback_to_ibkr_and_no_iv():
    leg = Leg(1, "C", 600.0, EXP, 5, iv=0.0)
    with pytest.raises(ValueError, match="no iv"):
        leg.greeks(600, NOW, R, Q, "bs", None, 120.0)
    leg.ibkr = IbkrGreeks(0.5, 0.01, None, None, 0.2, 600.0, NOW)
    ref = bs.greeks(600, 600, leg.T(NOW), R, Q, 0.2, "C").scale(500)
    assert leg.greeks(600, NOW, R, Q, "bs", None, 120.0) == ref


# 9. Late commission report replaces whatever was booked
def test_set_commission():
    b = Book()
    b.apply_fill(stk("x", 100, 600.0, None))
    assert b.commissions == 0.0
    b.set_commission("x", 0.42)
    assert b.commissions == pytest.approx(0.42)
    b.set_commission("x", 0.40)
    assert b.commissions == pytest.approx(0.40)
    b.apply_fill(stk("y", -100, 600.0, 0.35))
    b.set_commission("y", 0.30)
    assert b.commissions == pytest.approx(0.70)


# 10. Non-finite inputs never leak out as Greeks
@pytest.mark.parametrize("S", [math.nan, math.inf])
def test_non_finite_spot_raises(straddle_book, S):
    with pytest.raises(ValueError):
        straddle_book(now=NOW).greeks(S, NOW, R, Q)


def test_non_finite_iv_raises(straddle_book):
    b = straddle_book(now=NOW)
    b.legs[0].iv = math.nan
    with pytest.raises(ValueError):
        b.greeks(600, NOW, R, Q)
    b.legs[0].ibkr = IbkrGreeks(math.inf, 0.01, None, None, 0.15, 600.0, NOW)
    with pytest.raises(ValueError):
        b.greeks(600, NOW, R, Q, source="ibkr")


def test_empty_book():
    b = Book()
    assert b.is_flat() and b.dte(NOW) == math.inf and b.T(NOW) == 0.0
    assert b.greeks(600, NOW, R, Q) == bs.ZERO and b.pnl(600) == 0.0
    b.shares = 7
    assert b.greeks(600, NOW, R, Q).delta == 7
