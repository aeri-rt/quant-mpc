"""Boundary validation returns the input or None: never a swapped, back-filled or clamped value."""

import math
from datetime import datetime, timedelta

import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

from src.clock import NY, expiry_at
from src.types import Quote
from src.validation import fresh_quote, ibkr_greeks, option_quote, quote_age_s

NOW = datetime(2026, 3, 10, 10, 0, tzinfo=NY)
TS = NOW - timedelta(seconds=1)
NAN, INF = float("nan"), float("inf")
EXP = expiry_at(NOW.date() + timedelta(days=30))


def stock(bid=600.0, ask=600.05, last=600.02, ts=TS, md=1, max_age=15.0, live=True):
    return fresh_quote(bid, ask, last, ts, md, NOW, max_age, live)


def option(bid=5.0, ask=5.2, iv=0.15, delta=0.5, ts=TS, md=1, max_age=15.0):
    return option_quote(1001, "C", 600.0, EXP, bid, ask, iv, delta, ts, md, NOW, max_age)


# --- fresh_quote -------------------------------------------------------------------------------


@given(bid=st.floats(1, 1990), s=st.floats(1e-6, 0.005))
def test_fresh_quote_keeps_inputs_and_is_idempotent(bid, s):
    ask = bid * (1 + s)
    assume(ask > bid)
    q = stock(bid, ask, last=NAN)
    assert q is not None
    assert (q.bid, q.ask) == (bid, ask)  # never a price absent from the input
    assert q.last == q.mid  # last is informational: mid when absent
    assert q.ts == TS
    again = fresh_quote(q.bid, q.ask, q.last, q.ts, q.market_data_type, NOW, 15.0, True)
    assert again == q


def test_fresh_quote_keeps_finite_last():
    assert stock(last=599.9).last == 599.9


@pytest.mark.parametrize(
    "kw",
    [
        pytest.param({"ts": None}, id="ts_none"),
        pytest.param({"ts": TS.replace(tzinfo=None)}, id="ts_naive"),
        pytest.param({"bid": NAN}, id="nan_bid"),
        pytest.param({"ask": INF}, id="inf_ask"),
        pytest.param({"bid": None}, id="none_bid"),
        pytest.param({"bid": 0.0}, id="zero_bid"),
        pytest.param({"bid": -1.0, "ask": 600.0}, id="ib_no_bid_sentinel"),
        pytest.param({"bid": 600.05, "ask": 600.0}, id="crossed_not_swapped"),
        pytest.param({"bid": 600.0, "ask": 600.0}, id="locked"),
        pytest.param({"bid": 600.0, "ask": 606.0}, id="spread_1pct"),
        pytest.param({"ts": NOW - timedelta(seconds=16)}, id="age_16_max_15"),
        pytest.param({"md": 3, "live": True}, id="delayed_with_require_live"),
    ],
)
def test_fresh_quote_none_paths(kw):
    assert stock(**kw) is None


def test_fresh_quote_delayed_without_require_live():
    q = stock(md=3, live=False)
    assert isinstance(q, Quote) and q.market_data_type == 3


def test_fresh_quote_at_max_age_is_fresh():
    assert stock(ts=NOW - timedelta(seconds=15)) is not None


# --- quote_age_s -------------------------------------------------------------------------------


def test_quote_age():
    assert quote_age_s(None, NOW) is None
    assert quote_age_s(NOW - timedelta(seconds=3), NOW) == 3.0
    assert quote_age_s(NOW + timedelta(seconds=3), NOW) == 0.0  # future ts: clamp, don't raise
    assert quote_age_s(NOW.replace(tzinfo=None), NOW) is None


# --- option_quote ------------------------------------------------------------------------------


def test_option_quote_valid():
    q = option()
    assert q is not None
    assert (q.bid, q.ask, q.iv, q.delta, q.ts, q.con_id) == (5.0, 5.2, 0.15, 0.5, TS, 1001)


@pytest.mark.parametrize(
    "kw",
    [
        pytest.param({"bid": 0.0}, id="one_sided"),
        pytest.param({"bid": 5.0, "ask": 5.8}, id="spread_15pct"),
        pytest.param({"bid": 5.2, "ask": 5.0}, id="crossed"),
        pytest.param({"ask": NAN}, id="nan_ask"),
        pytest.param({"ts": None}, id="ts_none"),
        pytest.param({"ts": NOW - timedelta(seconds=16)}, id="stale"),
    ],
)
def test_option_quote_none_paths(kw):
    assert option(**kw) is None


def test_option_quote_locked_is_two_sided():
    assert option(bid=5.0, ask=5.0) is not None


@pytest.mark.parametrize(
    "kw, field",
    [
        ({"iv": -2.0}, "iv"),
        ({"iv": NAN}, "iv"),
        ({"iv": 0.0}, "iv"),
        ({"iv": 5.0}, "iv"),
        ({"delta": 1.5}, "delta"),
        ({"delta": -2.0}, "delta"),
        ({"delta": None}, "delta"),
    ],
)
def test_option_quote_bad_greek_dropped_quote_kept(kw, field):
    q = option(**kw)
    assert q is not None and getattr(q, field) is None


def test_option_quote_greek_edges_kept():
    q = option(delta=-1.0)
    assert q.delta == -1.0
    assert option(delta=1.0).delta == 1.0


# --- ibkr_greeks -------------------------------------------------------------------------------


def greeks(delta=0.52, gamma=0.02, vega=0.8, theta=-0.1, iv=0.15, S=600.0, ts=TS):
    return ibkr_greeks(delta, gamma, vega, theta, iv, S, ts)


def test_ibkr_greeks_valid():
    g = greeks()
    assert g is not None
    assert (g.delta, g.gamma, g.vega, g.theta, g.iv, g.S_ref, g.ts) == (
        0.52,
        0.02,
        0.8,
        -0.1,
        0.15,
        600.0,
        TS,
    )


def test_ibkr_greeks_vega_theta_sentinel():  # C11 regression
    g = greeks(vega=-2.0, theta=-2.0)
    assert g is not None and g.vega is None and g.theta is None
    g = greeks(vega=None, theta=NAN)
    assert g is not None and g.vega is None and g.theta is None


@pytest.mark.parametrize(
    "kw",
    [
        pytest.param({"delta": -2.0}, id="delta_sentinel"),
        pytest.param({"delta": 1.01}, id="delta_out_of_range"),
        pytest.param({"delta": NAN}, id="delta_nan"),
        pytest.param({"gamma": -0.1}, id="gamma_negative"),
        pytest.param({"gamma": 11.0}, id="gamma_huge"),
        pytest.param({"iv": 0.0}, id="iv_zero"),
        pytest.param({"iv": -2.0}, id="iv_sentinel"),
        pytest.param({"iv": INF}, id="iv_inf"),
        pytest.param({"S": 0.0}, id="und_zero"),
        pytest.param({"S": NAN}, id="und_nan"),
        pytest.param({"ts": None}, id="ts_none"),
        pytest.param({"ts": TS.replace(tzinfo=None)}, id="ts_naive"),
    ],
)
def test_ibkr_greeks_none_paths(kw):
    assert greeks(**kw) is None


def test_ibkr_greeks_never_clamps():
    assert greeks(delta=1.0000001) is None
    assert greeks(gamma=10.0) is not None  # closed interval edge
    assert not math.isnan(greeks().gamma)
