"""Black-Scholes identities, unit conventions and a py_vollib cross-check."""

import math

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from scipy.stats import norm

from src.bs import ZERO, Greeks, d1, d2, greeks, implied_vol, price, straddle_premium

S_ = st.floats(50, 2000)
M_ = st.floats(0.7, 1.3)  # K / S
T_ = st.floats(1 / 365, 2)
SIG_ = st.floats(0.05, 1.5)
R_ = st.floats(0, 0.08)
Q_ = st.floats(0, 0.05)
_SGN = {"C": 1.0, "P": -1.0}
prop = settings(deadline=None, max_examples=200)


@prop
@given(S=S_, m=M_, T=T_, sigma=SIG_, r=R_, q=Q_)
def test_put_call_parity(S, m, T, sigma, r, q):
    K = S * m
    c, p = price(S, K, T, r, q, sigma, "C"), price(S, K, T, r, q, sigma, "P")
    assert abs(c - p - (S * math.exp(-q * T) - K * math.exp(-r * T))) < 1e-8 * max(1.0, S)


@prop
@given(S=S_, m=M_, T=T_, sigma=SIG_, r=R_, q=Q_)
def test_call_put_greek_relations(S, m, T, sigma, r, q):
    K = S * m
    gc, gp = greeks(S, K, T, r, q, sigma, "C"), greeks(S, K, T, r, q, sigma, "P")
    assert abs(gc.delta - gp.delta - math.exp(-q * T)) < 1e-9
    assert gc.gamma == gp.gamma
    assert gc.vega == gp.vega >= 0
    assert gc.finite() and gp.finite()


@prop
@given(S=S_, m=M_, T=T_, sigma=SIG_, r=R_, q=Q_, right=st.sampled_from("CP"))
def test_implied_vol_round_trip(S, m, T, sigma, r, q, right):
    K = S * m
    mid = price(S, K, T, r, q, sigma, right)
    intrinsic = max(_SGN[right] * (S * math.exp(-q * T) - K * math.exp(-r * T)), 0.0)
    assume(mid - intrinsic > 1e-4)  # vol is unidentifiable from a ~zero time value
    iv = implied_vol(mid, S, K, T, r, q, right)
    assert iv is not None and abs(iv - sigma) < 1e-6


@prop
@given(S=S_, m=M_, T=T_, sigma=SIG_, r=R_, q=Q_, right=st.sampled_from("CP"))
def test_price_monotone_in_sigma(S, m, T, sigma, r, q, right):
    K = S * m
    assert price(S, K, T, r, q, sigma + 0.01, right) >= price(S, K, T, r, q, sigma, right) - 1e-9


def test_expiry_limits():
    r, q, sigma = 0.045, 0.0099, 0.15
    assert greeks(600, 590, 0, r, q, sigma, "C") == Greeks(1.0, 0.0, 0.0, 0.0)
    assert greeks(600, 610, 0, r, q, sigma, "P") == Greeks(-1.0, 0.0, 0.0, 0.0)
    assert greeks(600, 610, 0, r, q, sigma, "C") == ZERO
    assert price(600, 590, 0, r, q, sigma, "C") == 10
    assert price(600, 590, 0, r, q, sigma, "P") == 0


def test_units_at_reference_point():
    S = K = 600.0
    T, r, q, sigma = 30 / 365, 0.045, 0.0099, 0.15
    a, b = d1(S, K, T, r, q, sigma), d2(S, K, T, r, q, sigma)
    g = greeks(S, K, T, r, q, sigma, "C")
    assert abs(g.vega - S * math.exp(-q * T) * norm.pdf(a) * math.sqrt(T) / 100) < 1e-12
    theta_year = (
        -S * math.exp(-q * T) * norm.pdf(a) * sigma / (2 * math.sqrt(T))
        - r * K * math.exp(-r * T) * norm.cdf(b)
        + q * S * math.exp(-q * T) * norm.cdf(a)
    )
    assert abs(g.theta * 365 - theta_year) < 1e-12
    gp = greeks(S, K, T, r, q, sigma, "P")
    theta_year_p = (
        -S * math.exp(-q * T) * norm.pdf(a) * sigma / (2 * math.sqrt(T))
        + r * K * math.exp(-r * T) * norm.cdf(-b)
        - q * S * math.exp(-q * T) * norm.cdf(-a)
    )
    assert abs(gp.theta * 365 - theta_year_p) < 1e-12
    assert abs(2 * g.gamma - 0.0309) < 2e-4
    assert abs(straddle_premium(S, K, T, r, q, sigma) - 20.5) < 0.5


def test_greeks_algebra():
    g = Greeks(0.5, 0.01, 0.2, -0.05)
    assert g + g == g.scale(2)
    assert sum([g, g, ZERO], ZERO) == g.scale(2)
    assert not Greeks(math.nan, 0, 0, 0).finite()
    assert not Greeks(0, math.inf, 0, 0).finite()


def test_matches_py_vollib():
    pytest.importorskip("py_vollib")
    from py_vollib.black_scholes_merton import black_scholes_merton as bsm
    from py_vollib.black_scholes_merton.greeks.analytical import delta, gamma, theta, vega

    grid = [
        (600, 600, 30 / 365, 0.045, 0.0099, 0.15, "C"),
        (600, 600, 30 / 365, 0.045, 0.0099, 0.15, "P"),
        (600, 580, 7 / 365, 0.05, 0.01, 0.12, "C"),
        (600, 620, 7 / 365, 0.05, 0.01, 0.12, "P"),
        (500, 450, 0.5, 0.03, 0.0, 0.25, "C"),
        (500, 550, 0.5, 0.03, 0.0, 0.25, "P"),
        (100, 100, 1.0, 0.0, 0.0, 0.5, "C"),
        (100, 110, 2.0, 0.08, 0.05, 0.8, "P"),
        (1500, 1200, 1 / 365, 0.04, 0.02, 0.2, "C"),
        (1500, 1800, 0.25, 0.04, 0.02, 1.2, "P"),
    ]
    for S, K, T, r, q, sigma, right in grid:
        f = right.lower()
        g = greeks(S, K, T, r, q, sigma, right)
        assert abs(price(S, K, T, r, q, sigma, right) - bsm(f, S, K, T, r, sigma, q)) < 1e-8
        assert abs(g.delta - delta(f, S, K, T, r, sigma, q)) < 1e-9
        assert abs(g.gamma - gamma(f, S, K, T, r, sigma, q)) < 1e-9
        assert abs(g.vega - vega(f, S, K, T, r, sigma, q)) < 1e-9
        assert abs(g.theta - theta(f, S, K, T, r, sigma, q)) < 1e-9


def test_implied_vol_rejects_out_of_bounds():
    S, K, T, r, q = 600.0, 590.0, 30 / 365, 0.045, 0.0099
    fwd, disc = S * math.exp(-q * T), K * math.exp(-r * T)
    assert implied_vol(fwd - disc - 0.01, S, K, T, r, q, "C") is None  # below intrinsic
    assert implied_vol(fwd + 0.01, S, K, T, r, q, "C") is None  # above S e^{-qT}
    assert implied_vol(disc + 0.01, S, K, T, r, q, "P") is None  # above K e^{-rT}
    assert implied_vol(-1.0, S, K, T, r, q, "P") is None  # below intrinsic (0)
    assert implied_vol(12.0, S, K, 0.0, r, q, "C") is None  # T = 0
    assert implied_vol(math.nan, S, K, T, r, q, "C") is None
    assert implied_vol(math.inf, S, K, T, r, q, "C") is None
    assert implied_vol(20.0, S, K, T, r, q, "C") is not None
