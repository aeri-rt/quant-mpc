"""src/hedge.py: Whalley–Wilmott band, closed-form QP (vs a scipy SLSQP oracle on the full
N-step program), naive baseline, close window, guards."""

import math

import numpy as np
import pytest
from scipy.optimize import minimize

from src import hedge
from src.clock import dt_years
from src.config import Costs, Hedge
from src.costs import per_share_rate
from src.hedge import HedgeState, band_half_width, hedge_all, qp_half_width

C = Costs()
CFG = Hedge()
R = 0.04
CLOSE_S = 300.0
C_BUY = per_share_rate(600.0, 1, C)  # 0.0085
C_SELL = per_share_rate(600.0, -1, C)  # 0.0209
W_REF = 390 * 0.15**2 * dt_years(60) + 0.33 * 0.15**2 / 252  # flat profile, 390 x 60 s, overnight


def _state(delta_net, gamma=15.47, S=600.0, T=30 / 365, seconds_to_close=10_000.0, W_var=W_REF):
    return HedgeState(
        delta_net=delta_net, gamma=gamma, S=S, T=T, seconds_to_close=seconds_to_close, W_var=W_var
    )


def _program_u0(D, q, w, w_T, c_buy, c_sell, u_max, q_max, rng):
    """First action of the full N-step program: SLSQP over x = [p, m] in [0, u_max], objective
    divided by W = sum(w) + w_T, |q + cumsum(p - m)| <= q_max, best feasible of 5 restarts."""
    N = len(w)
    W = w.sum() + w_T
    L = np.tril(np.ones((N, N)))
    J = np.hstack([L, -L])  # d(cumulative trade)/dx
    lin = np.r_[np.full(N, c_buy), np.full(N, c_sell)]

    def f(x):
        Dk = D + J @ x
        return (w @ Dk**2 + w_T * Dk[-1] ** 2 + lin @ x) / W

    def jac(x):
        Dk = D + J @ x
        return (J.T @ (2 * w * Dk) + 2 * w_T * Dk[-1] * J[-1] + lin) / W

    cons = [
        {"type": "ineq", "fun": lambda x: q_max - (q + J @ x), "jac": lambda x: -J},
        {"type": "ineq", "fun": lambda x: q_max + (q + J @ x), "jac": lambda x: J},
    ]
    best = None
    for _ in range(5):
        res = minimize(
            f,
            rng.uniform(0, u_max, 2 * N),
            jac=jac,
            method="SLSQP",
            bounds=[(0, u_max)] * (2 * N),
            constraints=cons,
            options={"ftol": 1e-14, "maxiter": 3000},
        )
        if np.all(np.abs(q + J @ res.x) <= q_max + 1e-6) and (best is None or res.fun < best.fun):
            best = res
    return best.x[0] - best.x[N]


def test_qp_closed_form_equals_program():
    rng = np.random.default_rng(0)
    traded = bound = 0
    for _ in range(40):
        N = int(rng.integers(2, 31))
        S = rng.uniform(400, 800)
        lam = 10 ** rng.uniform(-6.5, -4.5)
        sigma = rng.uniform(0.08, 0.5)
        dt = dt_years(rng.choice([30, 60, 300]))
        prof = rng.uniform(0.3, 3, N)
        prof /= prof.mean()
        overnight = 0.33 if rng.random() < 0.5 else 0.0
        c_buy, c_sell = 0.0085, 0.0085 + 2.06e-5 * S
        if rng.random() < 0.3:
            c_buy, c_sell = c_sell, c_buy
        D = rng.uniform(-150, 150)
        q = int(rng.integers(-900, 901))
        u_max = int(rng.choice([20, 50, 200]))
        q_max = 1000
        w = lam * S**2 * sigma**2 * prof * dt
        w_T = lam * S**2 * overnight * sigma**2 / 252
        W_var = (sigma**2 * prof * dt).sum() + overnight * sigma**2 / 252
        W = lam * S**2 * W_var
        assert W == pytest.approx(w.sum() + w_T, rel=1e-12)
        c_side = c_sell if D > 0 else c_buy
        assert qp_half_width(c_side, S, lam, W_var) == pytest.approx(c_side / (2 * W), rel=1e-12)
        u_cf = -math.copysign(max(0.0, abs(D) - c_side / (2 * W)), D)
        u_cf = min(max(u_cf, -u_max), u_max)
        u_cf = min(max(u_cf, -q_max - q), q_max - q)
        u0 = _program_u0(D, q, w, w_T, c_buy, c_sell, u_max, q_max, rng)
        assert abs(u0 - u_cf) < 1e-3, (N, S, lam, sigma, D, q, u_max, u0, u_cf)
        traded += u_cf != 0
        bound += abs(u_cf) in (u_max, q_max - q, q_max + q)
    assert traded >= 8 and bound >= 1  # the sample covers both sides of the threshold and a bound


def test_qp_single_step_textbook():
    S, lam, sigma, dt = 600.0, 1e-2, 0.2, dt_years(60)
    w = lam * S**2 * sigma**2 * dt
    expected = -max(0.0, 50 - C_SELL / (2 * w))
    assert -50 < expected < 0  # a real trade, not the zero branch
    u0 = _program_u0(
        50.0, 0, np.array([w]), 0.0, C_BUY, C_SELL, 1000, 10_000, np.random.default_rng(1)
    )
    assert u0 == pytest.approx(expected, abs=1e-3)
    assert qp_half_width(C_SELL, S, lam, sigma**2 * dt) == pytest.approx(
        C_SELL / (2 * w), rel=1e-12
    )


def test_band_half_width_scaling():
    base = {
        "S": 600.0,
        "gamma": 15.47,
        "T": 30 / 365,
        "r": R,
        "k": C_BUY / 600,
        "risk_aversion": 1e-4,
    }
    h = band_half_width(**base)
    for key, mult, ratio in [
        ("gamma", 2, 2 ** (2 / 3)),
        ("k", 8, 2),
        ("S", 8, 2),
        ("risk_aversion", 1 / 8, 2),
    ]:
        assert band_half_width(**{**base, key: base[key] * mult}) / h == pytest.approx(
            ratio, abs=1e-9
        )


def test_reference_numbers():
    assert band_half_width(600.0, 15.47, 30 / 365, R, C_BUY / 600, 1e-4) == pytest.approx(
        31.1, abs=0.5
    )
    buy = hedge_all(_state(-10.0), CFG, C, R, CLOSE_S)
    sell = hedge_all(_state(+10.0), CFG, C, R, CLOSE_S)
    assert buy.band_h == pytest.approx(48, abs=2)
    assert W_REF == pytest.approx(1.19e-4, rel=0.01)
    assert buy.qp_h == pytest.approx(33, abs=1)
    assert sell.qp_h == pytest.approx(81, abs=2)


@pytest.mark.parametrize("sign", [1, -1])
def test_band_zero_inside_and_interior_target(sign):
    h = hedge_all(_state(sign * 10.0), CFG, C, R, CLOSE_S).band_h
    inside = hedge_all(_state(sign * h), CFG, C, R, CLOSE_S)
    assert inside.band == 0 and inside.chosen == 0 and inside.reason == "in_band"
    d = sign * (h + 1.0)
    out = hedge_all(_state(d), CFG, C, R, CLOSE_S)
    assert out.reason == "hedge" and out.chosen == out.band != 0
    assert abs(d + out.band - sign * CFG.inner_fraction * h) <= 1


@pytest.mark.parametrize("sign", [1, -1])
def test_qp_edge_target(sign):
    cfg = Hedge(policy="qp")
    d = sign * 150.0
    dec = hedge_all(_state(d), cfg, C, R, CLOSE_S)
    assert abs(d) > dec.qp_h and dec.chosen == dec.qp != 0 and dec.reason == "hedge"
    assert abs(d + dec.qp - sign * dec.qp_h) <= 1
    inside = hedge_all(_state(sign * (dec.qp_h - 1)), cfg, C, R, CLOSE_S)
    assert inside.qp == 0 and inside.reason == "in_band"


def test_naive():
    assert hedge_all(_state(0.9), Hedge(policy="naive"), C, R, CLOSE_S).naive == 0
    dec = hedge_all(_state(-37.4), Hedge(policy="naive"), C, R, CLOSE_S)
    assert dec.naive == dec.chosen == 37 and dec.reason == "hedge"


@pytest.mark.parametrize("d", [37.4, -2.0, 0.3])
def test_close_window(d):
    dec = hedge_all(_state(d, seconds_to_close=200.0), CFG, C, R, CLOSE_S)
    assert (dec.band, dec.qp, dec.naive, dec.chosen) == (-round(d),) * 4
    assert dec.reason == "close_hedge"
    assert hedge_all(_state(d, seconds_to_close=301.0), CFG, C, R, CLOSE_S).reason != "close_hedge"


@pytest.mark.parametrize(
    "bad",
    [
        {"delta_net": math.nan},
        {"gamma": math.inf},
        {"S": 0.0},
        {"S": -1.0},
        {"T": math.nan},
        {"seconds_to_close": math.nan},
    ],
)
def test_nan_inputs(bad):
    dec = hedge_all(_state(**{"delta_net": 80.0, **bad}), CFG, C, R, CLOSE_S)
    assert (dec.band, dec.qp, dec.naive, dec.chosen) == (0, 0, 0, 0)
    assert math.isnan(dec.band_h) and math.isnan(dec.qp_h) and dec.reason == "nan"


@pytest.mark.parametrize("w", [None, 0.0, math.nan, math.inf])
def test_no_vol(w):
    dec = hedge_all(_state(80.0, W_var=w), Hedge(policy="qp"), C, R, CLOSE_S)
    assert dec.qp == dec.chosen == 0 and math.isnan(dec.qp_h) and dec.reason == "no_vol"
    assert dec.band != 0  # the band never needs W_var
    assert hedge_all(_state(80.0, W_var=w), CFG, C, R, CLOSE_S).reason == "hedge"


def test_gamma_zero_is_min_band():
    dec = hedge_all(_state(3.0, gamma=0.0), CFG, C, R, CLOSE_S)
    assert dec.band_h == CFG.band_min_shares and dec.band == 0 and dec.reason == "in_band"


def test_clip_and_exposure_guard(monkeypatch):
    dec = hedge_all(_state(900.0), CFG, C, R, CLOSE_S)
    assert (dec.band, dec.qp, dec.naive) == (-300, -300, -300)
    monkeypatch.setattr(hedge, "_trade", lambda d, h, f: int(abs(d)))  # doubles the exposure
    dec = hedge_all(_state(50.0), CFG, C, R, CLOSE_S)
    assert dec.band == dec.qp == dec.chosen == 0 and dec.reason == "increases_exposure"
    assert dec.naive == -50


def test_qp_h_scales_with_horizon_variance():
    a = hedge_all(_state(-10.0, W_var=W_REF), CFG, C, R, CLOSE_S)
    b = hedge_all(_state(-10.0, W_var=W_REF / 2), CFG, C, R, CLOSE_S)
    assert b.qp_h == pytest.approx(2 * a.qp_h, rel=1e-12)
    assert b.band_h == a.band_h


@pytest.mark.parametrize("d", [7.0, 60.0, 61.5, 123.0, 400.0])
def test_band_sign_symmetry(d):
    c = Costs(sec_fee_rate=0.0)
    p, n = (hedge_all(_state(s * d), CFG, c, R, CLOSE_S) for s in (1, -1))
    assert p.band == -n.band and p.band_h == n.band_h and p.qp == -n.qp and p.reason == n.reason
