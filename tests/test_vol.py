"""RealizedVol null tests: frequency invariance, overnight share, profile, EWMA/HAR, horizon variance, bad input."""

import math
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from src.clock import NY, XNYS, dt_years
from src.config import Vol
from src.vol import SLOTS, RealizedVol, _rv5

SIGMA = 0.16
DAY_VAR = SIGMA**2 / 252
CFG = Vol()
MON = date(2026, 10, 19)


def sessions(n: int) -> list[date]:
    return [d.date() for d in XNYS.sessions_in_range("2026-01-05", "2026-12-31")[:n]]


def gbm(n_sessions, step_s=30, sigma=SIGMA, overnight_frac=0.0, profile=None, seed=0, S0=600.0):
    """One zero-drift GBM path on XNYS sessions: points at open + k·step_s for k < 390·60/step_s."""
    rng = np.random.default_rng(seed)
    n = 390 * 60 // step_s
    k = np.arange(n)
    w = np.ones(n) if profile is None else np.asarray(profile, float)[k * step_s // 300]
    sd = np.sqrt((1 - overnight_frac) * sigma**2 / 252 / n * w)
    ts, mids, x = [], [], math.log(S0)
    for d in sessions(n_sessions):
        o = XNYS.session_open(pd.Timestamp(d)).to_pydatetime()
        x += rng.normal(0, math.sqrt(overnight_frac * sigma**2 / 252)) if overnight_frac else 0.0
        path = x + np.concatenate([[0.0], np.cumsum(rng.normal(0, 1, n - 1) * sd[:-1])])
        ts += [o + timedelta(seconds=int(i * step_s)) for i in k]
        mids += list(np.exp(path))
        x = path[-1]
    return pd.DataFrame({"ts": ts, "mid": mids})


def feed(df: pd.DataFrame, every: int = 1, cfg: Vol = CFG) -> RealizedVol:
    rv = RealizedVol(cfg)
    for t, m in zip(df.ts[::every], df.mid[::every], strict=True):
        rv.update(t, m)
    return rv


def ewma(xs, lam=CFG.ewma_lambda):
    v = xs[0]
    for x in xs[1:]:
        v = lam * v + (1 - lam) * x
    return v


def test_rv5_strides_by_minute_slot():
    five = {m: math.exp(0.01 * m) for m in range(0, 390, 5)}  # 5-min bars: 77 returns of 0.05
    assert _rv5(five) == pytest.approx(77 * 0.05**2)
    one = {m: math.exp(0.01 * m) for m in range(390)}  # 1-min: each offset has 77 returns of 0.05
    assert _rv5(one) == pytest.approx(77 * 0.05**2)
    del one[5]  # offset 0 gets one 10-min return, the other offsets are untouched
    assert _rv5(one) == pytest.approx((4 * 77 * 0.05**2 + 75 * 0.05**2 + 0.1**2) / 5)
    assert _rv5({0: 1.0}) == 0.0


def test_frequency_invariance_null():
    df = gbm(40)
    sig = [feed(df, every=e).forecast_sigma() for e in (1, 2, 10)]  # 30 s, 1 min, 5 min
    for s in sig:
        assert abs(s / SIGMA - 1) < 0.10
    for a in sig:
        for b in sig:
            assert abs(a / b - 1) < 0.03
    assert feed(df, every=10).sessions() in (39, 40)


def test_overnight_share():
    rv = feed(gbm(100, overnight_frac=0.33))
    daily = rv.daily_rv()
    assert abs(daily.mean() / DAY_VAR - 1) < 0.15
    on = np.mean([rv._overnight[d] for d in daily.index])
    assert abs(on / daily.mean() - 0.33) < 0.1


def test_profile_flat_on_gbm_and_u_shape():
    p = feed(gbm(60)).profile()
    assert p.shape == (SLOTS,) and abs(p.mean() - 1) < 1e-12
    assert np.abs(p - 1).max() < 0.35
    u = np.ones(SLOTS)
    u[:6] = 4
    assert feed(gbm(60, profile=u)).profile()[0] > 2
    assert (
        abs(feed(gbm(40), every=10).profile().mean() - 1) < 1e-12
    )  # 5-min bars: slot 77 never covered


def test_flat_profile_until_20_sessions():
    assert RealizedVol(CFG).profile().tolist() == [1.0] * SLOTS


def test_har_and_ewma_paths(monkeypatch):
    rv = feed(gbm(120, step_s=60))
    assert rv.sessions() == 119
    har = rv.forecast_variance_day()
    assert abs(har / DAY_VAR - 1) < 0.10
    assert rv.forecast_sigma() == pytest.approx(math.sqrt(252 * har))
    monkeypatch.setattr(rv, "cfg", Vol(har_min_sessions=10_000))
    assert rv.forecast_variance_day() == pytest.approx(ewma(rv.daily_rv().to_numpy()))
    assert rv.forecast_variance_day() != har
    short = feed(gbm(50, step_s=60))
    assert short.forecast_variance_day() == pytest.approx(ewma(short.daily_rv().to_numpy()))
    few = feed(gbm(5, step_s=60))
    assert few.sessions() == 4 and few.forecast_variance_day() is None
    few.finalize_session(sessions(5)[-1])
    assert few.forecast_variance_day() is not None


def test_horizon_variance():
    rv = feed(gbm(40))
    sig2 = 252 * rv.forecast_variance_day()
    on = 0.33 * sig2 / 252
    at_close = rv.horizon_variance(datetime(2026, 10, 19, 15, 59, tzinfo=NY), 60, 0.33)
    assert at_close == pytest.approx(on + sig2 * dt_years(60) * rv.profile()[77])
    assert abs(at_close / on - 1) < 0.02
    at_open = rv.horizon_variance(datetime(2026, 10, 19, 9, 30, tzinfo=NY), 60, 0.33)
    slots = [
        rv.slot(datetime(2026, 10, 19, 9, 30, tzinfo=NY) + timedelta(seconds=60 * k))
        for k in range(390)
    ]
    assert at_open == pytest.approx(sig2 * dt_years(60) * rv.profile()[slots].sum() + on)
    assert at_open == pytest.approx(sig2 / 252 * 1.33)  # Σ profile over 390 one-minute steps == 390
    assert rv.slot(datetime(2026, 10, 19, 9, 0, tzinfo=NY)) == 0
    assert rv.slot(datetime(2026, 10, 19, 16, 30, tzinfo=NY)) == 77
    assert (
        RealizedVol(CFG).horizon_variance(datetime(2026, 10, 19, 10, 0, tzinfo=NY), 60, 0.33)
        is None
    )


def test_irregular_input():
    df = gbm(8, step_s=60)
    full, sparse = feed(df), feed(df[np.arange(len(df)) % 7 != 3])  # missing minutes
    assert full.sessions() == sparse.sessions() == 7
    assert abs(sparse.daily_rv().mean() / full.daily_rv().mean() - 1) < 0.2
    rv = RealizedVol(CFG)
    t = datetime(2026, 10, 19, 10, 0, tzinfo=NY)
    rv.update(t, 100.0)
    rv.update(t + timedelta(seconds=20), 101.0)  # same minute: last wins
    for bad in (float("nan"), float("inf"), 0.0, -1.0):
        rv.update(t, bad)
    rv.update(datetime(2026, 10, 17, 12, 0, tzinfo=NY), 100.0)  # Saturday
    rv.update(datetime(2026, 10, 19, 9, 29, tzinfo=NY), 100.0)  # pre-open
    rv.update(datetime(2026, 10, 19, 16, 0, tzinfo=NY), 100.0)  # post-close
    rv.update(
        datetime(2026, 10, 16, 10, 0, tzinfo=NY), 100.0
    )  # older session: out of order, ignored
    assert rv._mids == {MON: {30: 101.0}} and rv._overnight == {MON: 0.0}
    with pytest.raises(ValueError, match="naive"):
        rv.update(datetime(2026, 10, 19, 10, 0), 100.0)  # noqa: DTZ001
    rv.update(datetime(2026, 10, 20, 9, 30, tzinfo=NY), 102.0)  # new session finalizes the old one
    assert rv.sessions() == 0  # one mid: below half-session coverage, not booked
    assert rv._overnight[date(2026, 10, 20)] == pytest.approx(math.log(102 / 101) ** 2)
    fri = date(2026, 11, 27)  # 13:00 early close: post-close prints are not intraday RV
    rv = RealizedVol(CFG)
    rv.update(datetime(2026, 11, 27, 9, 30, tzinfo=NY), 600.0)
    rv.update(datetime(2026, 11, 27, 12, 59, tzinfo=NY), 601.0)
    rv.update(datetime(2026, 11, 27, 13, 30, tzinfo=NY), 603.0)
    rv.update(datetime(2026, 11, 27, 15, 59, tzinfo=NY), 603.0)
    assert max(rv._mids[fri]) == 209
    rv.update(datetime(2026, 11, 30, 9, 30, tzinfo=NY), 602.0)
    assert rv._prev_close_mid == 601.0


def test_thin_session_not_booked():
    df = gbm(80, step_s=60)
    d = sessions(80)[70]
    day = df.ts.map(lambda t: t.date()) == d
    thin = feed(df[~day | (np.arange(len(df)) % 390 == 389)])  # one mid at the close
    clean = feed(df)
    assert thin.sessions() == clean.sessions() - 1 and d not in thin.daily_rv().index
    assert abs(thin.forecast_sigma() / clean.forecast_sigma() - 1) < 0.01


def test_warm_from_bars_equals_updates():
    df = gbm(12, step_s=60)
    a = RealizedVol(CFG)
    a.warm_from_bars(df)
    b = feed(df)
    b.finalize_session(sessions(12)[-1])
    assert a.sessions() == b.sessions() == 12
    pd.testing.assert_series_equal(a.daily_rv(), b.daily_rv())
    np.testing.assert_array_equal(a.profile(), b.profile())
    c = RealizedVol(CFG)
    c.warm_from_bars(df.rename(columns={"mid": "close"}))
    assert c.daily_rv().equals(a.daily_rv())
    partial = RealizedVol(CFG)
    partial.warm_from_bars(df.iloc[:-1])  # last session stops at 15:58: not finalized
    assert partial.sessions() == 11
