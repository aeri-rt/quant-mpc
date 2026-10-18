"""backtest/data.py + backtest/iv.py: Cboe parsing/caching, prior-day ATM IV (DDKZ), synthetic bars, IBKR bar cache."""

import asyncio
import math
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from backtest.data import (
    fetch_cboe,
    fetch_spy_bars_ibkr,
    load_bars,
    load_term_structure,
    parse_cboe_csv,
    synthetic_bars,
    synthetic_term_structure,
)
from backtest.iv import TENORS, atm_iv
from src.clock import NY, XNYS, is_session, session_date
from src.config import Backtest, load_live

FIX = Path(__file__).parent / "fixtures"
FIXTURES = {
    "VIX9D": FIX / "VIX9D_sample.csv",
    "VIX": FIX / "vix_sample.csv",
    "VIX3M": FIX / "VIX3M_sample.csv",
}


def _term(rows: dict[str, tuple[float, float, float]]) -> pd.DataFrame:
    return pd.DataFrame.from_dict(
        rows, orient="index", columns=["vix9d", "vix", "vix3m"]
    ).set_index(pd.DatetimeIndex(list(rows)))


def _sigma_var(row, T):
    ts = [t for t, _ in TENORS]
    sig = [row[c] for _, c in TENORS]
    if T <= ts[0]:
        return sig[0]
    if T >= ts[-1]:
        return sig[-1]
    return math.sqrt(np.interp(T, ts, [s * s * t for s, t in zip(sig, ts, strict=True)]) / T)


# ---- Cboe ---------------------------------------------------------------------------------
def test_parse_and_cache(tmp_path, monkeypatch):
    s = parse_cboe_csv(FIXTURES["VIX"])
    assert len(s) == 30 and s.between(0.1, 0.3).all()
    assert isinstance(s.index, pd.DatetimeIndex) and (s.index == s.index.normalize()).all()
    assert s.index.tz is None and s.index.is_monotonic_increasing
    first = fetch_cboe("VIX", tmp_path, 2, url=str(FIXTURES["VIX"]))
    assert (tmp_path / "VIX_v2.parquet").exists() and first.name == "VIX"

    def no_network(*a, **k):
        raise RuntimeError("read_csv called")

    monkeypatch.setattr(pd, "read_csv", no_network)
    pd.testing.assert_series_equal(fetch_cboe("VIX", tmp_path, 2, url=str(FIXTURES["VIX"])), first)
    with pytest.raises(RuntimeError, match="read_csv called"):  # a version bump never reuses v2
        fetch_cboe("VIX", tmp_path, 3, url=str(FIXTURES["VIX"]))


def test_parse_tolerates_lowercase_and_blank_line():
    text = "date,open,high,low,close\n2025-01-02,17,18,16,17.5\n2025-01-03,17,18,16,18.0\n\n"
    s = parse_cboe_csv(text)
    assert s.tolist() == [0.175, 0.18]
    assert s.index[0] == pd.Timestamp("2025-01-02")


def test_load_term_structure(tmp_path):
    for k, p in FIXTURES.items():
        fetch_cboe(k, tmp_path, 2, url=str(p))
    term = load_term_structure(tmp_path, 2)
    assert list(term.columns) == ["vix9d", "vix", "vix3m"] and len(term) == 30
    assert term.index.is_monotonic_increasing
    assert term.loc["2025-01-27", "vix"] == pytest.approx(0.28)
    assert (term.vix9d < term.vix).all() and (term.vix3m > term.vix).all()


# ---- atm_iv -------------------------------------------------------------------------------
def test_c12_prior_day_only():
    term = _term({"2025-01-27": (0.28, 0.28, 0.28), "2025-01-28": (0.18, 0.18, 0.18)})
    T = 30 / 365
    assert atm_iv(term, date(2025, 1, 28), T, 1.0) == pytest.approx(
        0.28 / math.sqrt(1 + 3 * T * 1.0**2), abs=1e-9
    )
    assert atm_iv(term, date(2025, 1, 29), T, 1.0) == pytest.approx(
        0.18 / math.sqrt(1 + 3 * T), abs=1e-9
    )
    with pytest.raises(ValueError, match="no prior-day"):
        atm_iv(term, date(2025, 1, 27), T, 1.0)


def test_total_variance_interpolation():
    term = _term({"2025-01-27": (0.20, 0.16, 0.15)})
    d = date(2025, 1, 28)
    T = 20 / 365
    w = np.interp(T, [9 / 365, 30 / 365], [0.20**2 * 9 / 365, 0.16**2 * 30 / 365])
    hair = math.sqrt(1 + 3 * T)
    got = atm_iv(term, d, T, 1.0)
    assert got == pytest.approx(math.sqrt(w / T) / hair, abs=1e-12)
    assert 0.16 < got * hair < 0.20
    T2 = 120 / 365
    assert atm_iv(term, d, T2, 1.0) == pytest.approx(0.15 / math.sqrt(1 + 3 * T2), abs=1e-12)
    T3 = 2 / 365
    assert atm_iv(term, d, T3, 1.0) == pytest.approx(0.20 / math.sqrt(1 + 3 * T3), abs=1e-12)


@pytest.mark.parametrize("T", [9 / 365, 30 / 365, 60 / 365])
@pytest.mark.parametrize("b", [0.5, 1.0])
def test_ddkz_haircut(T, b):
    term = _term({"2025-01-27": (0.21, 0.17, 0.155)})
    d = date(2025, 1, 28)
    sv = _sigma_var(term.iloc[0], T)
    assert atm_iv(term, d, T, b) == pytest.approx(sv / math.sqrt(1 + 3 * T * b**2), abs=1e-12)
    assert atm_iv(term, d, T, 0.0) == pytest.approx(sv, abs=1e-12)


@pytest.mark.parametrize("b", [0.0, 0.5, 1.0])
def test_synthetic_term_flat_at_every_node(b):
    idx = XNYS.sessions_in_range(date(2025, 1, 2), date(2025, 1, 31))
    term = synthetic_term_structure(idx, 0.15, skew_b=b)
    assert term.loc["2025-01-02", "vix"] == pytest.approx(0.15 * math.sqrt(1 + 3 * 30 / 365 * b**2))
    for d in idx[1:]:
        for days in (9, 30, 91):
            assert atm_iv(term, d.date(), days / 365, b) == pytest.approx(0.15, abs=1e-9)
    # off the nodes: exactly flat only at b = 0 (linear total-variance interpolation); at b > 0
    # the drift over a straddle's life (7-35 DTE) stays below 0.4 vol-points
    off = [abs(atm_iv(term, date(2025, 1, 3), t / 365, b) - 0.15) for t in range(1, 120)]
    assert max(off) < 1e-12 if b == 0 else max(off[6:35]) < 0.004


# ---- synthetic bars -----------------------------------------------------------------------
def _by_day(bars):
    return bars.groupby(bars.ts.dt.date, sort=True)


def test_synthetic_bars_calendar_and_shape():
    bars = synthetic_bars(date(2025, 1, 2), date(2025, 1, 31), 60, 0.16, 42, 600.0)
    days = sorted(set(bars.ts.dt.date))
    sessions = XNYS.sessions_in_range(date(2025, 1, 2), date(2025, 1, 31))
    assert len(days) == len(sessions) and all(is_session(d) for d in days)
    assert not is_session(date(2025, 1, 9)) and date(2025, 1, 9) not in days
    assert str(bars.ts.dt.tz) == "America/New_York"
    assert _by_day(bars).size().eq(390).all()
    assert bars.ts.iloc[0] == datetime(2025, 1, 2, 9, 30, tzinfo=NY)
    assert bars.open.iloc[0] == 600.0
    assert (bars[["mid"]].to_numpy() == bars[["close"]].to_numpy()).all()
    assert (bars.high >= bars[["open", "close"]].max(axis=1)).all()
    for _, g in _by_day(bars):
        assert np.allclose(g.open.to_numpy()[1:], g.close.to_numpy()[:-1])
        assert g.ts.diff().dropna().eq(timedelta(seconds=60)).all()


def test_synthetic_bars_variance_split():
    sigma = 0.16
    bars = synthetic_bars(date(2025, 1, 2), date(2025, 6, 30), 60, sigma, 42, 600.0)
    closes = _by_day(bars).close.last()
    opens = _by_day(bars).open.first()
    assert len(closes) >= 60
    daily = np.log(closes).diff().dropna()
    assert daily.var() == pytest.approx(sigma**2 / 252, rel=0.25)
    gap = np.log(opens.to_numpy()[1:] / closes.to_numpy()[:-1])
    intra = np.log(bars.close / bars.open)
    share = (gap**2).sum() / ((gap**2).sum() + (intra**2).sum())
    assert share == pytest.approx(0.33, abs=0.12)


def test_synthetic_bars_5min_and_early_close():
    bars = synthetic_bars(date(2025, 11, 24), date(2025, 11, 28), 300, 0.16, 42, 600.0)
    n = _by_day(bars).size()
    assert n[date(2025, 11, 24)] == 78 and n[date(2025, 11, 28)] == 42
    assert date(2025, 11, 27) not in n.index  # Thanksgiving
    last = bars[bars.ts.dt.date == date(2025, 11, 28)].ts.iloc[-1]
    assert last == datetime(2025, 11, 28, 12, 55, tzinfo=NY)


# ---- IBKR bar cache -----------------------------------------------------------------------
class _FakeBroker:
    def __init__(self):
        self.calls = []

    async def historical_bars(self, symbol, end, sessions, bar_s, what="MIDPOINT"):
        self.calls.append(end)
        d = session_date(end)
        ts = pd.DatetimeIndex(
            [
                datetime(d.year, d.month, d.day, 9, 30, tzinfo=NY) + timedelta(seconds=bar_s * i)
                for i in range(2)
            ]
        )
        px = [600.0 + len(self.calls), 601.0 + len(self.calls)]
        return pd.DataFrame({"ts": ts, "open": px, "high": px, "low": px, "close": px, "mid": px})


def test_fetch_spy_bars_ibkr_cache(tmp_path, monkeypatch):
    async def no_sleep(*a, **k):
        raise AssertionError("data.py must not sleep; Broker.historical_bars paces itself")

    monkeypatch.setattr(asyncio, "sleep", no_sleep)
    start, end = date(2025, 1, 6), date(2025, 1, 10)  # 4 sessions (01-09 closed)
    b = _FakeBroker()
    bars = asyncio.run(fetch_spy_bars_ibkr(b, start, end, 60, tmp_path, 2))
    root = tmp_path / "SPY_60s_v2"
    assert sorted(p.name for p in root.iterdir()) == [
        f"{d.date()}.parquet" for d in XNYS.sessions_in_range(start, end)
    ]
    assert len(b.calls) == 4 and len(bars) == 8
    assert [session_date(e) for e in b.calls] == [
        d.date() for d in XNYS.sessions_in_range(start, end)
    ]
    assert str(bars.ts.dt.tz) == "America/New_York" and bars.ts.is_monotonic_increasing
    again = asyncio.run(fetch_spy_bars_ibkr(b, start, date(2025, 1, 13), 60, tmp_path, 2))
    assert len(b.calls) == 5 and len(again) == 10  # only the new session was fetched
    pd.testing.assert_frame_equal(again.iloc[:8].reset_index(drop=True), bars)
    assert not (tmp_path / "SPY_60s_v3").exists()


def test_load_bars_dispatch(tmp_path):
    live = load_live("config/live.yaml")
    bt = Backtest(start=date(2025, 1, 2), end=date(2025, 1, 3), cache_dir=str(tmp_path))
    assert len(load_bars(bt, live)) == 780
    bt_ibkr = bt.model_copy(update={"source": "ibkr"})
    with pytest.raises(ValueError, match="no cached bars.*fetch_data.py --ibkr"):
        load_bars(bt_ibkr, live)
