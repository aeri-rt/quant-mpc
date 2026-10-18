"""Backtest = the live Runner driven bar by bar through SimBroker + SimClock over a SyntheticMarket.

There is no second lifecycle: `run()` sets the clock to each bar, lets SimBroker fill last bar's
share orders at this bar's open +- half_spread, then calls `Runner.tick()` against the bar's close.
Options are BS-priced at the prior session's Cboe term structure (`backtest.iv.atm_iv`); the
expiry menu is XNYS sessions with 3rd-Friday monthlies. P&L starts at the first ticked bar; the
warm-up sessions feed RealizedVol only.

Attribution (`attribute`) uses the exposure carried INTO each bar: the post-trade Greeks of the
previous tick (`*_end` columns written by `run()`), so entry/exit/roll bars reconcile exactly.
"""

from __future__ import annotations

import math
import zlib
from dataclasses import dataclass
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

from backtest.iv import atm_iv
from src import bs
from src.clock import NY, XNYS, SimClock, T_years, expiry_at, pick_expiry, session_date
from src.config import Backtest, Live
from src.costs import option_half_spread
from src.log import get_logger
from src.runner import Runner
from src.sim_broker import SimBroker
from src.store import Store
from src.types import Account, IbkrGreeks, OptionQuote, Quote

log = get_logger("backtest")
STOCK_CON_ID = 756733
END_COLS = ["delta_end", "gamma_end", "theta_end", "vega_end", "iv_end"]
DAILY_COLS = [
    "date",
    "delta",
    "gamma",
    "theta",
    "vega",
    "hedge_spread",
    "commissions",
    "hedge_friction",
    "residual",
    "pnl",
    "trades",
    "shares",
    "mean_abs_delta",
    "sigma_f",
    "sigma_iv",
    "gate_open",
]


def _con_id(expiry: date, strike: float, right: str) -> int:
    return zlib.crc32(f"{expiry}|{strike}|{right}".encode()) % 9_990_000 + 10_000


class SyntheticMarket:
    """The `market` for SimBroker in backtests: SPY quotes from bars, BS-priced legs at the
    SimClock with IV from the prior-day term structure, listed daily expiries (monthlies
    preferred). Phase "open" quotes open +- h (fills last bar's orders), "close" quotes
    close +- h (the decision); quote ts = `now` (the clock sits at the bar start)."""

    market_data_type = 1
    connected = data_ok = True
    clock_offset_s = 0.0

    def __init__(
        self, bars: pd.DataFrame, term: pd.DataFrame, live: Live, bt: Backtest, clock: SimClock
    ):
        self.bars, self.term, self.live, self.bt, self.clock = bars, term, live, bt, clock
        self._open, self._close = bars.open.to_numpy(float), bars.close.to_numpy(float)
        self.i: int | None = None
        self.phase = "close"
        self.needs_reconcile = False
        self.book_pnl = self.day_start_pnl = 0.0
        self._day: date | None = None
        self._legs: dict[int, tuple[datetime, float, str]] = {}  # con_id -> (expiry, K, right)
        self._iv_memo: tuple[tuple | None, float] = (None, 0.0)
        self._expiry_memo: dict[tuple[date, float], date] = {}

    def set_bar(self, i: int, phase: str) -> None:
        self.i, self.phase = i, phase

    def _S(self) -> float:
        return float((self._open if self.phase == "open" else self._close)[self.i])

    def stock_con_id(self, symbol: str) -> int:
        return STOCK_CON_ID

    def quote(self, con_id: int, now: datetime, require_live: bool) -> Quote | None:
        if con_id != STOCK_CON_ID or self.i is None:
            return None
        S, h = self._S(), self.live.costs.half_spread
        return Quote(S - h, S + h, S, now, 1)

    def iv_for(self, expiry: datetime, now: datetime) -> float:
        """ATM IV from the PRIOR session's term structure (no lookahead), DDKZ-haircut at this T."""
        if self._iv_memo[0] != (expiry, now):
            iv = atm_iv(self.term, session_date(now), T_years(expiry, now), self.bt.skew_b)
            self._iv_memo = ((expiry, now), iv)
        return self._iv_memo[1]

    def _oq(self, con_id, expiry, K, right, S, now) -> OptionQuote:
        ins = self.live.instrument
        T, iv = T_years(expiry, now), self.iv_for(expiry, now)
        mid = bs.price(S, K, T, ins.risk_free_rate, ins.dividend_yield, iv, right)
        h = option_half_spread(T * 365, self.live.costs)
        delta = bs.greeks(S, K, T, ins.risk_free_rate, ins.dividend_yield, iv, right).delta
        return OptionQuote(con_id, right, K, expiry, max(mid - h, 0.0), mid + h, iv, delta, now, 1)

    def leg_quote(self, con_id: int, now: datetime) -> OptionQuote | None:
        meta = self._legs.get(con_id)
        if meta is None or self.i is None:
            return None
        return self._oq(con_id, *meta, self._S(), now)

    def leg_greeks(self, con_id: int, S: float, now: datetime) -> IbkrGreeks | None:
        """BS at the prior-day IV, so the runner stores leg.iv each bar (vega P&L flows)."""
        meta = self._legs.get(con_id)
        if meta is None:
            return None
        expiry, K, right = meta
        ins, iv = self.live.instrument, self.iv_for(expiry, now)
        g = bs.greeks(S, K, T_years(expiry, now), ins.risk_free_rate, ins.dividend_yield, iv, right)
        return IbkrGreeks(g.delta, g.gamma, None, None, iv, S, now)

    def _expiry(self, d: date, target_dte: float) -> date:
        """Listed expiry over a daily (every-session) menu: clock.pick_expiry, as the live chain."""
        key = (d, target_dte)
        if key not in self._expiry_memo:
            p = self.live.position
            end = d + timedelta(days=round(target_dte) + 10)
            cands = [s.date() for s in XNYS.sessions_in_range(d, end)]
            pick = pick_expiry(cands, d, target_dte, p.roll_dte, p.prefer_monthly)
            if pick is None:  # unreachable while roll_dte < entry_dte (config cross-check)
                raise ValueError(f"no session expiry with dte >= {p.roll_dte + 1}")
            self._expiry_memo[key] = pick
        return self._expiry_memo[key]

    async def chain(self, symbol: str, target_dte: float, now: datetime) -> list[OptionQuote]:
        S = self._S()
        expiry = expiry_at(self._expiry(session_date(now), target_dte))
        out = []
        for k in range(-3, 4):
            K = float(round(S) + k)
            for right in ("C", "P"):
                cid = _con_id(expiry.date(), K, right)
                self._legs[cid] = (expiry, K, right)
                out.append(self._oq(cid, expiry, K, right, S, now))
        return out

    async def account(self) -> Account:
        """NLV is the constant initial_nlv, so contract counts and limits never depend on the
        hedger's P&L path; daily_pnl is the book's since the session's first call (the runner
        calls this every tick, before `mark`)."""
        d = session_date(self.clock.now())
        if d != self._day:
            self._day, self.day_start_pnl = d, self.book_pnl
        nlv, daily = self.bt.initial_nlv, self.book_pnl - self.day_start_pnl
        return Account(nlv, 0.6 * nlv, 2 * nlv, daily, -1, self.clock.now())

    async def positions(self) -> dict:
        return {}  # SimBroker merges its own ledger

    def mark(self, book_pnl: float) -> None:
        self.book_pnl = book_pnl

    async def connect(self) -> None:
        pass

    async def disconnect(self) -> None:
        pass

    async def stream_stock(self, symbol: str) -> int:
        return STOCK_CON_ID

    async def stream_leg(self, con_id: int) -> None:
        pass

    async def unstream(self, con_id: int) -> None:
        pass

    async def historical_bars(self, *a, **k) -> pd.DataFrame:
        return pd.DataFrame()


@dataclass
class Result:
    hedger: str
    risk_aversion: float
    decisions: pd.DataFrame  # one row per ticked bar: store.decisions columns + pnl + *_end
    fills: pd.DataFrame  # ts, order_id, con_id, qty, price, commission, source, mid
    daily: pd.DataFrame  # DAILY_COLS, one row per ticked session
    summary: dict


async def run(
    bars: pd.DataFrame,
    term: pd.DataFrame,
    live: Live,
    bt: Backtest,
    hedger: str,
    risk_aversion: float | None = None,
    force_open: bool = False,
) -> Result:
    """Drive Runner.tick once per bar after a vol warm-up; see the module docstring."""
    knob = {"band": "band_risk_aversion", "qp": "qp_risk_aversion"}.get(hedger)
    h = {"policy": hedger}
    if risk_aversion is not None and knob:
        h[knob] = risk_aversion
    if live.loop.interval_s != bt.bar_s:
        log.warning(f"loop.interval_s {live.loop.interval_s} -> bar_s {bt.bar_s} for the backtest")
    cfg = live.model_copy(
        update={
            "hedge": live.hedge.model_copy(update=h),
            "loop": live.loop.model_copy(update={"interval_s": float(bt.bar_s)}),
        }
    )
    bars = bars.sort_values("ts", ignore_index=True)
    days = bars.ts.dt.tz_convert(NY).dt.date
    sessions = days.unique()
    n_warm = cfg.vol.har_min_sessions
    if len(sessions) <= n_warm:
        raise ValueError(f"{len(sessions)} sessions of bars: need {n_warm} warm-up + 1 to tick")
    warm, ticked = bars[days < sessions[n_warm]], bars[days >= sessions[n_warm]]
    clock = SimClock(ticked.ts.iloc[0].to_pydatetime())
    market = SyntheticMarket(bars, term, cfg, bt, clock)
    broker = SimBroker(market, cfg.costs, clock)
    store = Store(":memory:")
    runner = Runner(
        cfg, broker, store, clock, execute=False, force_open=force_open, mode="backtest"
    )
    if not warm.empty:
        runner.vol.warm_from_bars(warm)
    await runner.start()
    r, qy = cfg.instrument.risk_free_rate, cfg.instrument.dividend_yield
    last_bar = (days != days.shift(-1)).to_numpy()  # last bar of its session present in the data
    pnl, end = [], []
    for i in ticked.index:
        ts = bars.ts[i].to_pydatetime()
        clock.set(ts)
        market.set_bar(i, "open")
        broker.quote(STOCK_CON_ID, ts, False)  # last bar's share orders fill at open +- h
        market.set_bar(i, "close")
        await runner.tick()
        if last_bar[i]:  # a DAY order placed on a session's last bar fills before the bell,
            broker.quote(STOCK_CON_ID, ts, False)  # at close +- h, not at tomorrow's open
            runner._drain()  # so the bar's pnl/*_end carry the fill (exact attribution)
        S, book = float(bars.close[i]), runner.book
        p = book.pnl(S)
        market.mark(p)
        pnl.append(p)
        G = book.greeks(S, ts, r, qy)  # post-trade: the exposure carried into the next bar
        iv = sum(leg.iv for leg in book.legs) / len(book.legs) if book.legs else math.nan
        end.append((G.delta, G.gamma, G.theta, G.vega, iv))
    sid = runner.session_id
    dec = pd.read_sql_query(
        "SELECT * FROM decisions WHERE session_id=? ORDER BY id", store._db, params=[sid]
    ).drop(columns=["id", "session_id"])
    fl = pd.read_sql_query(
        "SELECT ts, order_id, con_id, qty, price, commission, source FROM fills"
        " WHERE session_id=? ORDER BY rowid",
        store._db,
        params=[sid],
    )
    await runner.shutdown()
    store.close()
    if runner.risk.halted:
        log.warning(f"backtest halted: {runner.risk.halt_reason}")
    if len(dec) != len(ticked):
        raise RuntimeError(f"{len(dec)} decision rows for {len(ticked)} bars")
    dec["ts"] = pd.to_datetime(dec.ts, utc=True).dt.tz_convert(NY)
    dec["pnl"] = pnl
    dec[END_COLS] = pd.DataFrame(end, columns=END_COLS)
    fl["ts"] = pd.to_datetime(fl.ts, utc=True).dt.tz_convert(NY)
    mids, hs = [], cfg.costs.half_spread
    for ts, cid, px, q in zip(fl.ts, fl.con_id, fl.price, fl.qty, strict=True):
        if cid == STOCK_CON_ID:
            mids.append(px - math.copysign(hs, q))
        else:  # the leg mid the fill was marked against: BS at that bar's close
            market.set_bar(int(bars.ts.searchsorted(ts)), "close")
            mids.append(market.leg_quote(int(cid), ts.to_pydatetime()).mid)
    fl["mid"] = pd.Series(mids, dtype=float)
    daily = attribute(dec, fl, cfg)
    ra = {"band": cfg.hedge.band_risk_aversion, "qp": cfg.hedge.qp_risk_aversion}
    return Result(hedger, ra.get(hedger, math.nan), dec, fl, daily, _summary(dec, daily))


def attribute(decisions: pd.DataFrame, fills: pd.DataFrame, live: Live) -> pd.DataFrame:
    """Per-session P&L attribution from per-bar terms (exposure carried into the bar):
    delta = delta_net·dS (what the hedger left unhedged) + the bar's share fills marked to the
    close; gamma = ½Γ dS²; theta = mean(θ_day at both ends of the bar)·Δt_days (the trapezoid
    halves the residual over weekend gaps); vega = ν·ΔIV·100; hedge_spread = half-spread of share
    fills + option spread actually paid; residual = Δpnl − (sum − costs); hedge_friction = the
    share fills' half-spread + commissions only (no option round trips: the hedger's own cost)."""
    d = decisions.reset_index(drop=True)
    S = d.S.astype(float)
    dS = S.diff()
    dt_days = d.ts.diff().dt.total_seconds() / 86400
    g = d[END_COLS].astype(float).shift(1)
    theta_mid = 0.5 * (g.theta_end + d.theta_day.astype(float).fillna(g.theta_end))
    f = fills
    bar = np.searchsorted(d.ts.astype("int64"), f.ts.astype("int64"), side="right") - 1
    stk = (f.con_id == STOCK_CON_ID).to_numpy()
    h, q, px = live.costs.half_spread, f.qty.to_numpy(float), f.price.to_numpy(float)
    mid = f.mid.to_numpy(float) if "mid" in f else px
    comm = f.commission.fillna(0).to_numpy(float)
    per = (
        pd.DataFrame(
            {
                "bar": bar,
                "spread": np.where(stk, h * np.abs(q), (px - mid) * q * 100),
                "commissions": comm,
                "hedge_friction": np.where(stk, h * np.abs(q) + comm, 0.0),
                "carry": np.where(stk, q * (S.to_numpy()[bar] - px) + h * np.abs(q), 0.0),
                "shares": np.where(stk, np.abs(q), 0.0),
                "trades": np.ones(len(f)),
            }
        )
        .groupby("bar")
        .sum()
        .reindex(d.index, fill_value=0.0)
    )
    day = d.ts.dt.tz_convert(NY).dt.date
    out = pd.DataFrame(
        {
            "date": day,
            "delta": (g.delta_end * dS).fillna(0.0) + per.carry,
            "gamma": (0.5 * g.gamma_end * dS**2).fillna(0.0),
            "theta": (theta_mid * dt_days).fillna(0.0),
            "vega": (g.vega_end * (d.iv_atm.astype(float) - g.iv_end) * 100).fillna(0.0),
            "hedge_spread": per.spread,
            "commissions": per.commissions,
            "hedge_friction": per.hedge_friction,
            "pnl": d.pnl.diff().fillna(d.pnl),
            "trades": per.trades,
            "shares": per.shares,
        }
    )
    out["residual"] = out.pnl - (
        out.delta + out.gamma + out.theta + out.vega - out.hedge_spread - out.commissions
    )
    daily = out.groupby("date", sort=True).sum()
    held = d.gamma.astype(float).fillna(0.0) != 0
    daily["mean_abs_delta"] = d.delta_net.astype(float).abs()[held].groupby(day[held]).mean()
    daily["sigma_f"] = d.sigma_f.astype(float).groupby(day).last()
    daily["sigma_iv"] = d.iv_atm.astype(float).groupby(day).last()
    daily["gate_open"] = (d.gate_state == "OPEN").groupby(day).any()
    return daily.reset_index()[DAILY_COLS]


def _turnover(d: pd.DataFrame, daily: pd.DataFrame) -> float:
    """Σ realised shares ÷ Σ |Γ|·S·σ_f·√(2/(π·252)) over held sessions (Γ, S, σ_f at each
    session's first held row). A ratio of sums: a roll day's dying Γ ≈ 0 cannot dominate."""
    held = d[d.gamma.astype(float).fillna(0.0) != 0]
    if held.empty:
        return math.nan
    first = held.groupby(held.ts.dt.tz_convert(NY).dt.date).first()
    expected = (
        first.gamma.astype(float).abs()
        * first.S.astype(float)
        * first.sigma_f.astype(float)
        * math.sqrt(2 / (math.pi * 252))
    )
    shares = daily.set_index("date").shares.reindex(first.index)
    return float(shares.sum() / expected.sum())


def _summary(d: pd.DataFrame, daily: pd.DataFrame) -> dict:
    """Frontier axes: `hedge_friction_per_day` and `hedge_error_std` = std of the daily
    hedging-error P&L (delta: unhedged Δ·dS + fill carry, plus residual). daily_std and
    hedge_cost_per_day also carry the book's Γ/θ/ν P&L and option round trips, common to every
    hedger."""
    p = daily.pnl
    std = float(p.std(ddof=1)) if len(p) > 1 else math.nan
    err = daily.delta + daily.residual
    cost = daily.hedge_spread + daily.commissions
    return {
        "total_pnl": float(d.pnl.iloc[-1]),
        "days": int(len(daily)),
        "daily_std": std,
        "sharpe": math.sqrt(252) * float(p.mean()) / std if std > 0 else math.nan,
        "max_drawdown": float((d.pnl.cummax().clip(lower=0.0) - d.pnl).max()),
        "trades_per_day": float(daily.trades.mean()),
        "hedge_cost_per_day": float(cost.mean()),
        "hedge_friction_per_day": float(daily.hedge_friction.mean()),
        "hedge_error_std": float(err.std(ddof=1)) if len(err) > 1 else math.nan,
        "mean_abs_delta": float(daily.mean_abs_delta.mean()),
        "turnover_frac_realised": _turnover(d, daily),
        "gamma_pnl": float(daily.gamma.sum()),
        "theta_pnl": float(daily.theta.sum()),
        "gate_open_frac": float(daily.gate_open.mean()),
    }
