"""Realized volatility from 1-minute mids: daily RV, diurnal profile, EWMA / log-HAR forecast.

Daily variance = overnight log-return² (last mid of the previous session → first mid of this one)
+ 5-min subsampled realized variance of the session's mids, so it is comparable to σ_iv²/252.
dt is implied by the minute-of-session grid, never assumed: 30-s ticks keep the last mid per
minute, 5-min bars populate every fifth minute, and all three feeds give the same σ_f.
Ex-dividend overnight returns are kept (a few bp² against ~1e-4 of daily variance).
"""

import math
from datetime import date, datetime, timedelta
from functools import lru_cache

import numpy as np
import pandas as pd

from src.clock import XNYS, dt_years, is_session, ny, steps_to_close
from src.config import Vol

SLOTS = 78  # 5-minute slots in a 6.5 h session
MINUTES = 5 * SLOTS


@lru_cache(maxsize=1024)
def _session(d: date) -> tuple[datetime, int] | None:
    """(open, minutes in the session) of XNYS session d; None when d is not a session."""
    if not is_session(d):
        return None
    ts = pd.Timestamp(d)
    o, c = XNYS.session_open(ts).to_pydatetime(), XNYS.session_close(ts).to_pydatetime()
    return o, int((c - o).total_seconds() // 60)


def _rv5(mids: dict[int, float]) -> float:
    """5-min subsampled realized variance of mids keyed by minute offset from the open.

    Offset o takes the minutes ≡ o (mod 5) — strided by minute slot, never by list position — so a
    missing minute only lengthens one return and 5-min bars (0, 5, 10, …) reduce to the plain
    5-min Σ r². Mean over the offsets with ≥ 2 points.
    """
    ms = sorted(mids)
    sums = []
    for o in range(5):
        pts = [mids[m] for m in ms if m % 5 == o]
        if len(pts) >= 2:
            r = np.diff(np.log(pts))
            sums.append(float(r @ r))
    return float(np.mean(sums)) if sums else 0.0


def _slot_r2(mids: dict[int, float]) -> np.ndarray:
    """Squared return per 5-min slot: Σ r² of consecutive mids in [5s, 5s+5], scaled to 5 min; NaN if < 2 mids."""
    ms = np.fromiter(sorted(mids), int)
    lp = np.log([mids[m] for m in ms])
    out = np.full(SLOTS, np.nan)
    for s in range(SLOTS):
        i = (ms >= 5 * s) & (ms <= 5 * s + 5)
        if i.sum() >= 2:
            out[s] = np.sum(np.diff(lp[i]) ** 2) * 5 / (ms[i][-1] - ms[i][0])
    return out


def _lmean(a: np.ndarray) -> float:
    return math.log(max(float(a.mean()), 1e-12))


class RealizedVol:
    def __init__(self, cfg: Vol):
        self.cfg = cfg
        self._mids: dict[date, dict[int, float]] = {}  # session → {minute from open: last mid}
        self._cur: date | None = None
        self._prev_close_mid: float | None = None
        self._daily: dict[date, float] = {}  # completed sessions' variance
        self._overnight: dict[date, float] = {}
        self._slot_r2: dict[date, np.ndarray] = {}

    def update(self, ts: datetime, mid: float) -> None:
        t = ny(ts)  # ValueError on naive
        d = t.date()
        s = _session(d)
        if (
            s is None
            or not math.isfinite(mid)
            or mid <= 0
            or (self._cur is not None and d < self._cur)
        ):
            return
        o, n_min = s
        minute = int((t - o).total_seconds() // 60)
        if not 0 <= minute < n_min:
            return
        if d != self._cur:
            if self._cur is not None:
                self.finalize_session(self._cur)
                prev = self._mids[self._cur]
                self._prev_close_mid = prev[max(prev)]
            self._cur = d
            self._mids[d] = {}
            on = math.log(mid / self._prev_close_mid) if self._prev_close_mid else 0.0
            self._overnight[d] = on * on
        self._mids[d][minute] = float(mid)

    def warm_from_bars(self, bars: pd.DataFrame) -> None:
        """Replay bars (columns ts, mid|close); finalizes the last session if it reaches its last minute."""
        col = "mid" if "mid" in bars.columns else "close"
        for ts, mid in zip(bars["ts"], bars[col], strict=True):
            self.update(ts, float(mid))
        if self._cur is not None and max(self._mids[self._cur]) >= _session(self._cur)[1] - 1:
            self.finalize_session(self._cur)

    def finalize_session(self, d: date) -> None:
        m = self._mids[d]
        # ponytail: half-session coverage guard; a thin session's log RV is a permanent leverage point in the HAR fit
        if max(m) - min(m) < MINUTES // 2:
            return
        self._daily[d] = _rv5(m) + self._overnight.get(d, 0.0)
        self._slot_r2[d] = _slot_r2(m)

    def daily_rv(self) -> pd.Series:
        return pd.Series(dict(sorted(self._daily.items())), dtype=float)

    def sessions(self) -> int:
        return len(self._daily)

    def forecast_variance_day(self) -> float | None:
        """Next-session daily variance: None < 5 sessions, EWMA < har_min_sessions, else log-HAR."""
        rv = self.daily_rv().to_numpy()
        n, lam = len(rv), self.cfg.ewma_lambda
        if n < 5:
            return None
        rows = range(21, n - 1)  # d with a full 22-session window and a next session
        if n < self.cfg.har_min_sessions or len(rows) < 8:
            v = rv[0]
            for x in rv[1:]:
                v = lam * v + (1 - lam) * x
            return float(v)
        # ponytail: log-HAR only; realized kernels / jump-robust RV when the backtest shows gate decisions change
        lrv = np.log(np.maximum(rv, 1e-12))

        def x(d: int) -> list[float]:
            return [1.0, lrv[d], _lmean(rv[d - 4 : d + 1]), _lmean(rv[d - 21 : d + 1])]

        X = np.array([x(d) for d in rows])
        y = lrv[np.array(rows) + 1]
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        s2 = float(np.sum((y - X @ beta) ** 2) / (len(y) - X.shape[1]))
        return float(math.exp(np.dot(x(n - 1), beta) + 0.5 * s2))

    def forecast_sigma(self) -> float | None:
        v = self.forecast_variance_day()
        return None if v is None else math.sqrt(252 * v)

    def profile(self) -> np.ndarray:
        """Diurnal variance profile (78 slots, mean 1) over the last profile_sessions completed sessions."""
        if self.sessions() < 20:
            p = np.ones(SLOTS)
        else:
            keys = sorted(self._slot_r2)[-self.cfg.profile_sessions :]
            a = np.vstack([self._slot_r2[k] for k in keys])
            cnt = np.sum(~np.isnan(a), axis=0)
            p = np.nansum(a, axis=0) / np.maximum(cnt, 1)
            p[cnt == 0] = (
                p[cnt > 0].mean() if cnt.any() else 1.0
            )  # never-covered slot (15:55 on 5-min bars)
        return p / p.mean()

    def slot(self, now: datetime) -> int:
        t = ny(now)
        s = (t - t.replace(hour=9, minute=30, second=0, microsecond=0)).total_seconds() // 300
        return int(min(max(s, 0), SLOTS - 1))

    def horizon_variance(self, now: datetime, step_s: float, overnight_frac: float) -> float | None:
        """W_var for the QP: Σ_k σ_f²·π(slot_k)·dt over the steps to close + overnight_frac·σ_f²/252."""
        v = self.forecast_variance_day()
        if v is None:
            return None
        sig2 = 252 * v
        t0 = ny(now)
        n = steps_to_close(now, step_s)
        slots = [self.slot(t0 + timedelta(seconds=k * step_s)) for k in range(n)]
        return (
            sig2 * dt_years(step_s) * float(self.profile()[slots].sum())
            + overnight_frac * sig2 / 252
        )
