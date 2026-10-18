"""Backtest inputs: Cboe term structure, IBKR 1-min bars (per-session parquet cache), synthetic GBM bars.

Every bar `ts` is the bar START, tz-aware America/New_York, inside an XNYS session.
Caches are keyed by `cache_version` (bumps invalidate; nothing is reused across versions).
"""

import io
import math
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from backtest.iv import TENORS
from src.clock import NY, XNYS, dt_years
from src.config import Backtest, Live
from src.log import get_logger

log = get_logger("data")

CBOE_URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/{index}_History.csv"
_TERM = {"vix9d": "VIX9D", "vix": "VIX", "vix3m": "VIX3M"}
_PX = ["open", "high", "low", "close"]


# ---- Cboe term structure ----------------------------------------------------------------
def parse_cboe_csv(src) -> pd.Series:
    """Cboe {index}_History.csv (DATE,OPEN,HIGH,LOW,CLOSE) -> CLOSE/100 by normalised date.

    `src` is a path/URL or the CSV text itself; header case and trailing blank lines are tolerated.
    """
    if isinstance(src, str) and "\n" in src:
        src = io.StringIO(src)
    df = pd.read_csv(src)
    df.columns = df.columns.str.strip().str.upper()
    idx = pd.DatetimeIndex(pd.to_datetime(df["DATE"]).dt.normalize(), name="date")
    return pd.Series(df["CLOSE"].to_numpy(float) / 100, index=idx, name="close").sort_index()


def fetch_cboe(
    index: str, cache_dir: Path, cache_version: int, url: str | None = None
) -> pd.Series:
    """Decimal closes of VIX9D | VIX | VIX3M; network only when the parquet cache is missing."""
    p = Path(cache_dir) / f"{index}_v{cache_version}.parquet"
    if p.exists():
        return pd.read_parquet(p)[index]
    s = parse_cboe_csv(url or CBOE_URL.format(index=index)).rename(index)
    p.parent.mkdir(parents=True, exist_ok=True)
    s.to_frame().to_parquet(p)
    return s


def load_term_structure(cache_dir: Path, cache_version: int) -> pd.DataFrame:
    """Columns vix9d, vix, vix3m (decimals), inner-joined on date, sorted."""
    cols = {k: fetch_cboe(v, cache_dir, cache_version) for k, v in _TERM.items()}
    return pd.concat(cols, axis=1, join="inner").sort_index()


def synthetic_term_structure(
    sessions: pd.DatetimeIndex, iv: float, skew_b: float = 1.0
) -> pd.DataFrame:
    """Each tenor node at iv·√(1+3·T·b²), so atm_iv(term, d, T, skew_b) == iv at 9/30/91 d.

    Exactly flat at every T only for skew_b = 0: with b > 0 the linear total-variance interpolation
    of the convex iv²·T·(1+3Tb²) overshoots between nodes (b = 1: +0.003 at 20 d, +0.007 at 45 d).
    """
    idx = pd.DatetimeIndex(sessions).tz_localize(None).normalize().rename("date")
    cols = {c: iv * math.sqrt(1 + 3 * T * skew_b**2) for T, c in TENORS}
    return pd.DataFrame(cols, index=idx)


# ---- IBKR bars (per-session parquet cache) -----------------------------------------------
def _bars_root(cache_dir: Path, symbol: str, bar_s: int, cache_version: int) -> Path:
    return Path(cache_dir) / f"{symbol}_{bar_s}s_v{cache_version}"


def cached_bars(
    start: date, end: date, bar_s: int, cache_dir: Path, cache_version: int, symbol: str = "SPY"
) -> pd.DataFrame:
    """Concatenate the cached sessions in [start, end]; raise with the fetch hint when none exist."""
    root = _bars_root(cache_dir, symbol, bar_s, cache_version)
    files = [root / f"{d.date()}.parquet" for d in XNYS.sessions_in_range(start, end)]
    frames = [pd.read_parquet(p) for p in files if p.exists()]
    if not frames:
        raise ValueError(
            f"source ibkr: no cached bars under {root}; run "
            f"`python scripts/fetch_data.py --ibkr --start {start} --end {end} --bar-s {bar_s}`"
        )
    return pd.concat(frames, ignore_index=True).sort_values("ts", ignore_index=True)


async def fetch_spy_bars_ibkr(
    broker,
    start: date,
    end: date,
    bar_s: int,
    cache_dir: Path,
    cache_version: int,
    symbol: str = "SPY",
) -> pd.DataFrame:
    """Fill the per-session cache via broker.historical_bars (it paces itself; no sleeps here)."""
    root = _bars_root(cache_dir, symbol, bar_s, cache_version)
    root.mkdir(parents=True, exist_ok=True)
    for d in XNYS.sessions_in_range(start, end):
        p = root / f"{d.date()}.parquet"
        if p.exists():
            continue
        df = await broker.historical_bars(symbol, XNYS.session_close(d).to_pydatetime(), 1, bar_s)
        df = df[(df[_PX] > 0).all(axis=1)]  # IBKR marks missing bars with -1/NaN
        if df.empty:
            log.warning(f"no bars for {d.date()}; not cached")
            continue
        df.to_parquet(p, index=False)
    return cached_bars(start, end, bar_s, cache_dir, cache_version, symbol)


# ---- synthetic bars ----------------------------------------------------------------------
def synthetic_bars(
    start: date,
    end: date,
    bar_s: int,
    sigma: float,
    seed: int,
    S0: float,
    overnight_frac: float = 0.33,
) -> pd.DataFrame:
    """Zero-drift log-normal bars on XNYS sessions (early closes shorter), ts = bar start.

    Daily variance sigma^2/252 splits into an overnight gap (overnight_frac) and flat intraday
    bars of variance sigma^2 * dt_years(bar_s) * (1 - overnight_frac). open[i] == close[i-1]
    within a session.
    """
    rng = np.random.default_rng(seed)
    gap_sd = math.sqrt(overnight_frac * sigma**2 / 252)
    bar_var = sigma**2 * dt_years(bar_s) * (1 - overnight_frac)
    S, frames = float(S0), []
    for d in XNYS.sessions_in_range(start, end):
        o, c = XNYS.session_open(d), XNYS.session_close(d)
        ts = pd.date_range(o, c, freq=f"{bar_s}s", inclusive="left").tz_convert(NY)
        if frames:
            S *= math.exp(rng.normal(0, gap_sd))
        close = S * np.exp(np.cumsum(rng.normal(0, math.sqrt(bar_var), len(ts))))
        opn = np.concatenate(([S], close[:-1]))
        frames.append(
            pd.DataFrame(
                {
                    "ts": ts,
                    "open": opn,
                    "high": np.maximum(opn, close),
                    "low": np.minimum(opn, close),
                    "close": close,
                    "mid": close,
                }
            )
        )
        S = float(close[-1])
    return pd.concat(frames, ignore_index=True)


def load_bars(bt: Backtest, live: Live) -> pd.DataFrame:
    """Dispatch on bt.source: synthetic GBM, or the IBKR per-session cache (scripts/fetch_data.py)."""
    if bt.source == "synthetic":
        s = bt.synthetic
        return synthetic_bars(
            bt.start,
            bt.end,
            bt.bar_s,
            s.sigma,
            s.seed,
            s.S0,
            overnight_frac=live.vol.overnight_frac,
        )
    return cached_bars(
        bt.start, bt.end, bt.bar_s, Path(bt.cache_dir), bt.cache_version, live.instrument.symbol
    )
