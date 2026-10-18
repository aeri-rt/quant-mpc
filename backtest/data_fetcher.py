"""
Data fetcher for backtesting.

Downloads and caches:
- SPY intraday bars (from IBKR TWS or generated synthetically for dev)
- VIX daily close (from yfinance)

Data is cached as parquet files for fast offline replay.
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
from src.utils.logging import get_logger

logger = get_logger("backtest")


# =========================================================================
# VIX (yfinance)
# =========================================================================

def fetch_vix(
    start_date: str,
    end_date: str,
    cache_dir: str = "data/cache",
) -> pd.DataFrame:
    """
    Download VIX daily data via yfinance and cache to parquet.

    Args:
        start_date: ISO date string (e.g. "2025-12-01").
        end_date: ISO date string.
        cache_dir: Directory for parquet cache.

    Returns:
        DataFrame with DatetimeIndex and columns [open, high, low, close, volume].
    """
    import yfinance as yf

    cache_path = Path(cache_dir) / f"vix_{start_date}_{end_date}.parquet"
    if cache_path.exists():
        logger.info(f"Loading cached VIX data from {cache_path}")
        return pd.read_parquet(cache_path)

    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading VIX data {start_date} to {end_date}")
    df = yf.download("^VIX", start=start_date, end=end_date, progress=False)
    if df.empty:
        raise RuntimeError("yfinance returned empty VIX data")

    # Normalise column names to lowercase
    df.columns = [c.lower() if isinstance(c, str) else c for c in df.columns]
    # Handle MultiIndex columns from newer yfinance versions
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0].lower() for c in df.columns]

    df.to_parquet(cache_path)
    logger.info(f"Cached VIX data to {cache_path} ({len(df)} rows)")
    return df


# =========================================================================
# SPY intraday bars (IBKR or synthetic)
# =========================================================================

def fetch_spy_bars_ibkr(
    start_date: str,
    end_date: str,
    bar_size_minutes: int = 5,
    cache_dir: str = "data/cache",
) -> pd.DataFrame:
    """
    Fetch SPY intraday bars from IBKR TWS (requires live/paper TWS connection).

    This uses ib_insync to download historical bars. The TWS must be running
    and accepting connections on the configured port.

    Bars are cached to parquet after download.
    """
    cache_path = (
        Path(cache_dir)
        / f"spy_bars_{bar_size_minutes}m_{start_date}_{end_date}.parquet"
    )
    if cache_path.exists():
        logger.info(f"Loading cached SPY bars from {cache_path}")
        return pd.read_parquet(cache_path)

    try:
        from ib_insync import IB, Stock, util
    except ImportError:
        raise ImportError(
            "ib_insync is required for IBKR data fetch. "
            "Install with: pip install ib_insync"
        )

    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    ib = IB()
    ib.connect("127.0.0.1", 7497, clientId=99)

    contract = Stock("SPY", "SMART", "USD")
    ib.qualifyContracts(contract)

    bar_size_str = f"{bar_size_minutes} mins"
    all_bars = []

    # IBKR limits historical data requests; fetch in daily chunks
    current = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    while current <= end:
        end_dt = (current + timedelta(days=1)).strftime("%Y%m%d 23:59:59")
        try:
            bars = ib.reqHistoricalData(
                contract,
                endDateTime=end_dt,
                durationStr="1 D",
                barSizeSetting=bar_size_str,
                whatToShow="TRADES",
                useRTH=True,
                formatDate=1,
            )
            if bars:
                df_chunk = util.df(bars)
                all_bars.append(df_chunk)
        except Exception as e:
            logger.warning(f"Failed to fetch bars for {current.date()}: {e}")
        current += timedelta(days=1)

    ib.disconnect()

    if not all_bars:
        raise RuntimeError("No SPY bars fetched from IBKR")

    df = pd.concat(all_bars, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
    df.sort_index(inplace=True)

    # Normalise column names
    df.columns = [c.lower() for c in df.columns]

    df.to_parquet(cache_path)
    logger.info(f"Cached SPY bars to {cache_path} ({len(df)} rows)")
    return df


def generate_synthetic_bars(
    start_date: str,
    end_date: str,
    bar_size_minutes: int = 5,
    initial_price: float = 590.0,
    daily_vol: float = 0.01,
    cache_dir: str = "data/cache",
) -> pd.DataFrame:
    """
    Generate synthetic SPY intraday bars for testing when IBKR isn't available.

    Uses geometric Brownian motion to produce realistic-looking intraday data
    with 6.5h trading days (78 five-minute bars per day).

    Args:
        start_date: ISO date string.
        end_date: ISO date string.
        bar_size_minutes: Bar interval in minutes.
        initial_price: Starting price.
        daily_vol: Daily volatility (std of returns), e.g. 0.01 = 1%.
        cache_dir: Parquet cache directory.

    Returns:
        DataFrame with DatetimeIndex and columns [open, high, low, close, volume].
    """
    import numpy as np

    cache_path = (
        Path(cache_dir)
        / f"spy_synth_{bar_size_minutes}m_{start_date}_{end_date}.parquet"
    )
    if cache_path.exists():
        logger.info(f"Loading cached synthetic bars from {cache_path}")
        return pd.read_parquet(cache_path)

    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    # Build trading-day timestamps
    trading_hours = pd.bdate_range(start_date, end_date)
    bars_per_day = int(6.5 * 60 / bar_size_minutes)  # 78 for 5-min bars

    timestamps = []
    for day in trading_hours:
        market_open = day.replace(hour=9, minute=30)
        for j in range(bars_per_day):
            timestamps.append(market_open + timedelta(minutes=j * bar_size_minutes))

    n_bars = len(timestamps)
    if n_bars == 0:
        raise RuntimeError("No trading bars generated for the given date range")

    rng = np.random.default_rng(42)

    # Per-bar volatility
    bar_vol = daily_vol / np.sqrt(bars_per_day)
    returns = rng.normal(0, bar_vol, size=n_bars)

    prices = np.empty(n_bars)
    prices[0] = initial_price
    for i in range(1, n_bars):
        prices[i] = prices[i - 1] * np.exp(returns[i])

    # Simulate OHLC from close
    noise = rng.uniform(0, bar_vol * initial_price * 0.3, size=n_bars)
    df = pd.DataFrame(
        {
            "open": prices + noise * rng.choice([-1, 1], n_bars),
            "high": prices + abs(noise),
            "low": prices - abs(noise),
            "close": prices,
            "volume": rng.integers(50_000, 500_000, size=n_bars),
        },
        index=pd.DatetimeIndex(timestamps, name="date"),
    )

    df.to_parquet(cache_path)
    logger.info(f"Generated {n_bars} synthetic bars -> {cache_path}")
    return df


def generate_synthetic_vix(
    start_date: str,
    end_date: str,
    mean_vix: float = 18.0,
    vix_vol: float = 2.0,
    cache_dir: str = "data/cache",
) -> pd.DataFrame:
    """
    Generate synthetic daily VIX data for testing.

    Mean-reverts around *mean_vix* with noise ~ N(0, vix_vol).
    """
    import numpy as np

    cache_path = Path(cache_dir) / f"vix_synth_{start_date}_{end_date}.parquet"
    if cache_path.exists():
        logger.info(f"Loading cached synthetic VIX from {cache_path}")
        return pd.read_parquet(cache_path)

    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    dates = pd.bdate_range(start_date, end_date)
    rng = np.random.default_rng(123)

    vix = np.empty(len(dates))
    vix[0] = mean_vix
    for i in range(1, len(dates)):
        vix[i] = vix[i - 1] + 0.1 * (mean_vix - vix[i - 1]) + rng.normal(0, vix_vol)
        vix[i] = max(vix[i], 9.0)  # VIX floor

    df = pd.DataFrame(
        {"close": vix},
        index=pd.DatetimeIndex(dates, name="date"),
    )
    df.to_parquet(cache_path)
    logger.info(f"Generated {len(df)} synthetic VIX rows -> {cache_path}")
    return df
