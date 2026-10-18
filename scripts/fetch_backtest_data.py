#!/usr/bin/env python3
"""
Fetch and cache historical data for backtesting.

Usage:
    # Fetch VIX from yfinance + synthetic SPY bars (no IBKR required)
    python scripts/fetch_backtest_data.py --synthetic

    # Fetch VIX from yfinance + SPY bars from IBKR TWS
    python scripts/fetch_backtest_data.py --ibkr

    # Custom date range
    python scripts/fetch_backtest_data.py --synthetic --start 2025-11-01 --end 2026-01-31
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtest.config import BacktestConfig
from backtest.data_fetcher import (
    fetch_vix,
    fetch_spy_bars_ibkr,
    generate_synthetic_bars,
    generate_synthetic_vix,
)


def main():
    parser = argparse.ArgumentParser(description="Fetch backtest data")
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to backtest YAML config (default: config/backtest.yaml)",
    )
    parser.add_argument("--start", type=str, default=None, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default=None, help="End date (YYYY-MM-DD)")
    parser.add_argument(
        "--ibkr", action="store_true",
        help="Fetch SPY bars from IBKR TWS (requires running TWS)",
    )
    parser.add_argument(
        "--synthetic", action="store_true",
        help="Generate synthetic SPY bars + VIX (no external data needed)",
    )
    args = parser.parse_args()

    # Load config
    if args.config:
        cfg = BacktestConfig.from_yaml(args.config)
    else:
        cfg = BacktestConfig()

    start = args.start or cfg.start_date
    end = args.end or cfg.end_date
    cache_dir = cfg.cache_dir

    print(f"Date range: {start} to {end}")
    print(f"Cache dir:  {cache_dir}")
    print()

    if args.synthetic:
        print("Generating synthetic SPY bars...")
        bars = generate_synthetic_bars(start, end, cfg.bar_size_minutes, cache_dir=cache_dir)
        print(f"  -> {len(bars)} bars\n")

        print("Generating synthetic VIX data...")
        vix = generate_synthetic_vix(start, end, cache_dir=cache_dir)
        print(f"  -> {len(vix)} daily rows\n")
    else:
        print("Fetching VIX from yfinance...")
        vix = fetch_vix(start, end, cache_dir=cache_dir)
        print(f"  -> {len(vix)} daily rows\n")

        if args.ibkr:
            print("Fetching SPY bars from IBKR TWS...")
            bars = fetch_spy_bars_ibkr(start, end, cfg.bar_size_minutes, cache_dir=cache_dir)
            print(f"  -> {len(bars)} bars\n")
        else:
            print("No SPY data source specified. Use --ibkr or --synthetic.")
            print("Run with --synthetic for development/testing without TWS.")
            sys.exit(1)

    print("Done! Data cached and ready for backtesting.")


if __name__ == "__main__":
    main()
