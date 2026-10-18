#!/usr/bin/env python3
"""
Run a backtest of the gamma scalping strategy.

Usage:
    # With default config and synthetic data
    python scripts/run_backtest.py --synthetic

    # With custom config
    python scripts/run_backtest.py --config config/backtest.yaml --synthetic

    # With cached real data (fetch first with scripts/fetch_backtest_data.py)
    python scripts/run_backtest.py --config config/backtest.yaml

    # Override date range
    python scripts/run_backtest.py --synthetic --start 2025-12-01 --end 2026-01-15
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.logging import get_logger

logger = get_logger("backtest")

from backtest.config import BacktestConfig
from backtest.data_fetcher import (
    fetch_vix,
    generate_synthetic_bars,
    generate_synthetic_vix,
)
from backtest.engine import BacktestEngine
from backtest.report import BacktestReport


def main():
    parser = argparse.ArgumentParser(description="Run gamma scalping backtest")
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to backtest YAML config",
    )
    parser.add_argument("--start", type=str, default=None, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default=None, help="End date (YYYY-MM-DD)")
    parser.add_argument(
        "--synthetic", action="store_true",
        help="Use synthetic data (no TWS or yfinance required)",
    )
    parser.add_argument(
        "--no-plots", action="store_true",
        help="Skip plot generation",
    )
    args = parser.parse_args()

    # -- Load config ---------------------------------------------------
    if args.config:
        cfg = BacktestConfig.from_yaml(args.config)
    else:
        cfg = BacktestConfig()

    if args.start:
        cfg.start_date = args.start
    if args.end:
        cfg.end_date = args.end

    cfg.ensure_dirs()

    # -- Load data -----------------------------------------------------
    if args.synthetic:
        logger.info("Using synthetic data")
        bars = generate_synthetic_bars(
            cfg.start_date, cfg.end_date,
            cfg.bar_size_minutes,
            cache_dir=cfg.cache_dir,
        )
        vix = generate_synthetic_vix(
            cfg.start_date, cfg.end_date,
            cache_dir=cfg.cache_dir,
        )
    else:
        # Try loading cached parquet files
        import pandas as pd

        bars_path = Path(cfg.cache_dir) / f"spy_bars_{cfg.bar_size_minutes}m_{cfg.start_date}_{cfg.end_date}.parquet"
        vix_path = Path(cfg.cache_dir) / f"vix_{cfg.start_date}_{cfg.end_date}.parquet"

        if not bars_path.exists():
            logger.error(
                f"SPY bars not found at {bars_path}. "
                "Run scripts/fetch_backtest_data.py first, or use --synthetic."
            )
            sys.exit(1)

        if not vix_path.exists():
            logger.warning(f"VIX data not found at {vix_path}, trying yfinance...")
            vix = fetch_vix(cfg.start_date, cfg.end_date, cfg.cache_dir)
        else:
            vix = pd.read_parquet(vix_path)

        bars = pd.read_parquet(bars_path)

    logger.info(f"Data loaded: {len(bars)} bars, {len(vix)} VIX rows")

    # -- Run backtest --------------------------------------------------
    engine = BacktestEngine(cfg)
    result = engine.run(bars, vix)

    # -- Report --------------------------------------------------------
    report = BacktestReport(result)
    report.print_summary()

    if not args.no_plots:
        report.generate_plots()

    logger.info("Backtest complete.")


if __name__ == "__main__":
    main()
