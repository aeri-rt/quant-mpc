#!/usr/bin/env python3
"""
Run LIVE trading for the quant-mpc system.

⚠️  WARNING: This script trades with REAL MONEY.
⚠️  Use with extreme caution.
⚠️  Start with very small position sizes.

Usage:
    python scripts/run_live.py --confirm
"""

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Run LIVE trading")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/settings.yaml"),
        help="Path to configuration file",
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Confirm that you want to trade with real money",
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("⚠️  Quant-MPC LIVE Trading ⚠️")
    print("=" * 60)
    
    if not args.confirm:
        print("\n❌ ERROR: Live trading requires --confirm flag")
        print("\nThis script trades with REAL MONEY.")
        print("If you understand the risks, run with:")
        print("    python scripts/run_live.py --confirm")
        sys.exit(1)
    
    print(f"Config: {args.config}")
    print("Mode: ⚠️  LIVE TRADING ⚠️")
    print("=" * 60)
    
    # Double confirmation
    response = input("\n⚠️  Type 'I UNDERSTAND THE RISKS' to continue: ")
    if response != "I UNDERSTAND THE RISKS":
        print("\n❌ Confirmation failed. Exiting.")
        sys.exit(1)
    
    # TODO: Implement live trading logic
    print("\n[INFO] Live trading not yet implemented.")
    print("[INFO] This is a placeholder script.")


if __name__ == "__main__":
    main()

