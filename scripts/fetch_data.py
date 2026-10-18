"""Populate data/cache: Cboe term structure (--cboe) or IBKR 1-min bars (--ibkr).
Synthetic backtests need no fetch: run_backtest.py --synthetic generates bars from the config seed.

usage: fetch_data.py [--cboe] [--ibkr [--port 7497] [--client-id 11]]
                     [--start YYYY-MM-DD] [--end YYYY-MM-DD] [--bar-s 60] [--config config/backtest.yaml]
"""

import argparse
import asyncio
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtest.data import fetch_spy_bars_ibkr, load_term_structure
from src.broker import Broker
from src.clock import Clock
from src.config import load_backtest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--cboe", action="store_true", help="VIX9D/VIX/VIX3M closes from Cboe (cached)")
    p.add_argument(
        "--ibkr", action="store_true", help="MIDPOINT bars via TWS/Gateway (cached per session)"
    )
    p.add_argument("--start", type=date.fromisoformat, help="override backtest.start")
    p.add_argument("--end", type=date.fromisoformat, help="override backtest.end")
    p.add_argument("--bar-s", type=int, help="override backtest.bar_s")
    p.add_argument("--port", type=int, help="override broker.port")
    p.add_argument("--client-id", type=int, default=11)
    p.add_argument("--config", default="config/backtest.yaml")
    a = p.parse_args(argv)
    if not (a.cboe or a.ibkr):
        p.error("choose at least one of --cboe, --ibkr")
    return a


async def main(argv: list[str] | None = None) -> int:
    a = parse_args(argv)
    bt, live = load_backtest(a.config)
    start, end, bar_s = a.start or bt.start, a.end or bt.end, a.bar_s or bt.bar_s
    cache, v = Path(bt.cache_dir), bt.cache_version
    cache.mkdir(parents=True, exist_ok=True)
    if a.cboe:
        print(load_term_structure(cache, v).tail(3))
    if a.ibkr:
        cfg = live.broker.model_copy(
            update={"port": a.port or live.broker.port, "client_id": a.client_id}
        )
        broker = Broker(cfg, Clock(), readonly=True)
        await broker.connect()
        try:
            bars = await fetch_spy_bars_ibkr(
                broker, start, end, bar_s, cache, v, live.instrument.symbol
            )
        finally:
            await broker.disconnect()
        print(f"ibkr: {bars['ts'].dt.date.nunique()} sessions, {len(bars)} bars")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
