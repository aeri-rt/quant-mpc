"""Replay SPY bars through the shared Runner per hedger; print summary, attribution, comparison (and --sweep frontier) tables; write PNGs + CSVs.

usage: run_backtest.py [--config config/backtest.yaml] [--synthetic | --ibkr]
                       [--hedger band|qp|naive|all]
                       [--start YYYY-MM-DD] [--end YYYY-MM-DD] [--sweep] [--force-open]
                       [--out DIR] [--interval S]
"""

import argparse
import asyncio
import sys
import time
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtest import report
from backtest.data import load_bars, load_term_structure, synthetic_term_structure
from src.clock import XNYS
from src.config import Backtest, Live, load_backtest
from src.log import setup

HEDGERS = ("band", "qp", "naive")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", default="config/backtest.yaml")
    src = p.add_mutually_exclusive_group()
    src.add_argument("--synthetic", action="store_true", help="force source: synthetic")
    src.add_argument("--ibkr", action="store_true", help="force source: ibkr (cached bars)")
    p.add_argument("--hedger", choices=[*HEDGERS, "all"], help="override backtest.hedger")
    p.add_argument("--start", type=date.fromisoformat, help="override backtest.start")
    p.add_argument("--end", type=date.fromisoformat, help="override backtest.end")
    p.add_argument("--sweep", action="store_true", help="also sweep risk aversion for band and qp")
    p.add_argument("--force-open", action="store_true", help="bypass the alpha gate (hold always)")
    p.add_argument("--out", help="override backtest.out_dir")
    p.add_argument(
        "--interval", type=int, help="bar size in seconds; overrides bar_s AND loop.interval_s"
    )
    return p.parse_args(argv)


def load(a: argparse.Namespace) -> tuple[Backtest, Live]:
    bt, live = load_backtest(a.config)
    over = {"start": a.start, "end": a.end, "hedger": a.hedger, "out_dir": a.out}
    upd = {k: v for k, v in over.items() if v is not None}
    if a.synthetic or a.ibkr:
        upd["source"] = "synthetic" if a.synthetic else "ibkr"
    if a.interval:
        upd["bar_s"] = a.interval
        loop = live.loop.model_copy(update={"interval_s": float(a.interval)})
        live = live.model_copy(update={"loop": loop})
    bt = bt.model_copy(update=upd)
    if bt.bar_s != live.loop.interval_s:
        sys.exit(f"bar_s {bt.bar_s} != loop.interval_s {live.loop.interval_s}: use --interval")
    return bt, live


async def main(argv: list[str] | None = None) -> int:
    t0 = time.monotonic()
    a = parse_args(argv)
    bt, live = load(a)
    setup(None, "WARNING")  # tables are the output; no JSONL audit trail for a replay
    from backtest.sim import run  # pulls in the Runner; keeps --help/parse_args import-light

    try:
        bars = load_bars(bt, live)
    except ValueError as e:  # ibkr source with nothing cached: carries the fetch_data.py hint
        sys.exit(str(e))
    if bt.source == "synthetic":
        sessions = XNYS.sessions_in_range(bt.start - timedelta(days=7), bt.end)  # ≥1 prior row
        term = synthetic_term_structure(sessions, bt.synthetic.iv, bt.skew_b)
    else:
        term = load_term_structure(Path(bt.cache_dir), bt.cache_version)

    hedgers = HEDGERS if bt.hedger == "all" else (bt.hedger,)
    results = {}
    for h in hedgers:
        results[h] = await run(bars, term, live, bt, h, force_open=a.force_open)
    first = next(iter(results.values()))
    print(report.summary_table(results, live.gate.turnover_frac))
    print(f"\nattribution ({first.hedger}, last 10 sessions)")
    print(report.attribution_table(first, last_n=10))
    print("\ncomparison")
    print(report.comparison_table(results))
    if a.sweep:
        sweep = {}
        for policy in ("band", "qp"):
            sweep[policy] = [
                await run(bars, term, live, bt, policy, risk_aversion=ra, force_open=a.force_open)
                for ra in getattr(bt.sweep_risk_aversion, policy)
            ]
        print("\nfrontier")
        print(report.frontier_table(sweep))

    out = Path(bt.out_dir)
    for p in report.plots(results, out) + report.write_csvs(results, out):
        print(p)
    print(f"elapsed {time.monotonic() - t0:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
