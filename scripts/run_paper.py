"""Dry-run (default) / paper (--execute) / live (--execute --live --port 7496) entrypoint.

argparse + wiring only: positions, market hours, retries and the kill switch live in src.runner.
Without --execute every order goes to SimBroker and no placeOrder can be reached.
"""

import argparse
import asyncio
import signal
import sys
from pathlib import Path
from typing import NamedTuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import log
from src.broker import Broker
from src.clock import Clock
from src.config import Live, load_live
from src.runner import Runner, State
from src.sim_broker import SimBroker
from src.store import Store

LIVE_PORT = 7496
USAGE = """run_paper.py [run] [--execute] [--live --port 7496] [--config PATH] [--interval S]
                    [--contracts N] [--force-open] [--client-id N] [--port N]
       run_paper.py flatten --execute [--config PATH] [--client-id N] [--port N]"""


class Wiring(NamedTuple):
    cfg: Live
    mode: str  # dry_run | paper | live
    store: Store
    broker: Broker | SimBroker
    runner: Runner


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in ("run", "flatten", "-h", "--help"):
        argv.insert(0, "run")  # `run` is the default subcommand
    p = argparse.ArgumentParser(
        usage=USAGE,
        description="Gated long-gamma SPY straddle, delta-hedged. Dry-run unless --execute.",
    )
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", default="config/live.yaml")
    common.add_argument(
        "--execute",
        action="store_true",
        help="place orders via TWS (default: dry-run via SimBroker)",
    )
    common.add_argument("--client-id", type=int, help="override broker.client_id")
    common.add_argument("--port", type=int, help="override broker.port (7497 paper, 7496 live)")
    sub = p.add_subparsers(dest="cmd", required=True, prog="run_paper.py")  # else USAGE+" run"
    run = sub.add_parser("run", parents=[common], help="trade (default)")
    run.add_argument(
        "--live",
        action="store_true",
        help="live account: needs --execute, --port 7496, typed CONFIRM",
    )
    run.add_argument("--interval", type=float, help="override loop.interval_s")
    run.add_argument("--contracts", type=int, help="fixed straddle count (default: sized from NLV)")
    run.add_argument(
        "--force-open",
        action="store_true",
        help="enter with the gate closed (paper checklist only)",
    )
    flat = sub.add_parser(
        "flatten", parents=[common], help="liquidate straddle + shares (--execute)"
    )
    flat.set_defaults(live=False, interval=None, contracts=None, force_open=False)
    return p.parse_args(argv)


def apply_overrides(cfg: Live, args: argparse.Namespace) -> Live:
    broker = {k: getattr(args, k) for k in ("client_id", "port") if getattr(args, k) is not None}
    loop = {} if args.interval is None else {"interval_s": args.interval}
    return cfg.model_copy(
        update={
            "broker": cfg.broker.model_copy(update=broker),
            "loop": cfg.loop.model_copy(update=loop),
        }
    )


def refuse_live(args: argparse.Namespace, cfg: Live) -> str | None:
    """Why --live must not proceed, or None; runs before anything connects."""
    if not args.live:
        return None
    if not args.execute:
        return "--live requires --execute"
    if args.force_open:
        return "--live with --force-open is refused"
    if args.port != LIVE_PORT:
        return f"--live requires --port {LIVE_PORT} on the command line"
    r, p = cfg.risk, cfg.position
    print(
        "LIVE: real orders, real money. Limits: "
        f"max_contracts={p.max_contracts} premium<={r.max_premium_frac_nlv:.0%} NLV "
        f"daily_loss<={r.max_daily_loss_frac_nlv:.0%} NLV "
        f"margin<={r.max_margin_frac_excess:.0%} excess_liquidity"
    )
    if input("Type CONFIRM to trade LIVE: ") != "CONFIRM":
        return "not confirmed"
    return None


def build_components(args: argparse.Namespace, cfg: Live | None = None) -> Wiring:
    """Wire everything without connecting. --execute is the only switch to the real order path."""
    if cfg is None:
        cfg = apply_overrides(load_live(args.config), args)
    clock = Clock()
    ib = Broker(
        cfg.broker,
        clock,
        readonly=not args.execute,
        stale_quote_s=cfg.loop.stale_quote_s,
        roll_dte=cfg.position.roll_dte,
        prefer_monthly=cfg.position.prefer_monthly,
    )
    broker = ib if args.execute else SimBroker(ib, cfg.costs, clock)
    mode = "live" if args.live else "paper" if args.execute else "dry_run"
    store = Store(cfg.store.path)
    runner = Runner(
        cfg,
        broker,
        store,
        clock,
        execute=args.execute,
        contracts=args.contracts,
        force_open=args.force_open,
        mode=mode,
    )
    return Wiring(cfg, mode, store, broker, runner)


def banner(w: Wiring, args: argparse.Namespace) -> str:
    cfg = w.cfg
    n = args.contracts or cfg.position.contracts or "sized from account"
    return (
        f"{args.cmd}: mode={w.mode} port={cfg.broker.port} client_id={cfg.broker.client_id} "
        f"market_data_type={cfg.broker.market_data_type} interval_s={cfg.loop.interval_s:g} "
        f"policy={cfg.hedge.policy} contracts={n} force_open={args.force_open}"
    )


def exit_code(runner: Runner) -> int:
    """0 clean stop, 3 halted (reason printed), 1 anything else."""
    if runner.state == State.HALTED or runner.risk.halted:
        print(f"HALTED: {runner.risk.halt_reason}", file=sys.stderr)
        return 3
    return 0 if runner.state == State.STOPPED else 1


def on_signals(fn, *args) -> None:
    """Ctrl-C and SIGTERM call fn(*args) on the event loop."""
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, fn, *args)
        except (NotImplementedError, AttributeError):  # Windows ProactorEventLoop / missing SIGTERM
            signal.signal(sig, lambda *_: loop.call_soon_threadsafe(fn, *args))


async def run(w: Wiring) -> int:
    runner = w.runner
    on_signals(runner.stop, "signal")
    try:
        await runner.run()
    except KeyboardInterrupt:  # before the handler is installed, or Windows
        runner.stop("kbd")
        await runner.shutdown()
    return exit_code(runner)


async def flatten(w: Wiring) -> int:
    """A signal cancels the flatten; Runner.flatten's finally cancels our orders first."""
    on_signals(asyncio.current_task().cancel)
    try:
        return await w.runner.flatten()
    except asyncio.CancelledError:
        print("flatten interrupted: open orders cancelled, check positions", file=sys.stderr)
        return 1


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.cmd == "flatten" and not args.execute:
        print("flatten requires --execute (it places real closing orders)", file=sys.stderr)
        return 1
    cfg = apply_overrides(load_live(args.config), args)
    if reason := refuse_live(args, cfg):
        print(f"refused: {reason}", file=sys.stderr)
        return 1
    log.setup(cfg.log.dir, cfg.log.level)
    w = build_components(args, cfg)
    print(banner(w, args))
    try:
        return asyncio.run(run(w) if args.cmd == "run" else flatten(w))
    finally:
        w.store.close()


if __name__ == "__main__":
    sys.exit(main())
