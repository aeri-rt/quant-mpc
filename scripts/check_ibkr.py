"""Paper-session checklist. Connects read-only and NEVER places an order.

Steps: clock offset, SPY quote freshness, one-expiry chain, one ATM leg's tick-106 Greeks against
Black-Scholes (the one-time vega/theta unit check that sets greeks.ibkr_* in live.yaml), BAG
whatIf margin for one straddle, account summary and the contract count the risk fractions allow.
Each step prints [ok]/[FAIL] with the measured value; exit 1 on any FAIL.
"""

import argparse
import asyncio
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import bs, log
from src.broker import Broker, BrokerError
from src.clock import Clock, T_years
from src.config import Live, load_live
from src.risk import Risk

RATIO_OK = (0.7, 1.4)  # IBKR/BS inside this band → already in our unit


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Paper checklist against TWS (read-only, never orders)."
    )
    p.add_argument("--config", default="config/live.yaml")
    p.add_argument("--port", type=int, help="override broker.port")
    p.add_argument("--client-id", type=int, help="override broker.client_id")
    return p.parse_args(argv)


def report(ok: bool, name: str, value: object) -> bool:
    print(f"[{'ok' if ok else 'FAIL'}] {name}: {value}")
    return ok


def unit_flag(ratio: float, scale: float) -> bool | None:
    """True: IBKR already in our unit; False: divide by `scale`; None: unknown (keep null)."""
    lo, hi = RATIO_OK
    if lo <= ratio <= hi:
        return True
    if lo * scale <= ratio <= hi * scale:
        return False
    return None


async def poll(fn, timeout_s: float, step_s: float = 0.5):
    """fn() every step_s until it returns non-None or timeout_s elapses (then its last value)."""
    for _ in range(math.ceil(timeout_s / step_s)):
        if (v := fn()) is not None:
            return v
        await asyncio.sleep(step_s)
    return fn()


async def checks(cfg: Live, broker: Broker, clock: Clock) -> bool:
    ok = True
    ins, pos = cfg.instrument, cfg.position
    sym, r, q = ins.symbol, ins.risk_free_rate, ins.dividend_yield

    # 1. clock offset + requested market data type
    off = broker.clock_offset_s
    ok &= report(
        abs(off) <= 2,
        "clock_offset_s",
        f"{off:+.2f} (market_data_type requested={broker.market_data_type})",
    )

    # 2. fresh stock quote (require_live=False, as the Runner)
    con_id = await broker.stream_stock(sym)
    quote = await poll(lambda: broker.quote(con_id, clock.now(), False), 10)
    if quote is None:
        return report(False, f"{sym} quote", "none within 10 s")
    age = (clock.now() - quote.ts).total_seconds()
    report(
        True,
        f"{sym} quote",
        f"bid={quote.bid} ask={quote.ask} age_s={age:.1f} md={quote.market_data_type}",
    )
    if quote.market_data_type != 1:
        print(
            "[WARN] delayed data: dry-run works; --execute refuses entries and hedges (`delayed`)"
        )
    S = quote.mid

    # 3. one-expiry chain, ATM pair (both rights quoted)
    chain = await broker.chain(sym, pos.entry_dte, clock.now())
    by_k: dict[float, dict[str, object]] = {}
    for oq in chain:
        by_k.setdefault(oq.strike, {})[oq.right] = oq
    pairs = {k: v for k, v in by_k.items() if len(v) == 2}
    expiry = f", expiry {chain[0].expiry.date()}" if chain else ""
    if not pairs:
        return report(False, "chain", f"{len(chain)} quotes, no two-sided strike{expiry}")
    K = min(pairs, key=lambda k: abs(k - S))
    call, put = pairs[K]["C"], pairs[K]["P"]
    report(True, "chain", f"{len(chain)} quotes, {len(pairs)} two-sided strikes{expiry}")
    print(f"      ATM K={K} S={S:.2f} call.mid={call.mid:.2f} put.mid={put.mid:.2f}")

    # 4. one streamed leg: tick-106 Greeks vs BS → the unit flags for live.yaml
    await broker.stream_leg(call.con_id)
    g = await poll(lambda: broker.leg_greeks(call.con_id, S, clock.now()), 15)
    lq = broker.leg_quote(call.con_id, clock.now())
    ok &= report(
        lq is not None,
        "leg_quote",
        "none after 15 s (the Runner cannot exit a held leg without it)"
        if lq is None
        else f"bid={lq.bid} ask={lq.ask} iv={lq.iv}",
    )
    if g is None:
        ok = report(False, "leg_greeks", "no modelGreeks after 15 s")
    else:
        b = bs.greeks(g.S_ref, K, T_years(call.expiry, clock.now()), r, q, g.iv, "C")
        print(
            f"      ibkr delta={g.delta:.4f} gamma={g.gamma:.5f} vega={g.vega} theta={g.theta} "
            f"iv={g.iv:.4f} undPrice={g.S_ref}"
        )
        print(
            f"      bs   delta={b.delta:.4f} gamma={b.gamma:.5f} vega={b.vega:.4f}/volpt "
            f"theta={b.theta:.4f}/day"
        )
        rv = g.vega / b.vega if g.vega is not None and b.vega else math.nan
        rt = g.theta / b.theta if g.theta is not None and b.theta else math.nan
        fv, ft = unit_flag(rv, 100), unit_flag(rt, 365)
        known = fv is not None and ft is not None
        ok &= report(known, "greeks units", f"ratio_vega={rv:.3f} ratio_theta={rt:.3f}")
        if known:
            print("      set in config/live.yaml:")
            print(f"      greeks.ibkr_vega_per_volpt: {str(fv).lower()}")
            print(f"      greeks.ibkr_theta_per_day: {str(ft).lower()}")
        else:
            print("      units unknown: keep greeks.ibkr_vega_per_volpt / ibkr_theta_per_day null")

    # 5. whatIf margin for one straddle (a whatIf is not an order)
    try:
        margin = await broker.whatif_straddle(call, put, 1, call.mid + put.mid + 0.06)
        report(True, "whatIf 1 straddle", f"initMarginChange={margin:.2f}")
    except BrokerError as e:
        ok = report(False, "whatIf 1 straddle", repr(e))

    # 6. account + the size the risk fractions allow
    acct = await broker.account()
    if acct is None:
        return report(False, "account", "summary unavailable")
    report(
        True,
        "account",
        f"NLV={acct.nlv:.0f} ExcessLiquidity={acct.excess_liquidity:.0f} "
        f"BuyingPower={acct.buying_power:.0f} DayTradesRemaining={acct.day_trades_remaining}",
    )
    risk = Risk(cfg.risk, cfg.loop, execute=False)
    n, why = risk.size_contracts(call.mid + put.mid, acct, None, pos.max_contracts)
    premium = (call.mid + put.mid) * 100
    print(f"      max contracts by risk fractions: {n} ({why}, premium per straddle {premium:.0f})")
    implied_nlv = premium * pos.max_contracts / cfg.risk.max_premium_frac_nlv
    if acct.nlv > 5 * implied_nlv:
        print(
            f"[WARN] NLV {acct.nlv:.0f} is > 5x the {implied_nlv:.0f} that max_contracts implies: "
            "looks like the $1M paper default; reset paper NLV to the intended live size (runbook)"
        )
    return ok


async def main_async(args: argparse.Namespace) -> int:
    cfg = load_live(args.config)
    over = {k: getattr(args, k) for k in ("client_id", "port") if getattr(args, k) is not None}
    cfg = cfg.model_copy(update={"broker": cfg.broker.model_copy(update=over)})
    log.setup(cfg.log.dir, cfg.log.level)
    clock = Clock()
    broker = Broker(
        cfg.broker,
        clock,
        readonly=True,
        stale_quote_s=cfg.loop.stale_quote_s,
        roll_dte=cfg.position.roll_dte,
        prefer_monthly=cfg.position.prefer_monthly,
    )
    b = cfg.broker
    print(
        f"connecting {b.host}:{b.port} client_id={b.client_id} market_data_type={b.market_data_type}"
    )
    try:
        await broker.connect()
    except (OSError, TimeoutError, BrokerError) as e:
        return int(not report(False, "connect", repr(e)))
    try:
        ok = await checks(cfg, broker, clock)
    finally:
        await broker.disconnect()
    print("SUMMARY: all checks passed" if ok else "SUMMARY: FAIL (fix before --execute)")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    return asyncio.run(main_async(parse_args(argv)))


if __name__ == "__main__":
    sys.exit(main())
