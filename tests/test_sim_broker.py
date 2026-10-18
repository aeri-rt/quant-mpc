"""src/sim_broker.py: share orders fill on the NEXT quote only (limit audit-only), BAG limit
honoured and capped at the touch, scripted outcomes, ledger merged over broker truth."""

import pytest

from src.costs import commission, reg_fees
from src.sim_broker import SimBroker
from src.types import OrderState, Position
from tests.conftest import make_chain

SYM = "SPY"
approx = pytest.approx


def tick(b, clock, seconds=60):
    """Advance the clock and pull the stock quote: the only moment pending share orders fill."""
    clock.advance(seconds)
    return b.quote(b.stock_con_id(SYM), clock.now(), False)


def cash(b):
    """Net cash from every fill so far (negative = paid out)."""
    return -sum(f.qty * f.price + (f.commission or 0.0) for f in b.fills)


def atm(now, S=600.0, dte=30, iv=0.15):
    by_id = {q.con_id: q for q in make_chain(S, dte, iv, now)}
    return by_id[1001], by_id[1002]


# 1. fill on the next quote at the ask with the modelled commission
async def test_share_order_fills_on_next_quote(sim_broker, fake_market, sim_clock):
    o = await sim_broker.place_shares(100, None, "hedge")
    assert o.state is OrderState.SUBMITTED and o.action == "BUY" and o.qty == 100
    assert not sim_broker.fills and sim_broker.open_orders() == [o]
    tick(sim_broker, sim_clock)
    f = sim_broker.fills[0]
    assert (f.qty, f.price, f.commission, f.source) == (100, approx(600.005), approx(0.35), "sim")
    assert f.order_id == o.id and f.con_id == fake_market.stock_con_id(SYM)
    assert o.state is OrderState.FILLED and o.filled == 100 and o.avg_price == approx(600.005)
    assert sim_broker.open_orders() == []
    p = (await sim_broker.positions())[fake_market.stock_con_id(SYM)]
    assert (p.qty, p.sec_type, p.symbol, p.mult) == (100, "STK", SYM, 1)


# 2. C5 regression: a round trip at constant S loses exactly spread + commissions + reg fees
async def test_round_trip_cost_identity(sim_broker, fake_market, sim_clock, live_cfg):
    c, h = live_cfg.costs, fake_market.h
    await sim_broker.place_shares(100, None, "hedge")
    tick(sim_broker, sim_clock)
    await sim_broker.place_shares(-100, None, "hedge")
    tick(sim_broker, sim_clock)
    expected = 2 * 100 * h + 2 * commission(100, c) + reg_fees(-100, fake_market.S - h, c)
    assert -cash(sim_broker) == approx(expected, abs=1e-9)
    assert await sim_broker.positions() == {}  # flat lot is dropped, not kept at qty 0


# 3. the limit is audit only; a cancel before the next quote means no fill
async def test_limit_is_audit_only_and_cancel_before_quote(sim_broker, sim_clock):
    o = await sim_broker.place_shares(100, 590.0, "hedge")  # far below the ask
    tick(sim_broker, sim_clock)
    assert o.state is OrderState.FILLED and sim_broker.fills[0].price == approx(600.005)
    assert sim_broker.orders[o.id].limit == 590.0
    o2 = await sim_broker.place_shares(-50, 610.0, "hedge")
    assert (await sim_broker.cancel(o2)).state is OrderState.CANCELLED
    tick(sim_broker, sim_clock)
    assert len(sim_broker.fills) == 1 and sim_broker.open_orders() == []
    assert (await sim_broker.cancel(o2)).state is OrderState.CANCELLED  # idempotent


# 4. scripted outcomes on share orders
@pytest.mark.parametrize(
    ("out", "quotes", "state", "filled"),
    [
        ("reject", 1, OrderState.REJECTED, 0),
        ("timeout", 3, OrderState.SUBMITTED, 0),
        ("partial:40", 1, OrderState.PARTIAL, 40),
    ],
)
async def test_scripted_share_outcomes(sim_broker, sim_clock, out, quotes, state, filled):
    sim_broker.script = lambda o: out
    o = await sim_broker.place_shares(100, None, "hedge")
    for _ in range(quotes):
        tick(sim_broker, sim_clock)
    assert o.state is state and o.filled == filled
    assert [f.qty for f in sim_broker.fills] == ([40] if filled else [])
    assert o.is_open == (state is not OrderState.REJECTED)
    if state is OrderState.REJECTED:
        assert o.error == "sim reject" and sim_broker.open_orders() == []


# 5. BAG honours its limit, never better than the limit, never worse than the touch
async def test_straddle_fill_prices_and_ledger(sim_broker, sim_clock, live_cfg):
    c = live_cfg.costs
    call, put = atm(sim_clock.now())
    net_mid = call.mid + put.mid
    o = await sim_broker.place_straddle(call, put, 5, "BUY", net_mid + 0.06, "entry")
    assert o.state is OrderState.FILLED and o.sec_type == "BAG" and o.con_id == 0
    assert o.leg_fills == {1001: 5, 1002: 5} and o.filled == 5
    assert o.avg_price == approx(net_mid + 0.06)
    fc, fp = sim_broker.fills
    assert (fc.con_id, fc.qty, fp.con_id, fp.qty) == (1001, 5, 1002, 5)
    assert fc.price == approx(call.mid + 0.03) and fp.price == approx(put.mid + 0.03)
    assert fc.commission + fp.commission == approx(
        max(c.option_min_commission, c.option_commission * 10) + c.option_fees_per_contract * 10
    )
    assert len({f.exec_id for f in sim_broker.fills}) == 2
    pos = await sim_broker.positions()
    assert pos[1001].qty == 5 and pos[1001].avg_cost == approx(call.mid + 0.03)
    assert (pos[1002].sec_type, pos[1002].right, pos[1002].strike, pos[1002].mult) == (
        "OPT",
        "P",
        put.strike,
        100,
    )
    assert pos[1002].expiry == put.expiry and pos[1002].symbol == SYM

    sim_broker.fills.clear()  # a limit far above the touch is capped at the touch
    await sim_broker.place_straddle(call, put, 1, "BUY", net_mid + 1.0, "entry")
    assert [f.price for f in sim_broker.fills] == approx([call.mid + 0.06, put.mid + 0.06])

    sim_broker.fills.clear()  # a SELL at net_mid - h fills at mid - h/2 per leg and closes
    o3 = await sim_broker.place_straddle(call, put, 6, "SELL", net_mid - 0.06, "exit")
    assert o3.leg_fills == {1001: -6, 1002: -6} and o3.avg_price == approx(net_mid - 0.06)
    assert [f.price for f in sim_broker.fills] == approx([call.mid - 0.03, put.mid - 0.03])
    assert await sim_broker.positions() == {}


# 6. a partial BAG fills k on BOTH legs; wait() cancels the remainder and keeps the fills
async def test_partial_straddle_never_naked_and_wait_cancels_rest(sim_broker, sim_clock):
    sim_broker.script = lambda o: "partial:2"
    call, put = atm(sim_clock.now())
    o = await sim_broker.place_straddle(call, put, 5, "BUY", call.mid + put.mid + 0.06, "entry")
    assert o.state is OrderState.PARTIAL and o.filled == 2
    assert o.leg_fills == {1001: 2, 1002: 2} and [f.qty for f in sim_broker.fills] == [2, 2]
    assert sim_broker.open_orders() == [o]
    assert (await sim_broker.wait(o, 30)).state is OrderState.CANCELLED
    assert o.filled == 2 and len(sim_broker.fills) == 2 and sim_broker.open_orders() == []
    pos = await sim_broker.positions()
    assert pos[1001].qty == pos[1002].qty == 2


async def test_wrong_side_straddle_limit_rests(sim_broker, sim_clock):
    """A BUY below the mid (SELL above) never fills better than the market: it rests."""
    call, put = atm(sim_clock.now())
    net_mid = call.mid + put.mid
    o = await sim_broker.place_straddle(call, put, 1, "BUY", net_mid - 1.0, "entry")
    assert o.state is OrderState.SUBMITTED and not sim_broker.fills and o.leg_fills == {}
    assert (await sim_broker.wait(o, 30)).state is OrderState.CANCELLED
    o2 = await sim_broker.place_straddle(call, put, 1, "SELL", net_mid + 1.0, "exit")
    assert o2.state is OrderState.SUBMITTED and not sim_broker.fills


@pytest.mark.parametrize("out", ["reject", "timeout"])
async def test_scripted_straddle_reject_and_timeout(sim_broker, sim_clock, out):
    sim_broker.script = lambda o: out
    call, put = atm(sim_clock.now())
    o = await sim_broker.place_straddle(call, put, 5, "BUY", call.mid + put.mid + 0.06, "entry")
    assert not sim_broker.fills and o.leg_fills == {} and o.filled == 0
    if out == "reject":
        assert o.state is OrderState.REJECTED and o.error == "sim reject" and not o.is_open
    else:
        assert o.state is OrderState.SUBMITTED and sim_broker.open_orders() == [o]
        assert (await sim_broker.wait(o, 30)).state is OrderState.CANCELLED
    assert await sim_broker.positions() == {}


# 7. positions(): broker truth (recovered state) merged with what the SimBroker filled
async def test_positions_merge_market_and_sim_ledger(sim_broker, fake_market, sim_clock):
    cid = fake_market.stock_con_id(SYM)
    call, put = atm(sim_clock.now())
    fake_market.pos = {
        cid: Position(cid, SYM, "STK", -37, 598.0),
        1001: Position(1001, SYM, "OPT", 5, call.mid, "C", call.strike, call.expiry, 100),
        1002: Position(1002, SYM, "OPT", 5, put.mid, "P", put.strike, put.expiry, 100),
    }
    assert (await sim_broker.positions()) == fake_market.pos  # nothing filled yet
    await sim_broker.place_shares(100, None, "hedge")
    tick(sim_broker, sim_clock)
    p = (await sim_broker.positions())[cid]
    assert p.qty == 63 and p.avg_cost == approx(600.005)  # flipped: remainder opens at the fill
    # dry-run exit of HELD (recovered) legs: the sim SELL nets them out of positions()
    await sim_broker.place_straddle(call, put, 5, "SELL", call.mid + put.mid - 0.06, "exit")
    await sim_broker.place_shares(-63, None, "flatten")
    tick(sim_broker, sim_clock)
    assert await sim_broker.positions() == {}
    assert len(fake_market.pos) == 3  # the market's own truth is never mutated


# 8. whatif is free for a long combo; cancel_all counts what it cancelled
async def test_whatif_and_cancel_all(sim_broker, sim_clock):
    call, put = atm(sim_clock.now())
    assert await sim_broker.whatif_straddle(call, put, 5, call.mid + put.mid) == 0.0
    sim_broker.script = lambda o: "timeout"
    await sim_broker.place_shares(100, None, "hedge")
    await sim_broker.place_shares(-20, None, "hedge")
    bag = await sim_broker.place_straddle(call, put, 5, "BUY", call.mid + put.mid, "entry")
    assert len(sim_broker.open_orders()) == 3
    assert await sim_broker.cancel_all() == 3
    assert sim_broker.open_orders() == [] and bag.state is OrderState.CANCELLED
    assert await sim_broker.cancel_all() == 0
    tick(sim_broker, sim_clock)
    assert not sim_broker.fills


# 9. ordering: a quote seen BEFORE place_shares never fills it; None quotes fill nothing
async def test_no_fill_inside_place_shares(sim_broker, fake_market, sim_clock):
    tick(sim_broker, sim_clock)
    o = await sim_broker.place_shares(100, None, "hedge")
    assert not sim_broker.fills and o.state is OrderState.SUBMITTED
    fake_market.quote_none = True
    assert tick(sim_broker, sim_clock) is None and not sim_broker.fills
    fake_market.quote_none, fake_market.market_data_type = False, 3
    assert sim_broker.quote(fake_market.stock_con_id(SYM), sim_clock.now(), True) is None
    assert not sim_broker.fills and o.is_open
    assert tick(sim_broker, sim_clock) is not None and o.state is OrderState.FILLED
    assert sim_broker.fills[0].ts == sim_clock.now()  # stamped with the fill quote's now


async def test_orders_need_a_bound_symbol_and_nonzero_qty(fake_market, sim_clock, live_cfg):
    b = SimBroker(fake_market, live_cfg.costs, sim_clock)
    with pytest.raises(RuntimeError):
        await b.place_shares(100, None, "hedge")
    assert await b.stream_stock(SYM) == fake_market.stock_con_id(SYM)
    assert fake_market.calls == [("stream_stock", SYM)]
    with pytest.raises(ValueError):
        await b.place_shares(0, None, "hedge")
    assert b.connected and b.data_ok and b.market_data_type == 1 and not b.needs_reconcile
    assert (await b.historical_bars(SYM)).empty
    assert await b.account() is fake_market.acct and not b.commission_updates
