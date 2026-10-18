"""src/runner.py lifecycle against SimBroker + FakeMarket + SimClock (no sleeps); the
`ib_runner` tests run the real Broker over FakeIB. Market open on 2026-03-10; the SimClock is
advanced by hand between ticks."""

import asyncio
from dataclasses import replace
from datetime import date, datetime, timedelta

import pytest
from ib_async import AccountValue, Bag, ComboLeg, Option, OrderStatus, Stock, Trade
from ib_async import Order as IbOrder
from ib_async import Position as IbPosition
from loguru import logger

from src import broker as broker_mod
from src import bs
from src.broker import Broker
from src.clock import NY, T_years, aware, expiry_at, session_date
from src.config import Broker as BrokerCfg
from src.runner import TRANSITIONS, IllegalTransition, Runner, State
from src.sim_broker import SimBroker
from src.types import Fill, IbkrGreeks, OrderState, Position
from tests.conftest import FakeMarket, make_chain
from tests.fake_ib import FakeIB, greeks, make_ticker

SYM, STOCK = "SPY", 756733
R, Q = 0.04, 0.0099
approx = pytest.approx


def seed(market, now, n=5, shares=-37, dte=30, iv=0.15, S=600.0, iv_quote=True):
    """Broker truth: `shares` + an n-lot straddle at the chain mids; leg quotes for both conIds."""
    chain = make_chain(S, dte, iv, now)
    call, put = (next(q for q in chain if q.con_id == cid) for cid in (1001, 1002))
    market.leg_quotes = {
        q.con_id: q if iv_quote else replace(q, iv=None, delta=None) for q in (call, put)
    }
    market.pos = {STOCK: Position(STOCK, SYM, "STK", shares, S)} if shares else {}
    for q in (call, put):
        market.pos[q.con_id] = Position(
            q.con_id, SYM, "OPT", n, q.mid, q.right, q.strike, q.expiry, 100
        )
    return call, put


async def started(runner, market, **kw):
    r = runner(market, **kw)
    await r.start()
    assert r.state is State.RUNNING
    return r


async def ticks(r, sim_clock, n, step=60):
    row = None
    for _ in range(n):
        sim_clock.advance(step)
        row = await r.tick()
    return row


def session_status(store, sid):
    return store._db.execute("SELECT status FROM sessions WHERE id=?", (sid,)).fetchone()[0]


def warnings_during():
    """loguru WARNING+ messages, as a list filled while the returned sink is attached."""
    logged = []
    return logged, logger.add(logged.append, level="WARNING")


def ib_runner(live_cfg, sim_clock, tmp_store, positions, **loop):
    """Runner(--execute) over the real Broker over FakeIB: no clock skew, the stock qualifies as
    STOCK, no sleeps (the test moves the SimClock)."""
    fake = FakeIB()
    fake.current_time, fake.next_con_id = aware(sim_clock.now()), STOCK
    fake.positions_list = positions
    fake.account_values = [
        AccountValue("DU1", k, v, "USD", "")
        for k, v in (
            ("NetLiquidation", "100000"),
            ("ExcessLiquidity", "60000"),
            ("BuyingPower", "4e5"),
        )
    ]
    loop = {"warmup_sessions": 1, **loop}
    cfg = live_cfg.model_copy(update={"loop": live_cfg.loop.model_copy(update=loop)})
    b = Broker(BrokerCfg(), sim_clock, readonly=False, ib=fake)

    async def no_sleep(s):
        pass

    return Runner(cfg, b, tmp_store, sim_clock, execute=True, mode="paper", sleep=no_sleep), fake


def quote_now(fake, sim_clock):
    """The SPY ticker quotes now; IBKR's clock agrees with ours (a reconnect re-measures it)."""
    t, now = fake.tickers[STOCK], aware(sim_clock.now())
    t.bid, t.ask, t.last, t.time, t.marketDataType = 599.9, 600.1, 600.0, now, 1
    fake.current_time = now


async def one_pass(r):
    """One Runner.loop iteration (the tick and its exception mapping), then the loop returns."""
    tick = Runner.tick.__get__(r)

    async def once():
        try:
            return await tick()
        finally:
            r._stop.set()

    r.tick, _ = once, r._stop.clear()
    await r.loop()


# 1. STOP during start -> no streams, no orders, STOPPED
async def test_stop_during_start(runner, fake_market, tmp_store):
    r = runner(fake_market)
    r.stop("test")
    await r.run()
    assert r.state is State.STOPPED
    assert not [c for c in fake_market.calls if "stream" in str(c)]
    assert r.broker.orders == {}
    assert tmp_store._db.execute("SELECT COUNT(*) FROM orders").fetchone()[0] == 0


# 2. dry-run: every hedge is a SimBroker order; the market never sees a placeOrder
async def test_dry_run_orders_stay_in_sim_broker(runner, fake_market, sim_clock):
    seed(fake_market, sim_clock.now())
    r = await started(runner, fake_market)
    for i in range(20):
        fake_market.set_spot(608.0 if i % 4 < 2 else 600.0)
        await ticks(r, sim_clock, 1)
    assert r.state is State.RUNNING and len(r.broker.orders) >= 2
    assert all(o.sec_type == "STK" for o in r.broker.orders.values())
    assert "placeOrder" not in str(fake_market.calls)
    assert r.book.shares != -37  # the sim fills reached the book


# 3. one working share order: the second tick sends nothing and cancels after fill_timeout_s
async def test_one_working_hedge(runner, fake_market, sim_clock):
    seed(fake_market, sim_clock.now())
    fake_market.set_spot(610.0)
    r = await started(runner, fake_market)
    r.broker.script = lambda o: "timeout"
    row = await ticks(r, sim_clock, 1)
    assert row["u_sent"] and len(r.broker.orders) == 1
    (o,) = r.broker.orders.values()
    row = await ticks(r, sim_clock, 1)  # +60 s > fill_timeout_s 20
    assert row["reason"] == "working_order" and row["u_sent"] is None
    assert o.state is OrderState.CANCELLED and len(r.broker.orders) == 1
    await ticks(r, sim_clock, 1)  # cancel-and-replace
    assert len(r.broker.orders) == 2 and r.working is not None


# 4. stale quote -> DEGRADED (fills drained, reconcile runs) -> HALT after stale_halt_s
async def test_stale_degraded_then_halt(runner, fake_market, sim_clock):
    seed(fake_market, sim_clock.now())
    r = await started(runner, fake_market)
    fake_market.quote_ts_lag_s = 20
    r.broker.fills.append(Fill("x1", "o0", STOCK, 10, 600.0, sim_clock.now(), 0.35, "sim"))
    row = await ticks(r, sim_clock, 1)
    assert r.state is State.DEGRADED and row["reason"] == "stale" and row["S"] is None
    assert r.book.shares == -27
    row = await ticks(r, sim_clock, 1, step=300)  # reconcile: broker says -37, book -27
    assert r.book.shares == -37 and row["reason"] == "reconcile_adopt"
    r.broker.script = lambda o: "timeout"
    o = await r.broker.place_shares(10, 600.0, "hedge")
    for _ in range(12):
        if r.state is State.HALTED:
            break
        await ticks(r, sim_clock, 1)
    assert r.state is State.HALTED and r.risk.halt_reason == "stale quote"
    assert o.state is OrderState.CANCELLED  # cancel_all


# 4b. socket down -> no quote (as Broker.quote) -> DEGRADED, reason disconnected, not stale
async def test_disconnected_degraded_then_recover(runner, fake_market, sim_clock):
    seed(fake_market, sim_clock.now())
    r = await started(runner, fake_market)
    fake_market.connected = False
    row = await ticks(r, sim_clock, 1)
    assert r.state is State.DEGRADED and row["reason"] == "disconnected" and row["S"] is None
    fake_market.connected = True
    row = await ticks(r, sim_clock, 1)
    assert r.state is State.RUNNING and row["reason"] == "in_band" and row["S"] == approx(600.0)


# 4c. ib_async raises ConnectionError on any request while the socket is down: a disconnect
# spanning a reconcile boundary or cutting an order wait is DEGRADED, never HALT("exception");
# the reconnect re-quotes, cancels what the outage left resting and runs RUNNING again
async def test_disconnect_across_reconcile_and_order_wait(
    live_cfg, sim_clock, tmp_store, monkeypatch
):
    monkeypatch.setattr(broker_mod, "BACKOFF_S", [0.01])
    stk = Stock(SYM, "SMART", "USD", conId=STOCK)
    r, fake = ib_runner(
        live_cfg, sim_clock, tmp_store, [IbPosition("DU1", stk, -37, 600.0)], fill_timeout_s=0.05
    )
    b = r.broker
    await r.start()
    fake.on_cancel = lambda t: fake.emit_status(t, "Cancelled")

    async def reconnect():
        fake.fail_connects = 0
        for _ in range(300):
            if b.connected:
                return
            await asyncio.sleep(0.01)
        raise AssertionError("no reconnect")

    try:
        # 1. the socket dies; ib_async raises before its disconnect event is handled
        sim_clock.advance(60)
        quote_now(fake, sim_clock)
        fake.connected, fake.fail_connects = False, 10**6
        await one_pass(r)
        assert r.state is State.DEGRADED and not fake.placed
        # 2. the event arrives; still down at the 300 s reconcile boundary
        fake.emit_disconnect()
        sim_clock.advance(240)
        await one_pass(r)
        assert r.state is State.DEGRADED
        assert tmp_store.last_decisions(r.session_id, 1)[0]["reason"] == "disconnected"
        # 3. back up: recover; the shares-only book is flattened and the link drops mid-wait
        quote_now(fake, sim_clock)
        await reconnect()
        place = fake.placeOrder

        def place_then_drop(contract, order):
            fake.placeOrder, fake.fail_connects = place, 10**6
            asyncio.get_running_loop().call_soon(fake.emit_disconnect)  # during the wait
            return place(contract, order)

        fake.placeOrder = place_then_drop
        await one_pass(r)
        assert r.state is State.DEGRADED and len(fake.placed) == 1
        (cut,) = [o for o in b.orders.values() if o.reason == "flatten"]
        assert cut.is_open  # still resting at TWS for all we know
        # 4. back up again: the order the outage left working is cancelled, trading resumes
        quote_now(fake, sim_clock)
        await reconnect()
        await one_pass(r)
        assert r.state is State.RUNNING and not r.risk.halted
        assert cut.state is OrderState.CANCELLED and b.needs_reconcile is False
    finally:
        await b.disconnect()


# 4d. start(): a rebound order that fills while it is being cancelled is inside the positions
# snapshot, so its queued fill is dropped, not booked a second time
async def test_rebound_fill_during_start_cancel_booked_once(live_cfg, sim_clock, tmp_store):
    stk = Stock(SYM, "SMART", "USD", conId=STOCK)
    r, fake = ib_runner(live_cfg, sim_clock, tmp_store, [IbPosition("DU1", stk, -200, 600.0)])
    order = IbOrder(
        orderId=7, permId=70, action="BUY", totalQuantity=100, orderType="LMT", lmtPrice=600.0
    )
    order.orderRef = "qmpc:previous"
    fake.trades.append(Trade(stk, order, OrderStatus(orderId=7, status="Submitted")))

    def fills_instead(t):  # TWS: too late to cancel, it filled
        fake.emit_exec(t, STOCK, "BOT", 100, 600.0, "race1")
        fake.emit_status(t, "Filled", filled=100, avg=600.0)
        fake.positions_list = [IbPosition("DU1", stk, -100, 600.0)]

    fake.on_cancel = fills_instead
    try:
        await r.start()
        assert r.state is State.RUNNING and r.book.shares == -100 and not r.broker.fills
        r._drain()
        assert r.book.shares == -100  # was 0: the race fill booked on top of the snapshot
    finally:
        await r.broker.disconnect()


def ib_straddle(live_cfg, sim_clock, tmp_store, dte):
    """ib_runner holding 5 x (1001, 1002) at TWS; quote_legs() stamps live leg quotes now."""
    chain = make_chain(600.0, dte, 0.15, sim_clock.now())
    legs = [next(q for q in chain if q.con_id == c) for c in (1001, 1002)]
    opts = {
        q.con_id: Option(
            SYM,
            q.expiry.strftime("%Y%m%d"),
            q.strike,
            q.right,
            "SMART",
            conId=q.con_id,
            multiplier="100",
            tradingClass=SYM,
        )
        for q in legs
    }
    pos = [IbPosition("DU1", opts[q.con_id], 5, q.mid * 100) for q in legs]
    r, fake = ib_runner(live_cfg, sim_clock, tmp_store, pos, straddle_timeout_s=0.05)
    fake.known.update(opts)

    def quote_legs():
        now = aware(sim_clock.now())
        for q in legs:
            t = fake.tickers.setdefault(q.con_id, make_ticker(opts[q.con_id]))
            t.bid, t.ask, t.time, t.marketDataType = q.mid - 0.02, q.mid + 0.02, now, 1
            d = 0.5 if q.right == "C" else -0.5
            t.modelGreeks = greeks(impliedVol=0.15, delta=d, gamma=0.05, vega=0.1, theta=-0.5)

    quote_legs()
    return r, fake, opts, quote_legs


# 4e. an exit the outage cut filled at TWS with no execDetails (ib_async drops re-fetched
# executions): the recover tick reconciles before deciding, so it never sells the legs again
async def test_recover_reconciles_before_deciding(live_cfg, sim_clock, tmp_store, monkeypatch):
    monkeypatch.setattr(broker_mod, "BACKOFF_S", [0.01])
    r, fake, _, quote_legs = ib_straddle(live_cfg, sim_clock, tmp_store, dte=0)  # hard_exit
    b = r.broker
    await r.start()
    try:
        sim_clock.advance(60)
        quote_now(fake, sim_clock)
        quote_legs()
        place = fake.placeOrder

        def place_then_drop(contract, order):
            fake.placeOrder, fake.fail_connects = place, 10**6
            asyncio.get_running_loop().call_soon(fake.emit_disconnect)  # during the wait
            return place(contract, order)

        fake.placeOrder = place_then_drop
        await one_pass(r)
        assert r.state is State.DEGRADED and len(fake.placed) == 1
        (tr,) = fake.placed  # TWS filled it while we were away; nothing tells the Broker
        tr.orderStatus.status, tr.orderStatus.filled = "Filled", 5
        fake.positions_list = []
        fake.on_cancel = lambda t: fake.emit_error(t.order.orderId, 10148, "state: Filled")
        sim_clock.advance(60)
        quote_now(fake, sim_clock)
        fake.fail_connects = 0
        for _ in range(300):
            if b.connected:
                break
            await asyncio.sleep(0.01)
        quote_legs()
        await one_pass(r)
        assert len(fake.placed) == 1  # was 4 more SELL BAGs for legs no longer held
        assert r.state is State.HALTED and r.risk.halt_reason.startswith("leg mismatch")
    finally:
        await b.disconnect()


# 4f. the vol warm-up is optional: a link drop during it is DEGRADED later, never a halt
async def test_link_drop_during_warmup_continues(live_cfg, sim_clock, tmp_store):
    r, fake = ib_runner(live_cfg, sim_clock, tmp_store, [], warmup_sessions=3)

    async def drop(*a, **k):
        fake.fail_connects = 10**6
        fake.emit_disconnect()
        raise ConnectionError("Socket disconnect")

    fake.reqHistoricalDataAsync = drop
    try:
        await r.start()
        assert r.state is State.RUNNING and r.vol.sessions() == 0
        sim_clock.advance(60)
        await one_pass(r)
        assert r.state is State.DEGRADED and not r.risk.halted
    finally:
        await r.broker.disconnect()


# 5. delayed data in execute: full row with S, refused by risk with "delayed", no order
async def test_delayed_data_in_execute(runner, sim_clock):
    market = FakeMarket(sim_clock, md_type=3)
    seed(market, sim_clock.now())
    market.set_spot(610.0)
    r = await started(runner, market, execute=True)
    row = await ticks(r, sim_clock, 1)
    assert row["risk_result"] == "delayed" and row["S"] == 610.0 and row["md_type"] == 3
    assert r.broker.orders == {} and r.state is State.RUNNING
    assert row["fill_source"] == "broker"


# 5b. delayed data in execute opens nothing either (dry-run unchanged); live SPY with delayed
# option quotes (no OPRA subscription) is delayed too: the BAG is priced from the legs
@pytest.mark.parametrize(
    ("execute", "stock_md", "opt_md"), [(True, 3, 1), (False, 3, 1), (True, 1, 3), (False, 1, 3)]
)
async def test_delayed_data_blocks_entry_in_execute(runner, sim_clock, execute, stock_md, opt_md):
    market = FakeMarket(sim_clock, md_type=stock_md)
    chain = make_chain(600.0, 30, 0.15, sim_clock.now())
    market.chain_quotes = [replace(q, market_data_type=opt_md) for q in chain]
    r = await started(runner, market, execute=execute, force_open=True)
    row = await ticks(r, sim_clock, 1)
    bags = [o.reason for o in r.broker.orders.values() if o.sec_type == "BAG"]
    assert bags == ([] if execute else ["entry"])
    assert row["reason"] == ("delayed" if execute else "entry") and row["md_type"] == stock_md


# 6. entry with --force-open: sized from the account, BAG filled, legs streamed
async def test_force_open_entry(runner, fake_market, sim_clock):
    now = sim_clock.now()
    fake_market.chain_quotes = make_chain(600.0, 30, 0.15, now)
    r = await started(runner, fake_market, force_open=True)
    row = await ticks(r, sim_clock, 1)
    assert r.contracts_held == 7 and len(r.book.legs) == 2  # floor(0.15*100k / 2050) = 7
    assert r.book.premium_paid > 0 and row["reason"] == "entry"
    assert [c for c in fake_market.calls if c[0] == "stream_leg"] == [
        ("stream_leg", 1001),
        ("stream_leg", 1002),
    ]
    assert row["campaign"] == "held" and row["force_open"] == 1
    (o,) = r.broker.orders.values()
    assert o.sec_type == "BAG" and o.state is OrderState.FILLED and o.qty == 7


# 7. scripted entries: partial -> k units accepted; reject -> flat; unequal legs -> halt
async def test_partial_entry_accepts_complete_units(runner, fake_market, sim_clock):
    fake_market.chain_quotes = make_chain(600.0, 30, 0.15, sim_clock.now())
    r = await started(runner, fake_market, force_open=True)
    r.broker.script = lambda o: "partial:2"
    await ticks(r, sim_clock, 1)
    assert r.contracts_held == 2 and r.state is State.RUNNING
    assert r.book.premium_paid == approx(next(iter(r.broker.orders.values())).avg_price * 200)


async def test_rejected_entry_leaves_book_flat(runner, fake_market, sim_clock):
    fake_market.chain_quotes = make_chain(600.0, 30, 0.15, sim_clock.now())
    r = await started(runner, fake_market, force_open=True)
    r.broker.script = lambda o: "reject"
    row = await ticks(r, sim_clock, 1)
    assert r.book.is_flat() and row["reason"] == "entry_unfilled" and r.state is State.RUNNING


async def test_unequal_legs_halt_for_unwind(runner, fake_market, sim_clock, monkeypatch):
    fake_market.chain_quotes = make_chain(600.0, 30, 0.15, sim_clock.now())

    async def call_only(self, call, put, n, action, limit, reason):
        o = self._new(0, "BAG", action, n, limit, reason)
        tmpl = Position(call.con_id, SYM, "OPT", 0, 0.0, call.right, call.strike, call.expiry, 100)
        self._fill(o, tmpl, n, call.mid, 1.0, o.created)
        o.leg_fills[call.con_id], o.filled, o.state = n, n, OrderState.PARTIAL
        return o

    monkeypatch.setattr(SimBroker, "place_straddle", call_only)
    r = await started(runner, fake_market, force_open=True)
    await ticks(r, sim_clock, 1)
    assert r.state is State.HALTED and "unwind" in r.risk.halt_reason


async def test_filled_bag_with_trailing_leg_report(runner, fake_market, sim_clock, monkeypatch):
    """TWS can report orderStatus Filled before the last leg's execDetails: trust the combo-level
    count, book the trailing leg when it arrives, never halt a complete straddle."""
    fake_market.chain_quotes = make_chain(600.0, 30, 0.15, sim_clock.now())
    orig, late = SimBroker.place_straddle, []

    async def put_trails(self, call, put, n, action, limit, reason):
        o = await orig(self, call, put, n, action, limit, reason)
        late[:] = [self.fills.pop(), o.leg_fills.pop(put.con_id)]  # put report still in flight
        return o

    monkeypatch.setattr(SimBroker, "place_straddle", put_trails)
    r = await started(runner, fake_market, force_open=True)
    row = await ticks(r, sim_clock, 1)
    assert r.state is State.RUNNING and len(r.book.legs) == 1 and r.contracts_held == 0
    assert row["u_sent"] is None and r.book.premium_paid > 0
    r.broker.fills.append(late[0])
    await ticks(r, sim_clock, 1)
    assert r.contracts_held == 7 and r.state is State.RUNNING


# 8. DTE <= roll_dte: roll when the candidate's gate is OPEN, else exit "roll_refused"
async def test_roll_into_later_expiry_when_gate_open(runner, fake_market, sim_clock):
    now = sim_clock.now()
    old_call, _ = seed(fake_market, now, dte=6, shares=-150)
    fake_market.chain_quotes = make_chain(600.0, 30, 0.15, now, call_id=2001, put_id=2002)
    r = await started(runner, fake_market)
    r.vol.forecast_sigma = lambda horizon_sessions=1: 0.25
    row = await ticks(r, sim_clock, 1)
    assert {leg.con_id for leg in r.book.legs} == {2001, 2002}
    assert all(leg.expiry > old_call.expiry for leg in r.book.legs)
    assert r.book.shares == 0 and r.contracts_held == 7 and r.gate.state == "OPEN"
    assert row["campaign"] == "held" and r.state is State.RUNNING and row["reason"] == "roll"
    # no hedge on the old book's Greeks after the roll: the only share order is the flatten
    assert row["u_sent"] is None
    assert [o.reason for o in r.broker.orders.values() if o.sec_type == "STK"] == ["flatten"]


# the re-entry obeys the entry guards: ex-dividend tomorrow, or delayed data in execute
@pytest.mark.parametrize("why", ["ex_div", "delayed"])
async def test_roll_reentry_obeys_entry_guards(runner, sim_clock, why):
    market = FakeMarket(sim_clock, md_type=3 if why == "delayed" else 1)
    now = sim_clock.now()
    seed(market, now, dte=6, shares=-150)
    market.chain_quotes = make_chain(600.0, 30, 0.15, now, call_id=2001, put_id=2002)
    r = await started(runner, market, execute=why == "delayed")
    ins = r.cfg.instrument.model_copy(update={"ex_dividend_dates": [date(2026, 3, 11)]})
    if why == "ex_div":
        r.cfg = r.cfg.model_copy(update={"instrument": ins})
    r.vol.forecast_sigma = lambda horizon_sessions=1: 0.25
    row = await ticks(r, sim_clock, 1)
    assert r.book.is_flat() and r.gate.reason == "roll_refused" and r.state is State.RUNNING
    assert [o.reason for o in r.broker.orders.values() if o.sec_type == "BAG"] == ["roll"]
    assert row["reason"] == "exit:roll_refused"


async def test_roll_refused_exits(runner, fake_market, sim_clock):
    now = sim_clock.now()
    seed(fake_market, now, dte=6)
    fake_market.chain_quotes = make_chain(600.0, 30, 0.15, now, call_id=2001, put_id=2002)
    r = await started(runner, fake_market)
    await ticks(r, sim_clock, 1)
    assert r.book.is_flat() and r.gate.reason == "roll_refused" and r.state is State.RUNNING
    assert ("unstream", 1001) in fake_market.calls and ("unstream", 1002) in fake_market.calls


# 9. hard exit at DTE <= 1 regardless of the gate
async def test_hard_exit(runner, fake_market, sim_clock):
    seed(fake_market, sim_clock.now(), dte=0)  # expires 16:00 today
    r = await started(runner, fake_market)
    row = await ticks(r, sim_clock, 1)
    assert r.book.is_flat() and r.gate.reason == "hard_exit" and r.state is State.RUNNING
    assert row["reason"] == "exit:hard_exit"


# 10. gate exit_confirmed: exit on the honest gate; --force-open holds (no exit/re-enter churn)
@pytest.mark.parametrize("force_open", [False, True])
async def test_exit_confirmed_honours_force_open(runner, fake_market, sim_clock, force_open):
    seed(fake_market, sim_clock.now())
    r = await started(runner, fake_market, force_open=force_open)
    r.gate.state, r.gate.reason = "CLOSED", "exit_confirmed"
    r.gate.last_eval_session = session_date(sim_clock.now())  # not due: the tick keeps it
    await ticks(r, sim_clock, 1)
    bags = [o.reason for o in r.broker.orders.values() if o.sec_type == "BAG"]
    assert r.book.is_flat() is not force_open and bags == ([] if force_open else ["exit_confirmed"])


def campaign_store(store, call, put, now, stock_fills):
    """An earlier session's broker fills: the straddle at its mids, then the hedge trades."""
    sid = store.start_session("paper", "{}", now - timedelta(days=20))
    t = now - timedelta(days=20)
    for i, q in enumerate((call, put)):
        store.save_fill(sid, Fill(f"leg{i}", "bag", q.con_id, 5, q.mid, t, 3.25, "broker"))
    for i, (qty, px) in enumerate(stock_fills):
        t += timedelta(hours=1)
        store.save_fill(sid, Fill(f"h{i}", f"o{i}", STOCK, qty, px, t, 0.35, "broker"))


# 11. exit on book loss since entry
async def test_exit_on_max_loss(runner, fake_market, sim_clock, tmp_store):
    call, put = seed(fake_market, sim_clock.now())
    fake_market.leg_quotes = {
        q.con_id: replace(q, bid=0.4 * q.mid - 0.01, ask=0.4 * q.mid + 0.01) for q in (call, put)
    }
    campaign_store(tmp_store, call, put, sim_clock.now(), [(-37, 600.0)])  # no hedge gains
    r = await started(runner, fake_market)
    assert r._entry_basis == 0.0  # the stored ledger was adopted: loss since the entry
    row = await ticks(r, sim_clock, 1)
    assert r.book.is_flat() and r.gate.reason == "max_loss" and r.state is State.RUNNING
    assert row["reason"] == "exit:max_loss"


# 11b. restart mid-campaign: the stored ledger (realized hedge gains) keeps max_loss honest
async def test_restart_replays_campaign_ledger(runner, fake_market, sim_clock, tmp_store):
    call, put = seed(fake_market, sim_clock.now())  # 5 straddles bought at these mids, -37 shares
    fake_market.leg_quotes = {  # down 55% since entry: past max_loss without the hedge gains
        q.con_id: replace(q, bid=0.45 * q.mid - 0.01, ask=0.45 * q.mid + 0.01) for q in (call, put)
    }
    campaign_store(
        tmp_store, call, put, sim_clock.now(), [(-100, 600.0), (100, 590.0), (-37, 600.0)]
    )
    r = await started(runner, fake_market)
    assert r.book.realized == approx(1000.0) and r.book.commissions == approx(4 * 3.25 / 2 + 1.05)
    assert r.book.premium_paid == approx((call.mid + put.mid) * 500)
    row = await ticks(r, sim_clock, 1)
    assert r.contracts_held == 5 and r.broker.orders.get("max_loss") is None
    assert not [o for o in r.broker.orders.values() if o.sec_type == "BAG"]
    assert row["book_pnl"] > -0.5 * r.book.premium_paid


async def test_restart_with_inconsistent_ledger_measures_loss_from_restart(
    runner, fake_market, sim_clock, tmp_store
):
    call, put = seed(fake_market, sim_clock.now())
    fake_market.leg_quotes = {
        q.con_id: replace(q, bid=0.45 * q.mid - 0.01, ask=0.45 * q.mid + 0.01) for q in (call, put)
    }
    campaign_store(tmp_store, call, put, sim_clock.now(), [(-50, 600.0)])  # broker says -37
    logged, sink = warnings_during()
    try:
        r = await started(runner, fake_market)
        row = await ticks(r, sim_clock, 1)
    finally:
        logger.remove(sink)
    assert r.book.realized == 0 and r._entry_basis == approx(row["book_pnl"])
    assert not [o for o in r.broker.orders.values() if o.sec_type == "BAG"]
    assert len([m for m in logged if "do not rebuild the broker position" in m]) == 1


async def test_no_hedge_on_the_tick_the_straddle_closed(runner, fake_market, sim_clock, tmp_store):
    """Exit filled but the share flatten found no quote: the step-4 Greeks describe a dead
    straddle, so no hedge this tick; the next tick flattens the naked shares."""
    call, put = seed(fake_market, sim_clock.now(), shares=-30)
    fake_market.set_spot(606.0)
    fake_market.leg_quotes = {
        q.con_id: replace(q, bid=0.4 * q.mid - 0.01, ask=0.4 * q.mid + 0.01) for q in (call, put)
    }
    campaign_store(tmp_store, call, put, sim_clock.now(), [(-30, 600.0)])
    r = await started(runner, fake_market, execute=True)
    orig, calls = fake_market.quote, []

    def first_only(*a):
        calls.append(1)
        return orig(*a) if len(calls) == 1 else None

    fake_market.quote = first_only
    row = await ticks(r, sim_clock, 1)
    assert [(o.sec_type, o.reason) for o in r.broker.orders.values()] == [("BAG", "max_loss")]
    assert row["u_sent"] is None and not r.book.legs and r.book.shares == -30
    fake_market.quote = orig
    await ticks(r, sim_clock, 1)
    assert [o.reason for o in r.broker.orders.values()] == ["max_loss", "flatten"]
    assert r.book.is_flat() and r.state is State.RUNNING


# 11. reconcile: adopt small share drift, halt on large drift or (execute) leg mismatch
async def test_reconcile_adopts_small_drift(runner, fake_market, sim_clock):
    seed(fake_market, sim_clock.now())
    r = await started(runner, fake_market)
    fake_market.pos[STOCK] = Position(STOCK, SYM, "STK", -40, 600.0)
    row = await ticks(r, sim_clock, 1, step=300)
    assert r.book.shares == -40 and row["reason"] == "reconcile_adopt"
    assert r.state is State.RUNNING


async def test_dry_run_unpaired_legs_observe_without_reconcile(runner, fake_market, sim_clock):
    call, _ = seed(fake_market, sim_clock.now(), shares=-150)
    fake_market.pos[3001] = Position(3001, SYM, "OPT", 1, 5.0, "C", 610.0, call.expiry, 100)
    r = await started(runner, fake_market)
    assert r.book.is_flat()
    row = await ticks(r, sim_clock, 1, step=300)  # reconcile due: the -150 is not drift
    assert r.state is State.RUNNING and r.book.is_flat() and row["reason"] != "reconcile_adopt"


async def test_reconcile_halts_on_large_drift(runner, fake_market, sim_clock):
    seed(fake_market, sim_clock.now())
    r = await started(runner, fake_market)
    fake_market.pos[STOCK] = Position(STOCK, SYM, "STK", -117, 600.0)
    await ticks(r, sim_clock, 1, step=300)
    assert r.state is State.HALTED and "drift" in r.risk.halt_reason


@pytest.mark.parametrize(("execute", "halted"), [(True, True), (False, False)])
async def test_reconcile_leg_mismatch(runner, sim_clock, execute, halted):
    market = FakeMarket(sim_clock)
    call, _ = seed(market, sim_clock.now())
    r = await started(runner, market, execute=execute)
    market.pos[3001] = Position(3001, SYM, "OPT", 1, 5.0, "C", 610.0, call.expiry, 100)
    await ticks(r, sim_clock, 1, step=300)
    assert (r.state is State.HALTED) is halted
    assert len(r.book.legs) == 2  # three legs are not a straddle: nothing adopted


async def test_broker_reconcile_flag_reconciles_once(runner, fake_market, sim_clock):
    """Broker sets needs_reconcile after a reconnect: one positions() round-trip, flag reset."""
    seed(fake_market, sim_clock.now())
    r = await started(runner, fake_market)
    n, orig = 0, fake_market.positions

    async def counted():
        nonlocal n
        n += 1
        return await orig()

    fake_market.positions = counted
    fake_market.needs_reconcile = True
    await ticks(r, sim_clock, 2)
    assert n == 1 and r.broker.needs_reconcile is False and fake_market.needs_reconcile is False


# 12. close-window hedge to centre, once per session
async def test_close_window_hedge_once(runner, fake_market, sim_clock):
    sim_clock.set(datetime(2026, 3, 10, 15, 56, tzinfo=NY))
    now = sim_clock.now()
    call, _ = seed(fake_market, now, shares=0)
    T = T_years(call.expiry, now)
    d_opt = sum(bs.greeks(600.0, 600.0, T, R, Q, 0.15, rt).delta for rt in "CP") * 500
    fake_market.pos[STOCK] = Position(STOCK, SYM, "STK", 10 - round(d_opt), 600.0)
    r = await started(runner, fake_market)
    row = await r.tick()
    assert row["reason"] == "close_hedge" and row["u_sent"] == -10
    assert r.hedged_for_close_session is None  # counts once it fills
    fake_market.set_spot(606.0)  # after the fill |delta| != 0 again
    row = await ticks(r, sim_clock, 1)
    assert r.hedged_for_close_session == date(2026, 3, 10)
    assert abs(row["delta_net"]) > 1 and row["u_sent"] is None and len(r.broker.orders) == 1


async def test_close_hedge_timed_out_is_resent(runner, fake_market, sim_clock):
    sim_clock.set(datetime(2026, 3, 10, 15, 56, tzinfo=NY))
    now = sim_clock.now()
    call, _ = seed(fake_market, now, shares=0)
    T = T_years(call.expiry, now)
    d_opt = sum(bs.greeks(600.0, 600.0, T, R, Q, 0.15, rt).delta for rt in "CP") * 500
    fake_market.pos[STOCK] = Position(STOCK, SYM, "STK", 10 - round(d_opt), 600.0)
    r = await started(runner, fake_market)
    r.broker.script = lambda o: "timeout"
    row = await r.tick()
    assert row["reason"] == "close_hedge" and row["u_sent"] == -10
    row = await ticks(r, sim_clock, 1)  # 15:57: older than fill_timeout_s -> cancelled
    assert row["reason"] == "working_order" and r.hedged_for_close_session is None
    row = await ticks(r, sim_clock, 1)  # 15:58: re-decided from the live delta, not skipped
    assert row["reason"] == "close_hedge" and row["u_sent"] == -10 and len(r.broker.orders) == 2


# 13. a blocked trade still sleeps the full interval (no busy loop)
async def test_blocked_trade_still_sleeps(runner, fake_market, sim_clock, monkeypatch):
    seed(fake_market, sim_clock.now())
    fake_market.set_spot(610.0)
    r = runner(fake_market)
    monkeypatch.setattr(r.risk, "check_hedge", lambda *a, **k: (0, "book_bound"))
    n, orig = 0, r.tick

    async def counted():
        nonlocal n
        n += 1
        row = await orig()
        if n == 3:
            r.stop("done")
        return row

    monkeypatch.setattr(r, "tick", counted)
    await r.run()
    assert n == 3 and r.sleeps == [60.0] * 3 and r.state is State.STOPPED
    assert r.broker.orders == {}


# 14. a decisions row every tick with the three shadow hedges
async def test_decision_row_every_tick(runner, fake_market, sim_clock, tmp_store):
    seed(fake_market, sim_clock.now())
    r = await started(runner, fake_market)
    await ticks(r, sim_clock, 5)
    rows = tmp_store.last_decisions(r.session_id, 5)
    assert len(rows) == 5
    for row in rows:
        assert all(isinstance(row[k], int) for k in ("band_u", "qp_u", "naive_u"))
        assert row["force_open"] == 0 and row["fill_source"] == "sim" and row["S"] == 600.0


# 15. TRANSITIONS: every entry, plus three illegal moves
@pytest.mark.parametrize(("edge", "target"), list(TRANSITIONS.items()))
def test_transitions(runner, fake_market, edge, target):
    r = runner(fake_market)
    r.state = edge[0]
    r.fire(edge[1])
    assert r.state is target


@pytest.mark.parametrize(
    ("state", "event"),
    [(State.STOPPED, "stop"), (State.HALTED, "recover"), (State.RUNNING, "started")],
)
def test_illegal_transitions(runner, fake_market, state, event):
    r = runner(fake_market)
    r.state = state
    with pytest.raises(IllegalTransition):
        r.fire(event)
    assert r.state is state


# 16. flatten(): straddle + shares -> SimBroker positions empty, 0; closed market -> 2
async def test_flatten(runner, fake_market, sim_clock):
    seed(fake_market, sim_clock.now())
    r = runner(fake_market, execute=True)
    assert await r.flatten() == 0
    pos = await r.broker.positions()
    assert not [p for p in pos.values() if p.symbol == SYM]
    assert r.book.is_flat() and r.state is State.STARTING
    kinds = sorted((o.sec_type, o.action) for o in r.broker.orders.values())
    assert kinds == [("BAG", "SELL"), ("STK", "BUY")]


# a re-adopted SELL BAG of a crashed session that fills while flatten cancels it: the snapshot
# is taken after the cancels, so flatten sells nothing more (it sold the legs a second time)
async def test_flatten_rebound_bag_fill_during_cancel(live_cfg, sim_clock, tmp_store):
    r, fake, opts, _ = ib_straddle(live_cfg, sim_clock, tmp_store, dte=30)
    fake.tickers[STOCK] = make_ticker(Stock(SYM, "SMART", "USD", conId=STOCK))
    quote_now(fake, sim_clock)
    bag = Bag(
        symbol=SYM,
        exchange="SMART",
        currency="USD",
        comboLegs=[ComboLeg(conId=c, ratio=1, action="SELL", exchange="SMART") for c in opts],
    )
    o = IbOrder(
        orderId=7, permId=70, action="SELL", totalQuantity=5, orderType="LMT", lmtPrice=20.0
    )
    o.orderRef = "qmpc:crashed1"
    rebound = Trade(bag, o, OrderStatus(orderId=7, status="Submitted"))
    fake.trades.append(rebound)

    def fills_on_cancel(t):  # too late: the crashed session's exit fills as we cancel it
        for i, c in enumerate(opts):
            fake.emit_exec(t, opts[c], "SLD", 5, 10.0, f"race{i}")
        fake.emit_status(t, "Filled", filled=5, avg=20.0)
        fake.positions_list = []

    fake.on_cancel = fills_on_cancel
    assert await r.flatten() == 0
    assert fake.placed == [] and r.book.is_flat() and not r.risk.halted


async def test_flatten_keeps_hedge_when_straddle_will_not_close(runner, fake_market, sim_clock):
    seed(fake_market, sim_clock.now())
    fake_market.leg_quotes = {}  # no usable leg quote: _exit refuses, shares must stay
    r = runner(fake_market, execute=True)
    assert await r.flatten() == 1
    assert r.broker.orders == {} and r.book.shares == -37 and r.contracts_held == 5


async def test_flatten_refuses_closed_market(runner, sim_clock, tmp_store):
    sim_clock.set(datetime(2026, 3, 10, 20, 0, tzinfo=NY))
    market = FakeMarket(sim_clock)
    seed(market, sim_clock.now())
    r = runner(market, execute=True)
    assert await r.flatten() == 2 and r.broker.orders == {}
    assert session_status(tmp_store, r.session_id) == "flatten" and "disconnect" in market.calls


async def test_flatten_refuses_unpaired_legs(runner, fake_market, sim_clock, tmp_store):
    call, _ = seed(fake_market, sim_clock.now())
    fake_market.pos[3001] = Position(3001, SYM, "OPT", 1, 5.0, "C", 610.0, call.expiry, 100)
    r = runner(fake_market, execute=True)
    assert await r.flatten() == 1 and r.broker.orders == {}
    assert session_status(tmp_store, r.session_id) == "flatten"  # not left "running"


async def test_flatten_cancelled_mid_exit_cancels_our_orders(
    runner, fake_market, sim_clock, tmp_store, monkeypatch
):
    """Ctrl-C / SIGTERM (run_paper cancels the task) while the SELL BAG rests: cancelled."""
    seed(fake_market, sim_clock.now())
    r = runner(fake_market, execute=True)
    r.broker.script = lambda o: "timeout"
    resting = asyncio.Event()

    async def hang(order, timeout_s):
        resting.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(r.broker, "wait", hang)
    task = asyncio.create_task(r.flatten())
    await resting.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    (bag,) = r.broker.orders.values()
    assert (bag.sec_type, bag.action, bag.state) == ("BAG", "SELL", OrderState.CANCELLED)
    assert (
        session_status(tmp_store, r.session_id) == "flatten" and "disconnect" in fake_market.calls
    )


# 17. an exception inside tick halts, run() returns, the session is stored as halted
async def test_exception_in_tick_halts(runner, fake_market, sim_clock, tmp_store, monkeypatch):
    seed(fake_market, sim_clock.now())

    def boom(*a, **k):
        raise RuntimeError("boom")

    monkeypatch.setattr("src.runner.hedge_all", boom)
    r = runner(fake_market)
    await r.run()
    assert r.state is State.HALTED and r.risk.halt_reason == "exception"
    assert session_status(tmp_store, r.session_id) == "halted"
    assert "disconnect" in fake_market.calls


# 18. restart IV seed: recovered legs (iv=0) take implied_vol of the first leg quote mid
async def test_restart_iv_seed(runner, fake_market, sim_clock):
    now = sim_clock.now()
    call, put = seed(fake_market, now, shares=-150, iv_quote=False)
    r = await started(runner, fake_market)
    assert all(leg.iv == 0 for leg in r.book.legs)
    row = await r.tick()
    for leg, q in zip(r.book.legs, (call, put), strict=True):
        want = bs.implied_vol(q.mid, 600.0, q.strike, T_years(q.expiry, now), R, Q, q.right)
        assert leg.iv == approx(want, abs=1e-6)
    assert row["reason"] != "greeks_nan" and row["u_sent"] and len(r.broker.orders) == 1


async def test_iv_frozen_warning_once_per_10_min_per_leg(runner, fake_market, sim_clock):
    seed(fake_market, sim_clock.now())  # leg quotes, but no tick-106 Greeks
    r = await started(runner, fake_market)
    logged, sink = warnings_during()
    try:
        await ticks(r, sim_clock, 11)  # +60 s .. +660 s
        fake_market.greeks = {
            cid: IbkrGreeks(0.5, 0.02, None, None, 0.15, 600.0, sim_clock.now())
            for cid in (1001, 1002)
        }
        await ticks(r, sim_clock, 11)
    finally:
        logger.remove(sink)
    frozen = [m for m in logged if "IV frozen" in m]
    assert len(frozen) == 4 and sum("leg 1001" in m for m in frozen) == 2


async def test_restart_without_leg_quotes_is_greeks_nan(runner, fake_market, sim_clock):
    seed(fake_market, sim_clock.now(), shares=-150)
    fake_market.leg_quotes = {}
    r = await started(runner, fake_market)
    row = await r.tick()
    assert row["reason"] == "greeks_nan" and row["u_sent"] is None
    assert r.broker.orders == {} and r.state is State.RUNNING


# 19. --execute over the real Broker(FakeIB): the broker persists its own order at placement and
# the runner saves it again; the second save must not raise (it did: UNIQUE on orders)
async def test_execute_hedge_persists_once(live_cfg, sim_clock, tmp_store):
    now = sim_clock.now()
    exp = expiry_at(session_date(now) + timedelta(days=38))
    T = T_years(exp, now)
    stk = Stock(SYM, "SMART", "USD", conId=STOCK)
    legs = {
        rt: Option(SYM, exp.strftime("%Y%m%d"), 600, rt, "SMART", multiplier="100", conId=cid)
        for rt, cid in (("C", 1001), ("P", 1002))
    }
    fake = FakeIB()
    fake.current_time, fake.next_con_id = (
        aware(now),
        STOCK,
    )  # no clock skew; stock qualifies as STOCK
    fake.positions_list = [IbPosition("DU1", stk, -200, 600.0)] + [
        IbPosition("DU1", c, 5, 100 * bs.price(600.0, 600.0, T, R, Q, 0.15, rt))
        for rt, c in legs.items()
    ]
    fake.account_values = [
        AccountValue("DU1", k, v, "USD", "")
        for k, v in (
            ("NetLiquidation", "100000"),
            ("ExcessLiquidity", "60000"),
            ("BuyingPower", "4e5"),
        )
    ]
    cfg = live_cfg.model_copy(
        update={"loop": live_cfg.loop.model_copy(update={"warmup_sessions": 1})}
    )
    b = Broker(BrokerCfg(), sim_clock, readonly=False, ib=fake)
    r = Runner(cfg, b, tmp_store, sim_clock, execute=True, mode="paper")
    await r.start()
    assert r.state is State.RUNNING and r.book.shares == -200 and r.contracts_held == 5
    t = fake.tickers[STOCK]
    t.bid, t.ask, t.last, t.time, t.marketDataType = 599.9, 600.1, 600.0, aware(now), 1
    for rt, c in legs.items():
        fake.tickers[c.conId].time = aware(now)
        fake.tickers[c.conId].modelGreeks = greeks(
            impliedVol=0.15, delta=0.52 if rt == "C" else -0.48, gamma=0.02, undPrice=600.0
        )
    row = await r.tick()
    assert row["u_sent"] and len(fake.placed) == 1 and r.state is State.RUNNING
    assert tmp_store._db.execute("SELECT COUNT(*) FROM orders").fetchone()[0] == 1
    await b.disconnect()
