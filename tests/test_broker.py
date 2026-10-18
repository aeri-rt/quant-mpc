"""Broker against FakeIB: conId positions, order states, BAG + whatIf, cancel-on-timeout,
re-query, rebind by orderRef, reconnect. No TWS involved."""

import asyncio
import subprocess
import sys
from datetime import UTC, date, datetime, timedelta
from math import nan
from pathlib import Path
from unittest.mock import Mock

import pytest
from ib_async import (
    AccountValue,
    Bag,
    BarData,
    Contract,
    Option,
    OptionChain,
    OrderStatus,
    Stock,
    Trade,
)
from ib_async import Order as IbOrder
from ib_async import Position as IbPosition

from src import broker as broker_mod
from src.broker import Broker, BrokerDisconnected
from src.clock import NY, SimClock, aware, expiry_at, session_date
from src.config import Broker as BrokerCfg
from src.store import Store
from src.types import OptionQuote, OrderState
from tests.fake_ib import FakeIB, greeks, make_ticker

ROOT = Path(__file__).resolve().parent.parent
STK = Stock("SPY", "SMART", "USD", conId=756733)
CALL = Option(
    "SPY", "20261120", 600, "C", "SMART", multiplier="100", conId=1001, tradingClass="SPY"
)
PUT = Option("SPY", "20261120", 600, "P", "SMART", multiplier="100", conId=1002, tradingClass="SPY")
EXP = expiry_at(date(2026, 11, 20))


def oq(c, bid, ask, now):
    return OptionQuote(c.conId, c.right, c.strike, EXP, bid, ask, 0.15, 0.5, aware(now), 1)


def chain_params(expirations, strikes, trading_class="SPY"):
    return OptionChain(
        exchange="SMART",
        underlyingConId=756733,
        tradingClass=trading_class,
        multiplier="100",
        expirations=[d.strftime("%Y%m%d") for d in expirations],
        strikes=list(strikes),
    )


@pytest.fixture
def now():
    return datetime(2026, 3, 15, 10, 0, tzinfo=NY)  # a Sunday: Apr 17 (3rd Friday) is 33 d out


@pytest.fixture
def fake(now):
    f = FakeIB()
    f.current_time = aware(now)
    f.known = {1001: CALL, 1002: PUT}
    return f


@pytest.fixture
async def broker(fake, now):
    b = Broker(BrokerCfg(), SimClock(now), readonly=False, ib=fake)
    await b.connect()
    yield b
    await b.disconnect()


async def stream_stock(broker, fake, now, age=1.0, md=1):
    cid = await broker.stream_stock("SPY")
    t = fake.tickers[cid]
    t.bid, t.ask, t.last, t.marketDataType = 599.9, 600.1, 600.0, md
    t.time = aware(now) - timedelta(seconds=age)
    return cid


async def placed(broker, fake, status="Submitted"):
    o = await broker.place_shares(100, 600.01, "hedge")
    tr = fake.placed[-1]
    fake.emit_status(tr, status)
    return o, tr


# ---- 1. positions ----------------------------------------------------------------------
@pytest.mark.parametrize("reverse", [False, True])
async def test_positions_keyed_by_con_id(broker, fake, reverse):
    rows = [
        IbPosition("DU1", STK, -37, 600.0),
        IbPosition("DU1", CALL, 5, 2800.0),
        IbPosition("DU1", PUT, 5, 2800.0),
        IbPosition("DU1", Option("SPY", "20261120", 610, "C", "SMART", conId=1003), 0, 0.0),
    ]
    fake.positions_list = rows[::-1] if reverse else rows
    pos = await broker.positions()
    assert set(pos) == {756733, 1001, 1002}
    assert (pos[756733].qty, pos[756733].avg_cost, pos[756733].mult) == (-37, 600.0, 1)
    call = pos[1001]
    assert (call.sec_type, call.qty, call.avg_cost, call.mult) == ("OPT", 5, 28.0, 100)
    assert (call.right, call.strike, call.expiry) == ("C", 600.0, EXP)
    assert pos[1002].right == "P"


# ---- 2. status map ---------------------------------------------------------------------
@pytest.mark.parametrize(
    "status,filled,expected",
    [
        ("PendingSubmit", 0, OrderState.SUBMITTED),
        ("ApiPending", 0, OrderState.SUBMITTED),
        ("PreSubmitted", 0, OrderState.SUBMITTED),
        ("Submitted", 0, OrderState.SUBMITTED),
        ("Submitted", 40, OrderState.PARTIAL),
        ("PendingCancel", 0, OrderState.SUBMITTED),
        ("PendingCancel", 40, OrderState.PARTIAL),
        ("ValidationError", 0, OrderState.SUBMITTED),
        ("ApiUpdate", 0, OrderState.SUBMITTED),
        ("Filled", 100, OrderState.FILLED),
        ("Cancelled", 0, OrderState.CANCELLED),
        ("ApiCancelled", 0, OrderState.CANCELLED),
        ("Inactive", 0, OrderState.INACTIVE),
        ("SomethingNew", 0, OrderState.UNKNOWN),
    ],
)
async def test_status_map(broker, fake, now, status, filled, expected):
    await stream_stock(broker, fake, now)
    o = await broker.place_shares(100, 600.01, "hedge")
    tr = fake.placed[-1]
    tr.order.permId = 777
    fake.emit_status(tr, status, filled=filled, avg=600.0 if filled else 0.0)
    assert o.state is expected
    assert o.filled == filled and o.perm_id == 777
    assert o.is_open == (expected in (OrderState.SUBMITTED, OrderState.PARTIAL, OrderState.UNKNOWN))


# ---- 3. errors -------------------------------------------------------------------------
async def test_error_202_cancels(broker, fake, now):
    await stream_stock(broker, fake, now)
    o, tr = await placed(broker, fake)
    fake.emit_error(tr.order.orderId, 202, "Order Canceled - reason:")
    assert o.state is OrderState.CANCELLED


async def test_error_201_rejects_once_and_sticks(broker, fake, now):
    hook = Mock()
    broker.on_reject(hook)
    await stream_stock(broker, fake, now)
    o, tr = await placed(broker, fake)
    fake.emit_status(tr, "Cancelled")  # ib_async synthesises this before the error event
    fake.emit_error(tr.order.orderId, 201, "Order rejected - reason: insufficient margin")
    fake.emit_status(tr, "Inactive")  # TWS trails the reject with a status
    assert o.state is OrderState.REJECTED and "margin" in o.error
    hook.assert_called_once_with(now)
    assert broker.open_orders() == []


async def test_10148_after_cancel_requeries_to_filled(broker, fake, now):
    """'cannot be cancelled, state: Filled' is the broker saying it filled, not a reject."""
    hook = Mock()
    broker.on_reject(hook)
    await stream_stock(broker, fake, now)
    o, tr = await placed(broker, fake)

    def on_cancel(t):
        fake.emit_status(t, "Cancelled")  # ib_async wrongly marks it Cancelled first
        fake.emit_error(t.order.orderId, 10148, "OrderId 1 ... can not be cancelled, state: Filled")
        t.orderStatus.status, t.orderStatus.filled = "Filled", 100  # the broker's real answer

    fake.on_cancel = on_cancel
    o = await broker.wait(o, 0.01)
    assert fake.cancelled == [tr.order.orderId]
    assert o.state is OrderState.FILLED and o.filled == 100
    hook.assert_not_called()


async def test_filled_before_10148_stays_filled(broker, fake, now):
    await stream_stock(broker, fake, now)
    o, tr = await placed(broker, fake)
    fake.on_cancel = lambda t: None
    task = asyncio.ensure_future(broker.cancel(o))
    await asyncio.sleep(0)
    fake.emit_status(tr, "Filled", filled=100, avg=600.0)
    fake.emit_error(tr.order.orderId, 10148, "state: Filled")
    assert (await task).state is OrderState.FILLED


async def test_inactive_is_cancelled_and_requeried(broker, fake, now):
    await stream_stock(broker, fake, now)
    o, tr = await placed(broker, fake, "Inactive")
    assert o.state is OrderState.INACTIVE and not o.is_open
    fake.on_cancel = lambda t: fake.emit_status(t, "Cancelled")
    o = await broker.wait(o, 1.0)
    assert fake.cancelled == [tr.order.orderId] and o.state is OrderState.CANCELLED


async def test_connectivity_codes(broker, fake):
    fake.emit_error(-1, 1100, "Connectivity between IB and TWS has been lost")
    assert broker.data_ok is False
    fake.emit_error(-1, 2104, "Market data farm connection is OK")
    assert broker.data_ok is True
    for code in (1101, 1102):  # "connectivity restored" ends the 1100 outage too
        fake.emit_error(-1, 1100, "Connectivity between IB and TWS has been lost")
        fake.emit_error(-1, code, "Connectivity restored")
        assert broker.data_ok is True
    assert broker.needs_reconcile is True  # 1102: data maintained -> reconcile


async def test_requests_while_link_down_raise_broker_disconnected(broker, fake, now):
    """ib_async's Client.send raises ConnectionError while the socket is down (before the
    disconnect event is handled): every request path says BrokerDisconnected instead."""
    await stream_stock(broker, fake, now)
    o, tr = await placed(broker, fake)
    call, put = oq(CALL, 10.0, 10.4, now), oq(PUT, 10.1, 10.5, now)
    fake.connected = False  # socket gone, broker.connected not yet reset
    for req in (
        broker.positions(),
        broker.account(),
        broker.chain("SPY", 30, now),
        broker.whatif_straddle(call, put, 1, 20.6),
        broker.place_shares(10, 600.01, "hedge"),
        broker.place_straddle(call, put, 1, "SELL", 20.0, "exit"),
        broker.cancel(o),
        broker.wait(o, 0.01),
        broker.historical_bars("SPY", now, 1, 60),
        broker.stream_leg(1002),
    ):
        with pytest.raises(BrokerDisconnected):
            await req
    assert fake.placed == [tr] and o.is_open  # nothing reached TWS, nothing marked rejected
    fake.connected = True

    async def boom(*_):
        raise OSError("not a disconnect")

    fake.reqPositionsAsync = boom
    with pytest.raises(OSError, match="not a disconnect"):  # link up: unchanged
        await broker.positions()


# ---- 4. wait() timeout -----------------------------------------------------------------
async def test_wait_timeout_cancels(broker, fake, now):
    await stream_stock(broker, fake, now)
    o, tr = await placed(broker, fake)
    fake.on_cancel = lambda t: fake.emit_status(t, "Cancelled")
    o = await broker.wait(o, 0.01)
    assert fake.cancelled == [tr.order.orderId] and o.state is OrderState.CANCELLED


async def test_requery_counts_executions_after_reset(broker, fake, now, monkeypatch):
    """ib_async forgets the Trade on reconnect but re-fetches executions: they say filled."""
    monkeypatch.setattr(broker_mod, "ACK_S", 0.01)
    await stream_stock(broker, fake, now)
    o = await broker.place_shares(100, 600.01, "hedge")
    tr = fake.placed[-1]
    tr.order.permId = 777
    fake.emit_status(tr, "Submitted")
    fake.emit_exec(tr, STK, "BOT", 100, 600.01, "e1")
    fake.trades.remove(tr)  # what wrapper.reset() does to the cached Trade
    o = await broker.wait(o, 0.01)
    assert o.state is OrderState.FILLED and o.filled == 100


async def test_wait_returns_on_fill_event(broker, fake, now):
    await stream_stock(broker, fake, now)
    o, tr = await placed(broker, fake)
    asyncio.get_running_loop().call_later(0.01, fake.emit_status, tr, "Filled", 100, 600.01)
    o = await broker.wait(o, 5.0)
    assert o.state is OrderState.FILLED and o.avg_price == 600.01 and fake.cancelled == []


# ---- 5. place_shares -------------------------------------------------------------------
async def test_place_shares_limit_only(broker, fake, now):
    cid = await stream_stock(broker, fake, now)
    with pytest.raises(ValueError, match="limit"):
        await broker.place_shares(100, None, "hedge")
    assert fake.placed == []
    o = await broker.place_shares(100, 600.01, "hedge")
    ib_order = fake.placed[-1].order
    assert (ib_order.action, ib_order.totalQuantity, ib_order.lmtPrice) == ("BUY", 100, 600.01)
    assert (ib_order.orderType, ib_order.tif, ib_order.outsideRth) == ("LMT", "DAY", False)
    assert ib_order.orderRef.startswith("qmpc:")
    assert o.broker_order_id == ib_order.orderId and o.state is OrderState.SUBMITTED
    assert (o.con_id, o.sec_type, o.action, o.qty, o.reason) == (cid, "STK", "BUY", 100, "hedge")
    assert (await broker.place_shares(-50, 599.99, "hedge")).action == "SELL"


async def test_place_shares_disconnected_or_failed(broker, fake, now):
    hook = Mock()
    broker.on_reject(hook)
    await stream_stock(broker, fake, now)
    broker.connected = False
    with pytest.raises(BrokerDisconnected):
        await broker.place_shares(100, 600.01, "hedge")
    broker.connected = True

    def boom(contract, order):
        raise ConnectionError("Not connected")

    fake.placeOrder = boom
    o = await broker.place_shares(100, 600.01, "hedge")
    assert o.state is OrderState.REJECTED and o.error == "Not connected"
    hook.assert_called_once()


# ---- 6. straddle: BAG + whatIf + leg fills + commission --------------------------------
async def test_straddle_bag_whatif_fills_commission(broker, fake, now):
    store = Store(":memory:")
    sid = store.start_session("paper", "{}", aware(now))
    spy = Mock(wraps=store.set_commission)
    store.set_commission = spy
    broker.bind_session(store, sid)
    await stream_stock(broker, fake, now)
    call, put = oq(CALL, 10.0, 10.4, now), oq(PUT, 10.1, 10.5, now)

    assert await broker.whatif_straddle(call, put, 5, 20.6) == 1234.5
    bag, order = fake.whatifs[0]
    assert isinstance(bag, Bag) and order.totalQuantity == 5 and fake.placed == []

    o = await broker.place_straddle(call, put, 5, "BUY", 20.6, "entry")
    tr = fake.placed[-1]
    assert isinstance(tr.contract, Bag)
    assert [(c.conId, c.ratio, c.action, c.exchange) for c in tr.contract.comboLegs] == [
        (1001, 1, "BUY", "SMART"),
        (1002, 1, "BUY", "SMART"),
    ]
    assert (tr.contract.symbol, tr.contract.exchange, tr.contract.currency) == (
        "SPY",
        "SMART",
        "USD",
    )
    io = tr.order
    assert (io.action, io.totalQuantity, io.lmtPrice, io.tif) == ("BUY", 5, 20.6, "DAY")
    assert io.orderRef == "qmpc:" + sid[:8]
    assert (o.con_id, o.sec_type, o.state) == (0, "BAG", OrderState.SUBMITTED)

    fake.emit_status(tr, "Submitted")
    fake.emit_exec(tr, Contract(secType="BAG", symbol="SPY"), "BOT", 5, 20.6, "e0")  # combo-level
    fake.emit_exec(tr, 1001, "BOT", 5, 10.2, "e1")
    fake.emit_exec(tr, 1002, "BOT", 5, 10.4, "e2")
    fake.emit_status(tr, "Filled", filled=5, avg=20.6)
    assert o.leg_fills == {1001: 5, 1002: 5} and o.state is OrderState.FILLED
    fills = list(broker.fills)
    assert [(f.con_id, f.qty, f.price, f.commission, f.source) for f in fills] == [
        (1001, 5, 10.2, None, "broker"),
        (1002, 5, 10.4, None, "broker"),
    ]
    assert all(f.order_id == o.id and f.ts == aware(now) for f in fills)

    fake.emit_commission("e1", 0.65)
    assert broker.commission_updates[-1] == ("e1", 0.65)
    spy.assert_called_once_with("e1", 0.65)
    fake.emit_commission("e0", 7.5)  # combo-level report: no leg fill to patch, never forwarded
    assert list(broker.commission_updates) == [("e1", 0.65)]
    spy.assert_called_once()
    assert broker.leg_quote(1001, now) is None  # remembered contract, but not streamed yet
    await broker.stream_leg(1001)
    assert fake.subscribed[-1] == (1001, "106")


# ---- 7. Greeks sentinels ---------------------------------------------------------------
async def test_leg_greeks_sentinels(broker, fake, now):
    await broker.stream_leg(1001)
    t = fake.tickers[1001]
    t.time = aware(now)
    t.modelGreeks = greeks(
        impliedVol=0.15, delta=0.5, gamma=0.02, vega=-2, theta=-2, undPrice=600.0
    )
    g = broker.leg_greeks(1001, 600.0, now)
    assert (g.delta, g.gamma, g.vega, g.theta, g.iv, g.S_ref) == (
        0.5,
        0.02,
        None,
        None,
        0.15,
        600.0,
    )
    t.modelGreeks = greeks(impliedVol=0.15, delta=-2, gamma=0.02, undPrice=600.0)
    assert broker.leg_greeks(1001, 600.0, now) is None
    t.modelGreeks = None
    assert broker.leg_greeks(1001, 600.0, now) is None
    assert broker.leg_greeks(1002, 600.0, now) is None


# ---- 8. quote() / leg_quote() ----------------------------------------------------------
async def test_quote_freshness_and_skew(broker, fake, now):
    cid = await broker.stream_stock("SPY")
    t = fake.tickers[cid]
    assert broker.quote(cid, now, False) is None  # ticker.time None
    t.bid, t.ask, t.last = 599.9, 600.1, 600.0
    t.time = aware(now) - timedelta(seconds=20)
    assert broker.quote(cid, now, False) is None  # stale
    t.time = aware(now) - timedelta(seconds=5)
    q = broker.quote(cid, now, False)
    assert q.mid == pytest.approx(600.0) and q.ts == t.time and q.market_data_type == 1
    t.marketDataType = 3
    assert broker.quote(cid, now, True) is None
    assert broker.quote(cid, now, False).market_data_type == 3
    broker.clock_offset_s = 12.0
    assert broker.quote(cid, now, False) is None  # readonly=False refuses skewed quotes
    broker.readonly = True
    assert broker.quote(cid, now, False) is not None  # skew is not staleness
    broker.connected = False
    assert broker.quote(cid, now, False) is None
    assert broker.quote(999, now, False) is None


async def test_leg_quote(broker, fake, now):
    await broker.stream_leg(1001)
    t = fake.tickers[1001]
    t.bid, t.ask, t.time = 2.10, 2.20, aware(now)
    t.modelGreeks = greeks(impliedVol=0.15, delta=0.5)
    q = broker.leg_quote(1001, now)
    assert (q.con_id, q.right, q.strike, q.expiry) == (1001, "C", 600.0, EXP)
    assert q.mid == pytest.approx(2.15) and q.iv == 0.15 and q.bid == 2.10
    t.bid = 0.0
    assert broker.leg_quote(1001, now) is None  # one-sided
    assert broker.leg_quote(1002, now) is None  # not streamed
    await broker.unstream(1001)
    assert fake.unsubscribed == [1001] and broker.leg_quote(1001, now) is None


async def test_stream_leg_requalifies_recovered_contract(broker, fake):
    """TWS sends positions without an exchange; subscribing with '' is error 321."""
    bare = Option("SPY", "20261120", 600, "C", conId=1001)
    fake.positions_list = [IbPosition("DU1", bare, 5, 2800.0)]
    await broker.positions()
    assert broker._contracts[1001].exchange == ""
    await broker.stream_leg(1001)
    assert fake.subscribed[-1] == (1001, "106") and broker._contracts[1001].exchange == "SMART"


# ---- 9. chain() ------------------------------------------------------------------------
TODAY = date(2026, 3, 15)
D = {
    k: TODAY + timedelta(days=k) for k in (2, 5, 8, 12, 29, 30, 31, 33, 40, 61)
}  # 5=Mar20, 33=Apr17


def snapshot_factory(now):
    def make(c):
        g = greeks(impliedVol=0.2, delta=0.5 if c.right == "C" else -0.5)
        return make_ticker(c, 2.0, 2.1, time=aware(now), greeks=g)

    return make


async def chain_setup(broker, fake, now, expirations, strikes=range(595, 606)):
    await stream_stock(broker, fake, now)
    fake.sec_def = [chain_params(expirations, strikes)]
    fake.snapshot = snapshot_factory(now)


async def test_chain_prefers_monthly_within_5_days(broker, fake, now):
    await chain_setup(broker, fake, now, [D[5], D[29], D[30], D[31], D[33], D[61]])
    out = await broker.chain("SPY", 30, now)
    assert len(out) == 12 and {q.expiry for q in out} == {expiry_at(D[33])}
    assert {q.strike for q in out} == {597.0, 598.0, 599.0, 600.0, 601.0, 602.0}
    assert {q.right for q in out} == {"C", "P"} and all(q.iv == 0.2 for q in out)
    assert all(q.delta == (0.5 if q.right == "C" else -0.5) for q in out)
    assert broker.leg_quote(out[0].con_id, now) is None  # chain snapshots are not streams


async def test_chain_falls_back_to_nearest_daily(broker, fake, now):
    await chain_setup(broker, fake, now, [D[30], D[40], D[61]])
    out = await broker.chain("SPY", 30, now)
    assert {q.expiry for q in out} == {expiry_at(D[30])}


async def test_chain_prefer_monthly_off_takes_nearest(broker, fake, now):
    await chain_setup(broker, fake, now, [D[5], D[29], D[30], D[31], D[33], D[61]])
    broker.prefer_monthly = False  # position.prefer_monthly, set by the scripts at construction
    out = await broker.chain("SPY", 30, now)
    assert {q.expiry for q in out} == {expiry_at(D[30])}


async def test_chain_holiday_monthly_is_the_session_before(broker, fake, now):
    # Juneteenth 2026 is June's 3rd Friday: the monthly lists Thu Jun 18 (clock.pick_expiry)
    jun18, jun22 = date(2026, 6, 18), date(2026, 6, 22)
    await chain_setup(broker, fake, now, [jun18, jun22])
    out = await broker.chain("SPY", (jun22 - TODAY).days - 1, now)  # Jun 22 is the nearer daily
    assert {q.expiry for q in out} == {expiry_at(jun18)}


async def test_chain_never_below_roll_dte_plus_one(broker, fake, now):
    await chain_setup(broker, fake, now, [D[2], D[5], D[8], D[12]])
    out = await broker.chain("SPY", 3, now)
    assert {q.expiry for q in out} == {expiry_at(D[8])}
    assert all((session_date(q.expiry) - TODAY).days >= broker.roll_dte + 1 for q in out)
    await chain_setup(broker, fake, now, [D[2], D[5]])
    assert await broker.chain("SPY", 3, now) == []


async def test_chain_drops_unqualifiable_strike(broker, fake, now):
    await chain_setup(broker, fake, now, [D[33]])
    fake.unqualifiable = {597.0}
    out = await broker.chain("SPY", 30, now)
    assert len(out) == 10 and 597.0 not in {q.strike for q in out}


async def test_chain_prefers_symbol_trading_class(broker, fake, now):
    await chain_setup(broker, fake, now, [D[33]])
    fake.sec_def = [chain_params([D[30]], [600], "SPYW"), chain_params([D[33]], [600], "SPY")]
    out = await broker.chain("SPY", 30, now)
    assert {q.expiry for q in out} == {expiry_at(D[33])} and len(out) == 2


async def test_chain_without_stock_quote(broker, fake, now):
    await chain_setup(broker, fake, now, [D[33]])
    fake.tickers[broker.stock_con_id("SPY")].time = None
    assert await broker.chain("SPY", 30, now) == []
    with pytest.raises(KeyError):
        broker.stock_con_id("QQQ")


# ---- 10. rebind + cancel_all -----------------------------------------------------------
async def test_rebind_only_our_order_ref(fake, now):
    def resting(oid, ref):
        order = IbOrder(
            orderId=oid,
            permId=oid * 10,
            action="BUY",
            totalQuantity=100,
            orderType="LMT",
            lmtPrice=600.0,
            orderRef=ref,
        )
        return Trade(STK, order, OrderStatus(orderId=oid, status="Submitted"))

    fake.trades += [resting(7, "qmpc:abc"), resting(8, "other")]
    b = Broker(BrokerCfg(), SimClock(now), readonly=False, ib=fake)
    await b.connect()
    (o,) = b.open_orders()
    assert (o.broker_order_id, o.perm_id, o.action, o.qty, o.sec_type) == (7, 70, "BUY", 100, "STK")
    assert o.state is OrderState.SUBMITTED and o.reason == "rebound"
    fake.on_cancel = lambda t: fake.emit_status(t, "Cancelled")
    assert await b.cancel_all() == 1
    assert fake.cancelled == [7] and o.state is OrderState.CANCELLED and b.open_orders() == []
    assert await b.cancel_all() == 0
    await b.disconnect()


# ---- 11. reconnect ---------------------------------------------------------------------
async def test_reconnect_resubscribes_and_flags_reconcile(broker, fake, now, monkeypatch):
    monkeypatch.setattr(broker_mod, "BACKOFF_S", [0.01])
    cid = await stream_stock(broker, fake, now)
    await broker.stream_leg(1001)
    assert fake.connect_calls == 1 and fake.pnl_requests == 1
    fake.fail_connects = 1
    fake.emit_disconnect()
    assert broker.connected is False
    for _ in range(200):
        await asyncio.sleep(0.01)
        if broker.connected:
            break
    assert broker.connected and fake.connect_calls == 3 and fake.pnl_requests == 2
    assert fake.subscribed.count((cid, "")) == 2 and fake.subscribed.count((1001, "106")) == 2
    assert broker.needs_reconcile is True
    assert broker.quote(cid, now, False) is not None


async def test_reconnect_survives_drop_during_rebind(broker, fake, now, monkeypatch):
    monkeypatch.setattr(broker_mod, "BACKOFF_S", [0.01])
    orig = fake.reqAllOpenOrdersAsync

    async def drop_once():
        fake.reqAllOpenOrdersAsync = orig
        fake.emit_disconnect()  # the socket drops while _open() is still rebinding
        return await orig()

    fake.reqAllOpenOrdersAsync = drop_once
    fake.emit_disconnect()
    for _ in range(200):
        await asyncio.sleep(0.01)
        if broker.connected:
            break
    assert broker.connected and fake.connect_calls == 3


# ---- 12. import hygiene ----------------------------------------------------------------
def test_import_emits_no_deprecation_warning():
    r = subprocess.run(
        [sys.executable, "-W", "error", "-c", "import src.broker"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert r.returncode == 0, r.stderr


# ---- account / bars / offset -----------------------------------------------------------
async def test_account(broker, fake):
    fake.account_values = [
        AccountValue("DU1", "NetLiquidation", "100000", "USD", ""),
        AccountValue("DU1", "ExcessLiquidity", "60000", "USD", ""),
        AccountValue("DU1", "BuyingPower", "400000", "USD", ""),
        AccountValue("DU1", "DayTradesRemaining", "-1", "", ""),
    ]
    fake.pnl.dailyPnL = -123.0
    a = await broker.account()
    assert (a.nlv, a.excess_liquidity, a.buying_power) == (100000.0, 60000.0, 400000.0)
    assert a.daily_pnl == -123.0 and a.day_trades_remaining == -1
    fake.pnl.dailyPnL = nan
    assert (await broker.account()).daily_pnl is None
    fake.account_values = []
    assert await broker.account() is None
    broker.connected = False
    assert await broker.account() is None


async def test_historical_bars(broker, fake, now):
    fake.bars = [
        BarData(datetime(2026, 3, 13, 14, 30 + i, tzinfo=UTC), 1.0, 2.0, 0.5, 1.5 + i, 0, 0, 0)
        for i in range(2)
    ]
    df = await broker.historical_bars("SPY", now, 1, 60)
    assert list(df.columns) == ["ts", "open", "high", "low", "close", "mid"]
    assert str(df.ts.dt.tz) == "America/New_York" and df.ts[0].hour == 10
    assert list(df.mid) == [1.5, 2.5]
    (req,) = fake.hist_requests
    assert req[1] == datetime(2026, 3, 13, 16, 0, tzinfo=NY)
    assert req[2:] == ("1 D", "1 min", "MIDPOINT", True, 2)


async def test_clock_offset_measured_at_connect(fake, now):
    fake.current_time = aware(now) + timedelta(seconds=3)
    b = Broker(BrokerCfg(), SimClock(now), readonly=True, ib=fake)
    await b.connect()
    assert b.clock_offset_s == pytest.approx(3.0) and fake.md_type == 3
    await b.disconnect()
    assert b.connected is False


async def test_idle_timeout_probes_before_reconnect(broker, fake, now, monkeypatch):
    monkeypatch.setattr(broker_mod, "BACKOFF_S", [0.01])
    await stream_stock(broker, fake, now)
    fake.timeoutEvent.emit()  # idle feed, link alive: the probe succeeds and nothing reconnects
    await asyncio.sleep(0.05)
    assert broker.connected and fake.connect_calls == 1

    async def dead(*_):
        raise OSError("socket gone")

    fake.reqCurrentTimeAsync = dead
    fake.timeoutEvent.emit()  # probe fails: treated as a drop
    await asyncio.sleep(0.05)
    assert broker.connected is False
    fake.reqCurrentTimeAsync = type(fake).reqCurrentTimeAsync.__get__(fake)
    for _ in range(200):
        await asyncio.sleep(0.01)
        if broker.connected:
            break
    assert broker.connected and fake.connect_calls >= 2 and broker.needs_reconcile is True
