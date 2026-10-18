"""ib_async broker: conId-keyed positions, streamed legs (tick 106), one expiry chain,
BAG straddles with whatIf, typed order states, cancel-on-timeout with re-query,
orderRef rebind, clock offset and a reconnect loop.

Used with --execute for orders and as the `market` of SimBroker in dry-run. `readonly`
is not a safety belt; the single order path is. Every order is a DAY limit with
orderRef "qmpc:<session_id[:8]>" so rebind and cancel_all only ever touch this bot's orders.
"""

import asyncio
import functools
import itertools
import math
import time
from collections import deque
from collections.abc import Callable
from datetime import date, datetime, timedelta
from uuid import uuid4

import pandas as pd
from ib_async import IB, Bag, ComboLeg, Contract, LimitOrder, Option, Stock, Ticker, Trade
from ib_async import OrderState as IbOrderState

from src import validation
from src.clock import (
    Clock,
    aware,
    expiry_at,
    is_session,
    ny,
    pick_expiry,
    session_close,
    session_date,
)
from src.config import Broker as BrokerCfg
from src.log import get_logger
from src.types import Account, Fill, IbkrGreeks, OptionQuote, Order, OrderState, Position, Quote

log = get_logger("broker")


class BrokerError(Exception):
    pass


class BrokerDisconnected(BrokerError):
    pass


REJECT_CODES = frozenset({201, 203, 110, 354, 10090, 10167, 10197, 10349, 10242, 10243})
REQUERY_CODES = frozenset({10147, 10148})  # "cannot be cancelled" / "state: Filled" → ask again
STATUS_MAP = {
    "PendingSubmit": OrderState.SUBMITTED,
    "ApiPending": OrderState.SUBMITTED,
    "PreSubmitted": OrderState.SUBMITTED,
    "Submitted": OrderState.SUBMITTED,
    "PendingCancel": OrderState.SUBMITTED,  # ib_async sets it inside cancelOrder, before the ack
    "ValidationError": OrderState.SUBMITTED,
    "ApiUpdate": OrderState.SUBMITTED,
    "Filled": OrderState.FILLED,
    "Cancelled": OrderState.CANCELLED,
    "ApiCancelled": OrderState.CANCELLED,
    "Inactive": OrderState.INACTIVE,
}
BACKOFF_S = (1, 2, 4, 8, 16, 32, 60)  # reconnect delays; tests shorten it
OFFSET_INTERVAL_S = 3600.0  # clock offset re-measured this often
ACK_S = 5.0  # wait for a cancel acknowledgement before re-querying
FARM_OK = frozenset({2104, 2106, 2158})


def _expiry_of(c: Contract) -> datetime:
    return expiry_at(date.fromisoformat(c.lastTradeDateOrContractMonth[:8]))


def _linked(fn):
    """ib_async raises ConnectionError (an OSError) from a request sent while the socket is down
    and fails pending requests with it on a drop: that is BrokerDisconnected (the Runner goes
    DEGRADED), not a bug (HALT). An OSError while the link is up propagates unchanged."""

    @functools.wraps(fn)
    async def call(self, *a, **k):
        try:
            return await fn(self, *a, **k)
        except OSError as e:
            if self.ib.isConnected():
                raise
            raise BrokerDisconnected(f"{fn.__name__}: {e!r}") from e

    return call


class Broker:
    def __init__(
        self,
        cfg: BrokerCfg,
        clock: Clock,
        readonly: bool,
        order_ref: str = "qmpc:",
        stale_quote_s: float = 15.0,
        roll_dte: int = 7,
        prefer_monthly: bool = True,
        ib: IB | None = None,
    ):
        self.cfg, self.clock, self.readonly = cfg, clock, readonly
        self.order_ref, self.stale_quote_s, self.roll_dte = order_ref, stale_quote_s, roll_dte
        self.prefer_monthly = prefer_monthly
        self.ib = ib or IB()
        self.connected = False
        self.data_ok = True
        self.market_data_type = cfg.market_data_type
        self.clock_offset_s = 0.0
        self.needs_reconcile = False
        self.fills: deque[Fill] = deque()
        self.commission_updates: deque[tuple[str, float]] = deque()
        self.orders: dict[str, Order] = {}
        self._trades: dict[str, Trade] = {}
        self._events: dict[str, asyncio.Event] = {}
        self._requery: set[str] = set()
        self._tickers: dict[int, Ticker] = {}
        self._contracts: dict[int, Contract] = {}
        self._stock: Contract | None = None
        self._pnl = None
        self._reject_hook: Callable[[datetime], None] | None = None
        self._store = None
        self._session_id = ""
        self._lost = asyncio.Event()
        self._watch_task: asyncio.Task | None = None
        self._skew_logged: datetime | None = None

    # ---- wiring -------------------------------------------------------------------------
    def bind_session(self, store, session_id: str) -> None:
        self._store, self._session_id = store, session_id

    def on_reject(self, hook: Callable[[datetime], None]) -> None:
        self._reject_hook = hook

    async def connect(self) -> None:
        await self._open()
        ib = self.ib
        ib.orderStatusEvent += self._on_status
        ib.execDetailsEvent += self._on_exec
        ib.commissionReportEvent += self._on_commission
        ib.errorEvent += self._on_error
        ib.disconnectedEvent += self._on_disconnected
        ib.timeoutEvent += self._on_timeout
        self._watch_task = asyncio.create_task(self._watch())

    async def _open(self) -> None:
        """connectAsync + everything that must be redone after a reconnect."""
        cfg = self.cfg
        await self.ib.connectAsync(
            cfg.host,
            cfg.port,
            clientId=cfg.client_id,
            timeout=cfg.timeout_s,
            readonly=self.readonly,
            account=cfg.account,
        )
        self.ib.reqMarketDataType(cfg.market_data_type)
        self.ib.setTimeout(60)
        await self._measure_offset()
        self._pnl = self.ib.reqPnL(cfg.account or self.ib.managedAccounts()[0])
        self.connected = True
        await self._rebind()
        self._resubscribe()

    async def disconnect(self) -> None:
        if self._watch_task:
            self._watch_task.cancel()
            self._watch_task = None
        self.connected = False
        self.ib.disconnect()

    async def _measure_offset(self) -> None:
        t_ib = await asyncio.wait_for(self.ib.reqCurrentTimeAsync(), self.cfg.timeout_s)
        self.clock_offset_s = (aware(t_ib) - self.clock.now()).total_seconds()
        if abs(self.clock_offset_s) > 2:
            log.warning(f"clock offset vs IBKR {self.clock_offset_s:+.1f} s")

    async def _watch(self) -> None:
        """Reconnect after a disconnect/timeout; re-measure the clock offset hourly."""
        while True:
            try:
                await asyncio.wait_for(self._lost.wait(), OFFSET_INTERVAL_S)
            except TimeoutError:
                if self.connected:
                    try:
                        await self._measure_offset()
                    except OSError as e:
                        log.warning(f"clock offset probe failed: {e}")
                continue
            await self._reconnect()

    async def _reconnect(self) -> None:
        for i in itertools.count():
            await asyncio.sleep(BACKOFF_S[min(i, len(BACKOFF_S) - 1)])
            self.ib.disconnect()
            self._lost.clear()  # a drop during _open() must set it again and loop us back here
            self.needs_reconcile = True  # _open() sets connected before its awaits: set it first
            try:
                await self._open()
            except Exception as e:  # any failure → next backoff step; the task must not die
                log.warning(f"reconnect attempt {i + 1} failed: {e}")
                continue
            log.warning(f"reconnected after {i + 1} attempt(s); reconcile pending")
            return

    def _on_disconnected(self, *_) -> None:
        self.connected = False
        self._lost.set()

    def _on_timeout(self, *_) -> None:
        """No inbound message for 60 s (idle feed): probe the link before treating it as a drop."""
        asyncio.get_running_loop().create_task(self._probe_or_drop())

    async def _probe_or_drop(self) -> None:
        try:
            await asyncio.wait_for(self.ib.reqCurrentTimeAsync(), self.cfg.timeout_s)
            log.debug("idle feed; link alive")
        except Exception as e:  # any failure means the link is gone
            log.warning(f"liveness probe failed: {e}")
            self._on_disconnected()

    # ---- market data --------------------------------------------------------------------
    def _subscribe(self, c: Contract) -> Ticker:
        t = self.ib.reqMktData(c, "106" if c.secType == "OPT" else "", False, False)
        self._tickers[c.conId] = t
        return t

    def _resubscribe(self) -> None:
        for cid in list(self._tickers):
            self._subscribe(self._contracts[cid])

    @_linked
    async def stream_stock(self, symbol: str) -> int:
        (c,) = await self.ib.qualifyContractsAsync(Stock(symbol, "SMART", "USD"))
        if c is None:
            raise BrokerError(f"{symbol}: stock not qualifiable")
        self._stock = self._contracts[c.conId] = c
        self._subscribe(c)
        return c.conId

    def stock_con_id(self, symbol: str) -> int:
        if self._stock is None or self._stock.symbol != symbol:
            raise KeyError(symbol)
        return self._stock.conId

    @_linked
    async def stream_leg(self, con_id: int) -> None:
        c = self._contracts.get(con_id)
        if c is None or not c.exchange:  # positions() contracts come without an exchange
            (c,) = await self.ib.qualifyContractsAsync(Contract(conId=con_id))
            if c is None:
                raise BrokerError(f"conId {con_id} not qualifiable")
            self._contracts[con_id] = c
        self._subscribe(c)

    async def unstream(self, con_id: int) -> None:
        if (c := self._contracts.get(con_id)) is not None:
            self.ib.cancelMktData(c)
        self._tickers.pop(con_id, None)

    def quote(self, con_id: int, now: datetime, require_live: bool) -> Quote | None:
        t = self._tickers.get(con_id)
        if t is None or not self.connected:
            return None
        if not self.readonly and abs(self.clock_offset_s) > 10:
            if self._skew_logged is None or (now - self._skew_logged) >= timedelta(seconds=60):
                self._skew_logged = now
                log.warning(f"clock_skew {self.clock_offset_s:+.1f} s: quotes refused")
            return None
        return validation.fresh_quote(
            t.bid,
            t.ask,
            t.last,
            t.time,
            t.marketDataType or self.market_data_type,
            now,
            self.stale_quote_s,
            require_live,
        )  # ticker.time is our own arrival stamp: same clock as now, no offset

    def leg_greeks(self, con_id: int, S: float, now: datetime) -> IbkrGreeks | None:
        t = self._tickers.get(con_id)
        mg = t.modelGreeks if t is not None else None
        if mg is None:
            return None
        return validation.ibkr_greeks(
            mg.delta, mg.gamma, mg.vega, mg.theta, mg.impliedVol, mg.undPrice, t.time
        )

    def _option_quote(self, c: Contract, t: Ticker, now: datetime) -> OptionQuote | None:
        mg = t.modelGreeks
        return validation.option_quote(
            c.conId,
            c.right,
            c.strike,
            _expiry_of(c),
            t.bid,
            t.ask,
            mg.impliedVol if mg else None,
            mg.delta if mg else None,
            t.time,
            t.marketDataType or self.market_data_type,
            now,
            self.stale_quote_s,
        )

    def leg_quote(self, con_id: int, now: datetime) -> OptionQuote | None:
        t, c = self._tickers.get(con_id), self._contracts.get(con_id)
        return self._option_quote(c, t, now) if t is not None and c is not None else None

    @_linked
    async def chain(self, symbol: str, target_dte: int, now: datetime) -> list[OptionQuote]:
        q = self.quote(self.stock_con_id(symbol), now, False)
        if q is None:
            log.warning("chain: no fresh stock quote")
            return []
        params = await self.ib.reqSecDefOptParamsAsync(symbol, "", "STK", self.stock_con_id(symbol))
        smart = [p for p in params if p.exchange == "SMART"]
        p = next((p for p in smart if p.tradingClass == symbol), smart[0] if smart else None)
        if p is None:
            log.warning("chain: no SMART option parameters")
            return []
        cands = [date.fromisoformat(e[:8]) for e in p.expirations]
        exp = pick_expiry(cands, session_date(now), target_dte, self.roll_dte, self.prefer_monthly)
        if exp is None:
            log.warning(f"chain: no expiry with dte >= {self.roll_dte + 1}")
            return []
        strikes = sorted(p.strikes, key=lambda k: abs(k - q.mid))[:6]  # the 6 nearest S
        contracts = [
            Option(symbol, exp.strftime("%Y%m%d"), k, r, "SMART", tradingClass=symbol)
            for k in strikes
            for r in ("C", "P")
        ]
        kept = [c for c in await self.ib.qualifyContractsAsync(*contracts) if c is not None]
        for c in kept:
            self._contracts[c.conId] = c
        tickers = await self.ib.reqTickersAsync(*kept)
        out = [
            oq for c, t in zip(kept, tickers, strict=True) if (oq := self._option_quote(c, t, now))
        ]
        if not out:
            log.warning(f"chain: {len(contracts)} contracts, {len(kept)} qualified, none quoted")
        return out

    @_linked
    async def historical_bars(
        self, symbol: str, end: datetime, sessions: int, bar_s: int, what: str = "MIDPOINT"
    ) -> pd.DataFrame:
        """One request per session (the only paced loop: 0.6 s apart), oldest first."""
        d, days = session_date(end), []
        while len(days) < sessions:
            if is_session(d):
                days.append(d)
            d -= timedelta(days=1)
        c = (
            self._stock
            if self._stock and self._stock.symbol == symbol
            else Stock(symbol, "SMART", "USD")
        )
        size = "1 min" if bar_s == 60 else f"{bar_s} secs"
        rows = []
        for i, day in enumerate(reversed(days)):
            if i:
                await asyncio.sleep(0.6)
            bars = await self.ib.reqHistoricalDataAsync(
                c, session_close(expiry_at(day)), "1 D", size, what, True, formatDate=2
            )
            rows += [(ny(b.date), b.open, b.high, b.low, b.close) for b in bars]
        df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close"])
        df["mid"] = df["close"]
        return df

    # ---- account / positions ------------------------------------------------------------
    @_linked
    async def positions(self) -> dict[int, Position]:
        out = {}
        for p in await self.ib.reqPositionsAsync():
            c, qty = p.contract, int(p.position)
            if qty == 0:
                continue
            opt = c.secType == "OPT"
            mult = int(c.multiplier or 100) if opt else 1
            self._contracts.setdefault(c.conId, c)
            out[c.conId] = Position(
                c.conId,
                c.symbol,
                c.secType,
                qty,
                p.avgCost / mult,
                right=c.right or None,
                strike=c.strike or None,
                expiry=_expiry_of(c) if opt else None,
                mult=mult,
            )
        return out

    @_linked
    async def account(self) -> Account | None:
        if not self.connected:
            return None
        try:
            vals = await asyncio.wait_for(
                self.ib.accountSummaryAsync(self.cfg.account), self.cfg.timeout_s
            )
            v = {a.tag: a.value for a in vals if a.currency in ("USD", "")}
            nlv, el, bp = (
                float(v[k]) for k in ("NetLiquidation", "ExcessLiquidity", "BuyingPower")
            )
        except (TimeoutError, KeyError, ValueError) as e:  # a ConnectionError goes to _linked
            log.warning(f"account summary unavailable: {e!r}")
            return None
        daily = self._pnl.dailyPnL if self._pnl is not None else math.nan
        return Account(
            nlv,
            el,
            bp,
            daily if math.isfinite(daily) else None,
            int(float(v.get("DayTradesRemaining", -1))),
            self.clock.now(),
        )

    # ---- orders -------------------------------------------------------------------------
    def _ref(self) -> str:
        return f"{self.order_ref}{self._session_id[:8]}"

    def _place(self, o: Order, contract: Contract, ib_order) -> Order:
        if not (self.connected and self.ib.isConnected()):  # placeOrder would raise, unsent
            raise BrokerDisconnected("not connected")
        self.orders[o.id] = o
        try:
            tr = self.ib.placeOrder(contract, ib_order)
        except Exception as e:  # placeOrder failed → the order never reached TWS
            o.state, o.error, o.updated = OrderState.REJECTED, str(e), self.clock.now()
            log.error(f"placeOrder failed: {e}")
            if self._reject_hook:
                self._reject_hook(o.updated)
        else:
            o.broker_order_id = tr.order.orderId
            self._trades[o.id] = tr
        if self._store:
            self._store.save_order(self._session_id, o)
        return o

    async def place_shares(self, qty: int, limit: float | None, reason: str) -> Order:
        if limit is None:
            raise ValueError("marketable limit required")
        if self._stock is None:
            raise BrokerError("stock not streamed")
        o = Order(
            uuid4().hex[:12],
            self._stock.conId,
            "STK",
            "BUY" if qty > 0 else "SELL",
            abs(qty),
            limit,
            reason,
            OrderState.SUBMITTED,
            created=self.clock.now(),
        )
        ib_order = LimitOrder(
            o.action, o.qty, limit, tif="DAY", orderRef=self._ref(), outsideRth=False
        )
        return self._place(o, self._stock, ib_order)

    def _bag(self, call: OptionQuote, put: OptionQuote, action: str) -> Bag:
        symbol = self._stock.symbol if self._stock else self._contracts[call.con_id].symbol
        for q in (call, put):
            self._contracts.setdefault(
                q.con_id,
                Option(
                    symbol,
                    ny(q.expiry).strftime("%Y%m%d"),
                    q.strike,
                    q.right,
                    "SMART",
                    multiplier="100",
                    conId=q.con_id,
                    tradingClass=symbol,
                ),
            )
        legs = [
            ComboLeg(conId=q.con_id, ratio=1, action=action, exchange="SMART") for q in (call, put)
        ]
        return Bag(symbol=symbol, exchange="SMART", currency="USD", comboLegs=legs)

    @_linked
    async def whatif_straddle(
        self, call: OptionQuote, put: OptionQuote, n: int, limit: float
    ) -> float:
        st: IbOrderState = await self.ib.whatIfOrderAsync(
            self._bag(call, put, "BUY"), LimitOrder("BUY", n, limit)
        )
        try:
            margin = float(st.initMarginChange)
        except (TypeError, ValueError):
            margin = math.nan
        if not math.isfinite(margin):
            raise BrokerError(
                f"whatIf: no initMarginChange ({st.warningText or st.initMarginChange!r})"
            )
        return margin

    async def place_straddle(
        self, call: OptionQuote, put: OptionQuote, n: int, action: str, limit: float, reason: str
    ) -> Order:
        bag = self._bag(call, put, action)
        o = Order(
            uuid4().hex[:12],
            0,
            "BAG",
            action,
            n,
            limit,
            reason,
            OrderState.SUBMITTED,
            created=self.clock.now(),
        )
        ib_order = LimitOrder(action, n, limit, tif="DAY", orderRef=self._ref(), outsideRth=False)
        return self._place(o, bag, ib_order)

    def _event(self, o: Order) -> asyncio.Event:
        return self._events.setdefault(o.id, asyncio.Event())

    async def _await(self, o: Order, timeout_s: float, done: Callable[[], bool]) -> None:
        ev, deadline = self._event(o), time.monotonic() + timeout_s
        while not done() and (remaining := deadline - time.monotonic()) > 0:
            ev.clear()
            try:
                await asyncio.wait_for(ev.wait(), remaining)
            except TimeoutError:
                return

    async def wait(self, order: Order, timeout_s: float) -> Order:
        await self._await(order, timeout_s, lambda: not order.is_open)
        if order.is_open or order.state is OrderState.INACTIVE or order.id in self._requery:
            await self._settle(order, ACK_S)
        return order

    async def cancel(self, order: Order, timeout_s: float = 10.0) -> Order:
        return await self._settle(order, timeout_s)

    @_linked
    async def _settle(self, o: Order, ack_s: float) -> Order:
        """Cancel whatever is still open at the broker and take its final answer."""
        tr = self._trades.get(o.id)
        if tr is not None and (o.is_open or o.state is OrderState.INACTIVE):
            self.ib.cancelOrder(tr.order)
            await self._await(o, ack_s, lambda: not o.is_open or o.id in self._requery)
        if o.is_open or o.state is OrderState.INACTIVE or o.id in self._requery:
            await self._requery_order(o)
        return o

    async def _requery_order(self, o: Order) -> None:
        self._requery.discard(o.id)
        for tr in await self.ib.reqAllOpenOrdersAsync():
            if self._find(tr) is o:
                self._apply_status(o, tr)  # still at the broker: its state, whatever it is
                return
        tr = self._trades.get(o.id)  # stale after a reconnect; executions are re-fetched
        execs = [
            int(f.execution.cumQty)
            for f in self.ib.fills()
            if o.perm_id and f.execution.permId == o.perm_id and f.contract.secType != "BAG"
        ]
        o.filled = max([o.filled, int(tr.orderStatus.filled) if tr else 0, *execs])
        o.state = OrderState.FILLED if o.filled >= o.qty else OrderState.CANCELLED
        o.updated = self.clock.now()
        log.warning(
            f"order {o.id} re-queried: gone from broker, {o.filled}/{o.qty} → {o.state.value}"
        )
        self._persist(o)

    async def cancel_all(self) -> int:
        ours = [o for o in self.orders.values() if o.is_open]
        await asyncio.gather(*(self.cancel(o) for o in ours))
        return sum(not o.is_open for o in ours)

    def open_orders(self) -> list[Order]:
        return [o for o in self.orders.values() if o.is_open]

    async def _rebind(self) -> None:
        """Adopt resting orders carrying our orderRef (ours from a previous session or client)."""
        seen = set()
        for tr in self.ib.openTrades() + list(await self.ib.reqAllOpenOrdersAsync()):
            key = tr.order.permId or tr.order.orderId
            if key in seen or not tr.order.orderRef.startswith(self.order_ref):
                continue
            seen.add(key)
            o = self._find(tr)
            if o is None:
                c = tr.contract
                o = Order(
                    f"ib{key}",
                    0 if c.secType == "BAG" else c.conId,
                    c.secType,
                    tr.order.action,
                    int(tr.order.totalQuantity),
                    tr.order.lmtPrice or None,
                    "rebound",
                    created=self.clock.now(),
                )
                log.warning(
                    f"rebound order {tr.order.orderRef} id={key} {o.action} {o.qty} {o.sec_type}"
                )
            self.orders[o.id], self._trades[o.id] = o, tr
            o.broker_order_id = tr.order.orderId
            self._apply_status(o, tr)

    # ---- events -------------------------------------------------------------------------
    def _find(self, trade: Trade) -> Order | None:
        oid, pid = trade.order.orderId, trade.order.permId
        for o in self.orders.values():
            if (pid and o.perm_id == pid) or (oid and o.broker_order_id == oid):
                return o
        return None

    def _persist(self, o: Order) -> None:
        if self._store:
            self._store.update_order(self._session_id, o)
        self._event(o).set()

    def _apply_status(self, o: Order, trade: Trade) -> None:
        if o.state is OrderState.REJECTED:  # sticky: TWS trails a reject with Cancelled/Inactive
            self._event(o).set()
            return
        st = trade.orderStatus
        state = STATUS_MAP.get(st.status)
        if state is None:
            log.warning(f"order {o.id}: unknown TWS status {st.status!r}")
            state = OrderState.UNKNOWN
        elif state is OrderState.SUBMITTED and st.filled > 0:
            state = OrderState.PARTIAL
        o.state, o.filled, o.avg_price = state, int(st.filled), st.avgFillPrice
        o.perm_id, o.updated = trade.order.permId, self.clock.now()
        self._persist(o)

    def _on_status(self, trade: Trade) -> None:
        if (o := self._find(trade)) is not None:
            self._apply_status(o, trade)

    def _on_exec(self, trade: Trade, fill) -> None:
        o = self._find(trade)
        if o is None or fill.contract.secType == "BAG":  # combo-level report: legs carry the fills
            return
        ex, cid = fill.execution, fill.contract.conId
        qty = int(ex.shares) * (1 if ex.side == "BOT" else -1)
        if o.sec_type == "BAG":
            o.leg_fills[cid] = o.leg_fills.get(cid, 0) + qty
        f = Fill(ex.execId, o.id, cid, qty, ex.price, aware(ex.time), None, "broker")
        self.fills.append(f)
        if self._store:
            self._store.save_fill(self._session_id, f)
        log.info(f"fill {o.reason} {o.id} conId={cid} qty={qty} @ {ex.price}")

    def _on_commission(self, trade: Trade, fill, report) -> None:
        if fill.contract.secType == "BAG" or not math.isfinite(report.commission):
            return  # mirrors _on_exec: only leg fills are booked
        self.commission_updates.append((report.execId, report.commission))
        if self._store:
            self._store.set_commission(report.execId, report.commission)

    def _on_error(self, reqId: int, code: int, msg: str, contract) -> None:
        o = next(
            (o for o in self.orders.values() if reqId > 0 and o.broker_order_id == reqId), None
        )
        if o is not None:
            if code == 202 and o.is_open:
                o.state = OrderState.CANCELLED
            elif code in REQUERY_CODES:
                self._requery.add(o.id)
                log.warning(f"order {o.id}: {code} {msg} → re-query")
            elif code in REJECT_CODES:
                o.state, o.error = OrderState.REJECTED, msg
                log.error(f"order {o.id} rejected: {code} {msg}")
                if self._reject_hook:
                    self._reject_hook(self.clock.now())
            else:
                log.warning(f"order {o.id}: {code} {msg}")
                return
            o.updated = self.clock.now()
            self._persist(o)
        elif code == 1100:
            self.data_ok = False
            log.error("1100: connectivity between IB and TWS lost")
        elif code == 1101:
            log.warning("1101: connectivity restored, data lost → resubscribing")
            self.data_ok = True
            self._resubscribe()
        elif code == 1102:
            self.data_ok = self.needs_reconcile = True
            log.warning("1102: connectivity restored, data maintained → reconcile")
        elif code == 10167:
            log.warning(f"10167: delayed market data ({msg})")
        elif code in FARM_OK:
            self.data_ok = True
        elif code not in range(2100, 2200):
            log.warning(f"IBKR error {code} reqId={reqId}: {msg}")
