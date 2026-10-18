"""Duck-typed stand-in for ib_async.IB: records every call and lets tests emit the events
TWS would send. Object shapes are the real ib_async classes. Like ib_async's Client.send, every
request made while disconnected raises ConnectionError("Not connected"); local caches
(openTrades, fills, managedAccounts) and cancelMktData (ib_async drops tickers on a disconnect)
do not."""

from datetime import UTC, datetime
from math import nan
from types import SimpleNamespace

from eventkit import Event
from ib_async import (
    CommissionReport,
    Contract,
    Execution,
    OptionComputation,
    OrderStatus,
    PnL,
    Ticker,
    Trade,
)
from ib_async import Fill as IbFill

EVENTS = (
    "orderStatusEvent",
    "execDetailsEvent",
    "commissionReportEvent",
    "errorEvent",
    "disconnectedEvent",
    "timeoutEvent",
    "pendingTickersEvent",
    "positionEvent",
    "connectedEvent",
)


def greeks(**kw) -> OptionComputation:
    return OptionComputation(0, **kw)


def make_ticker(c, bid=nan, ask=nan, last=nan, time=None, md=1, greeks=None) -> Ticker:
    t = Ticker(contract=c)  # Ticker.__init__ ignores field kwargs; assign after construction
    t.bid, t.ask, t.last, t.time, t.marketDataType, t.modelGreeks = bid, ask, last, time, md, greeks
    return t


class FakeIB:
    def __init__(self):
        for name in EVENTS:  # per instance, so brokers in different tests never share handlers
            setattr(self, name, Event(name))
        self.positions_list = []
        self.tickers: dict[int, Ticker] = {}
        self.known: dict[int, Contract] = {}  # conId -> contract returned by qualify
        self.trades: list[Trade] = []
        self.fills_list: list[IbFill] = []  # survives a trade being forgotten (reqExecutions)
        self.placed: list[Trade] = []
        self.cancelled: list[int] = []
        self.whatifs = []
        self.subscribed: list[tuple[int, str]] = []
        self.unsubscribed: list[int] = []
        self.unqualifiable: set[float] = set()
        self.sec_def = []
        self.account_values = []
        self.pnl = PnL("DU1", "", 0.0, 0.0, 0.0)
        self.pnl_requests = 0
        self.bars = []
        self.hist_requests = []
        self.next_id = 1
        self.next_con_id = 5000
        self.current_time = datetime(2026, 3, 15, 14, 0, tzinfo=UTC)
        self.connected = False
        self.connect_calls = 0
        self.fail_connects = 0
        self.md_type = None
        self.on_cancel = None  # callable(trade) run inside cancelOrder, like TWS answering
        self.snapshot = lambda c: Ticker(contract=c)  # reqTickersAsync for unknown conIds

    def _send(self):
        if not self.connected:
            raise ConnectionError("Not connected")

    # ---- connection
    async def connectAsync(self, host, port, clientId, timeout, readonly, account=""):  # noqa: ASYNC109
        self.connect_calls += 1
        if self.fail_connects:
            self.fail_connects -= 1
            raise ConnectionRefusedError("TWS down")
        self.connected = True

    def disconnect(self):
        if self.connected:
            self.connected = False
            self.disconnectedEvent.emit()

    def isConnected(self):
        return self.connected

    def reqMarketDataType(self, t):
        self._send()
        self.md_type = t

    def setTimeout(self, s):
        self.timeout_s = s

    async def reqCurrentTimeAsync(self):
        self._send()
        return self.current_time

    def reqPnL(self, account, modelCode=""):
        self._send()
        self.pnl_requests += 1
        return self.pnl

    def managedAccounts(self):
        return ["DU1"]

    # ---- orders
    def openTrades(self):
        return [t for t in self.trades if t.orderStatus.status not in OrderStatus.DoneStates]

    async def reqAllOpenOrdersAsync(self):
        self._send()
        return self.openTrades()

    def placeOrder(self, contract, order):
        self._send()
        order.orderId, self.next_id = self.next_id, self.next_id + 1
        tr = Trade(contract, order, OrderStatus(orderId=order.orderId, status="Submitted"))
        self.trades.append(tr)
        self.placed.append(tr)
        return tr

    def cancelOrder(self, order, manualCancelOrderTime=""):
        self._send()
        self.cancelled.append(order.orderId)
        tr = next((t for t in self.trades if t.order is order), None)
        if tr is None:  # ib_async logs "Unknown orderId" and returns None
            return None
        if tr.orderStatus.status not in OrderStatus.DoneStates:  # ib_async emits this synchronously
            st = tr.orderStatus
            self.emit_status(tr, "PendingCancel", filled=st.filled, avg=st.avgFillPrice)
        if self.on_cancel:
            self.on_cancel(tr)
        return tr

    def fills(self):
        return list(self.fills_list)

    async def whatIfOrderAsync(self, contract, order):
        self._send()
        self.whatifs.append((contract, order))
        return SimpleNamespace(initMarginChange="1234.5", warningText="")

    # ---- contracts / market data
    def _stamp(self, c):
        if c.conId in self.known:
            return self.known[c.conId]
        if not c.conId:
            c.conId, self.next_con_id = self.next_con_id, self.next_con_id + 1
        return c

    async def qualifyContractsAsync(self, *cs):
        self._send()
        return [None if c.strike in self.unqualifiable else self._stamp(c) for c in cs]

    async def reqSecDefOptParamsAsync(self, *a):
        self._send()
        return self.sec_def

    async def reqTickersAsync(self, *cs):
        self._send()
        return [self.tickers.get(c.conId) or self.snapshot(c) for c in cs]

    def reqMktData(self, c, generic="", snapshot=False, reg=False):
        self._send()
        self.subscribed.append((c.conId, generic))
        return self.tickers.setdefault(c.conId, Ticker(contract=c))

    def cancelMktData(self, c):
        self.unsubscribed.append(c.conId)

    async def reqHistoricalDataAsync(self, contract, end, duration, size, what, rth, formatDate=1):
        self._send()
        self.hist_requests.append((contract, end, duration, size, what, rth, formatDate))
        return list(self.bars)

    # ---- account
    async def reqPositionsAsync(self):
        self._send()
        return list(self.positions_list)

    async def accountSummaryAsync(self, account=""):
        self._send()
        return self.account_values

    # ---- test helpers: emit what TWS would
    def emit_status(self, tr, status, filled=0, avg=0.0):
        st = tr.orderStatus
        st.status, st.filled, st.avgFillPrice = status, filled, avg
        st.remaining = tr.order.totalQuantity - filled
        self.orderStatusEvent.emit(tr)

    def emit_exec(self, tr, contract, side, shares, price, exec_id, t=None):
        if isinstance(contract, int):
            contract = self.known.get(contract) or Contract(conId=contract)
        ex = Execution(
            execId=exec_id,
            time=t or self.current_time,
            side=side,
            shares=shares,
            price=price,
            permId=tr.order.permId,
            orderId=tr.order.orderId,
            cumQty=shares + sum(f.execution.shares for f in tr.fills if f.contract == contract),
        )
        fill = IbFill(contract, ex, CommissionReport(), ex.time)
        tr.fills.append(fill)
        self.fills_list.append(fill)
        self.execDetailsEvent.emit(tr, fill)
        return fill

    def emit_commission(self, exec_id, commission):
        tr, fill = next(
            (t, f) for t in self.trades for f in t.fills if f.execution.execId == exec_id
        )
        rep = CommissionReport(execId=exec_id, commission=commission, currency="USD")
        self.commissionReportEvent.emit(tr, fill, rep)

    def emit_error(self, order_id, code, msg="", contract=None):
        self.errorEvent.emit(order_id, code, msg, contract)

    def emit_disconnect(self):
        self.connected = False
        self.disconnectedEvent.emit()
