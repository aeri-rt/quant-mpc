"""SimBroker: the ONLY order path outside `--execute` (dry-run, backtest, tests).

THE broker surface. `src.broker.Broker` implements the same one; the Runner codes against it:

    connected, data_ok, market_data_type, clock_offset_s, needs_reconcile   # read from the market
    fills: deque[Fill]; commission_updates: deque[tuple[str, float]]; orders: dict[str, Order]
    async connect() / disconnect() / stream_stock(symbol) -> int / stream_leg(con_id) / unstream(con_id)
    stock_con_id(symbol) -> int
    quote(con_id, now, require_live) -> Quote | None
    leg_greeks(con_id, S, now) -> IbkrGreeks | None
    leg_quote(con_id, now) -> OptionQuote | None
    async chain(symbol, target_dte, now) -> list[OptionQuote]
    async positions() -> dict[int, Position]
    async account() -> Account | None
    async place_shares(qty: int, limit: float | None, reason: str) -> Order
    async whatif_straddle(call: OptionQuote, put: OptionQuote, n: int, limit: float) -> float
    async place_straddle(call: OptionQuote, put: OptionQuote, n: int, action: str, limit: float,
                         reason: str) -> Order
    async wait(order, timeout_s) -> Order
    async cancel(order) -> Order
    async cancel_all() -> int
    open_orders() -> list[Order]
    async historical_bars(*a, **k) -> pd.DataFrame

Fill timing rule (why this class exists): a share order placed during tick t fills against the
first quote seen AFTER placement -- the next tick in dry-run, the next bar's opening quote in the
backtest -- at that quote's ask (BUY) or bid (SELL). No order ever fills inside `place_shares`.
There is no same-bar lookahead and nothing to "fall back" to: this IS the simulation.

`market` is a duck type with the read side of the surface above (quote/leg_greeks/leg_quote/
chain/account/positions/stock_con_id/stream_*/connect/disconnect and the five flags): the live
`Broker` in readonly mode for dry-run, `backtest.sim.SyntheticMarket`, `tests.conftest.FakeMarket`.
"""

from __future__ import annotations

from collections import Counter, deque
from collections.abc import Callable
from dataclasses import replace
from datetime import datetime
from operator import attrgetter
from typing import TYPE_CHECKING
from uuid import uuid4

import pandas as pd

from src.book import _ledger
from src.costs import commission, option_half_spread, reg_fees
from src.types import Fill, OptionQuote, Order, OrderState, Position, Quote

if TYPE_CHECKING:
    from src.clock import Clock
    from src.config import Costs


def _add(pos: dict[int, Position], tmpl: Position, qty: int, px: float) -> None:
    """Average-cost ledger step for `tmpl.con_id` inside `pos`; a flat position is dropped."""
    cur = pos.get(tmpl.con_id)
    q, avg, _ = _ledger(cur.qty if cur else 0, cur.avg_cost if cur else 0.0, qty, px)
    if q:
        pos[tmpl.con_id] = replace(tmpl, qty=q, avg_cost=avg)
    else:
        pos.pop(tmpl.con_id, None)


class SimBroker:
    """Fills share orders on the next stock quote and straddles at placement, against `market`.

    Share fills are UNCONDITIONAL at the next quote's touch: `Order.limit` is persisted for audit
    and never compared with the quote. The Runner only sends marketable limits (ask+0.01 /
    bid-0.01 at the decision quote); checking them against the NEXT quote would, on real MIDPOINT
    bars where open_{i+1} != close_i by cents, skip exactly the fills whose price moved against
    the limit (free-option bias) and churn cancel/re-decide (fill_timeout_s 20 < bar 60 s).
    Synthetic bars hide this because open_{i+1} == close_i.

    `script(order)` forces an outcome: "fill" (default) | "partial:<k>" | "reject" | "timeout".
    A BAG never fills a naked leg: a partial fills k units on both legs and `wait()` cancels the
    rest. Commissions are known at fill time, so `commission_updates` stays empty (interface
    parity with the live Broker).
    """

    def __init__(
        self,
        market,
        costs: Costs,
        clock: Clock,
        script: Callable[[Order], str] | None = None,
    ):
        self._m, self._c, self._clock = market, costs, clock
        self.script = script
        self.fills: deque[Fill] = deque()
        self.commission_updates: deque[tuple[str, float]] = deque()
        self.orders: dict[str, Order] = {}
        self._pending: list[Order] = []  # STK orders waiting for the next quote
        self._pos: dict[int, Position] = {}  # what THIS broker filled: stock + legs
        self._seq: Counter[str] = Counter()  # fills per order, for exec ids
        self._symbol: str | None = None
        self._stock_id: int | None = None

    connected = property(attrgetter("_m.connected"))
    data_ok = property(attrgetter("_m.data_ok"))
    market_data_type = property(attrgetter("_m.market_data_type"))
    clock_offset_s = property(attrgetter("_m.clock_offset_s"))

    @property
    def needs_reconcile(self) -> bool:
        return self._m.needs_reconcile

    @needs_reconcile.setter
    def needs_reconcile(self, v: bool) -> None:  # the Runner resets the market's flag
        self._m.needs_reconcile = v

    # --- market read side, forwarded ---------------------------------------------------------

    async def connect(self) -> None:
        await self._m.connect()

    async def disconnect(self) -> None:
        await self._m.disconnect()

    async def stream_stock(self, symbol: str) -> int:
        self._symbol, self._stock_id = symbol, await self._m.stream_stock(symbol)
        return self._stock_id

    async def stream_leg(self, con_id: int) -> None:
        await self._m.stream_leg(con_id)

    async def unstream(self, con_id: int) -> None:
        await self._m.unstream(con_id)

    def stock_con_id(self, symbol: str) -> int:
        self._symbol, self._stock_id = symbol, self._m.stock_con_id(symbol)
        return self._stock_id

    def leg_greeks(self, con_id: int, S: float, now: datetime):
        return self._m.leg_greeks(con_id, S, now)

    def leg_quote(self, con_id: int, now: datetime) -> OptionQuote | None:
        return self._m.leg_quote(con_id, now)

    async def chain(self, symbol: str, target_dte: float, now: datetime) -> list[OptionQuote]:
        return await self._m.chain(symbol, target_dte, now)

    async def account(self):
        return await self._m.account()

    async def historical_bars(self, *a, **k) -> pd.DataFrame:
        f = getattr(self._m, "historical_bars", None)
        return await f(*a, **k) if f else pd.DataFrame()

    def quote(self, con_id: int, now: datetime, require_live: bool) -> Quote | None:
        q = self._m.quote(con_id, now, require_live)
        if q is not None and con_id == self._stock_id:
            self._fill_pending(q, now)
        return q

    async def positions(self) -> dict[int, Position]:
        """Broker truth (recovered state in dry-run, scripted in tests) merged with our ledger:
        a leg or stock lot closed by this broker nets out and disappears."""
        pos = dict(await self._m.positions())
        for p in self._pos.values():
            _add(pos, p, p.qty, p.avg_cost)
        return pos

    # --- orders --------------------------------------------------------------------------------

    async def place_shares(self, qty: int, limit: float | None, reason: str) -> Order:
        if not qty:
            raise ValueError("qty 0")
        _, cid = self._bound()
        o = self._new(cid, "STK", "BUY" if qty > 0 else "SELL", abs(qty), limit, reason)
        self._pending.append(o)  # fills on the next quote(), never here
        return o

    async def whatif_straddle(self, call: OptionQuote, put: OptionQuote, n: int, limit: float):
        return 0.0  # a long combo has no margin impact

    async def place_straddle(
        self, call: OptionQuote, put: OptionQuote, n: int, action: str, limit: float, reason: str
    ) -> Order:
        symbol, _ = self._bound()
        o = self._new(0, "BAG", action, n, limit, reason)
        now = o.created
        out = self.script(o) if self.script else "fill"
        sgn = 1 if action == "BUY" else -1
        net_mid = call.mid + put.mid
        if out == "reject":
            o.state, o.error = OrderState.REJECTED, "sim reject"
        elif out != "timeout" and (limit - net_mid) * sgn >= 0:
            # "fill" | "partial:<k>"; a wrong-side limit rests (SUBMITTED) until wait() cancels it
            k = n if out == "fill" else min(int(out.partition(":")[2]), n)
            c = self._c
            dte = (call.expiry - now).total_seconds() / 86400
            touch = 2 * option_half_spread(dte, c)
            # never better than the limit, never worse than the touch
            net = min(limit, net_mid + touch) if sgn > 0 else max(limit, net_mid - touch)
            comm = (
                max(c.option_min_commission, c.option_commission * 2 * k)
                + c.option_fees_per_contract * 2 * k
            ) / 2
            for leg in (call, put):
                tmpl = Position(
                    leg.con_id, symbol, "OPT", 0, 0.0, leg.right, leg.strike, leg.expiry, 100
                )
                self._fill(o, tmpl, sgn * k, leg.mid + (net - net_mid) / 2, comm, now)
                o.leg_fills[leg.con_id] = sgn * k
            o.filled, o.avg_price = k, net
            o.state = OrderState.FILLED if k == n else OrderState.PARTIAL
        return o

    async def wait(self, order: Order, timeout_s: float) -> Order:
        """STK: as is (fills happen on quotes; the Runner cancels after fill_timeout_s).
        BAG: what did not fill at placement never will -> cancel the remainder."""
        if order.sec_type == "BAG" and order.is_open:
            self._done(order, OrderState.CANCELLED)
        return order

    async def cancel(self, order: Order) -> Order:
        if order.is_open:
            self._done(order, OrderState.CANCELLED)  # PARTIAL keeps its fills
        return order

    async def cancel_all(self) -> int:
        opens = self.open_orders()
        for o in opens:
            await self.cancel(o)
        return len(opens)

    def open_orders(self) -> list[Order]:
        return [o for o in self.orders.values() if o.is_open]

    # --- internals -----------------------------------------------------------------------------

    def _bound(self) -> tuple[str, int]:
        if self._stock_id is None or self._symbol is None:
            raise RuntimeError("stream_stock(symbol) first")
        return self._symbol, self._stock_id

    def _new(self, con_id: int, sec_type: str, action: str, qty: int, limit, reason: str) -> Order:
        now = self._clock.now()
        o = Order(
            uuid4().hex[:12],
            con_id,
            sec_type,
            action,
            qty,
            limit,
            reason,
            OrderState.SUBMITTED,
            created=now,
            updated=now,
        )
        self.orders[o.id] = o
        return o

    def _done(self, o: Order, state: OrderState, error: str = "") -> None:
        o.state, o.error, o.updated = state, error, self._clock.now()
        if o in self._pending:
            self._pending.remove(o)

    def _fill(self, o: Order, tmpl: Position, qty: int, px: float, comm: float, now) -> None:
        self._seq[o.id] += 1
        exec_id = f"sim-{o.id}-{self._seq[o.id]}"
        self.fills.append(Fill(exec_id, o.id, tmpl.con_id, qty, px, now, comm, "sim"))
        _add(self._pos, tmpl, qty, px)

    def _fill_pending(self, q: Quote, now: datetime) -> None:
        """Every pending share order (FIFO) meets this quote: BUY at ask, SELL at bid."""
        c = self._c
        for o in list(self._pending):
            out = self.script(o) if self.script else "fill"
            if out == "timeout":
                continue
            if out == "reject":
                self._done(o, OrderState.REJECTED, "sim reject")
                continue
            left = o.qty - o.filled
            k = left if out == "fill" else min(int(out.partition(":")[2]), left)
            if not k:
                continue
            px = q.ask if o.action == "BUY" else q.bid
            qty = k if o.action == "BUY" else -k
            tmpl = Position(o.con_id, self._symbol or "", "STK", 0, 0.0)
            self._fill(o, tmpl, qty, px, commission(qty, c) + reg_fees(qty, px, c), now)
            o.avg_price = (o.avg_price * o.filled + px * k) / (o.filled + k)
            o.filled += k
            o.state = OrderState.FILLED if o.filled == o.qty else OrderState.PARTIAL
            o.updated = now
            if o.state is OrderState.FILLED:
                self._pending.remove(o)
