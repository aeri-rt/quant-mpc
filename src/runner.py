"""Trading lifecycle: start -> tick loop -> shutdown, one straddle campaign at a time.

STARTING -> RUNNING <-> DEGRADED -> HALTED -> STOPPING -> STOPPED; TRANSITIONS is the whole
machine, Gate + Book carry the campaign state. `tick()` is the per-interval pipeline of ADR-004
§3-§4 (README "Tick pipeline") with no sleeps and no wall-clock reads, so the backtest drives it
per bar. Every order goes through the injected broker (SimBroker outside --execute) and fills
reach the book only by draining `broker.fills`. A halt cancels everything and stops deciding; it
never flattens (bad data is a halt cause, and trading on bad data is worse than holding a bounded
book). Liquidation is `flatten()`. A lost broker link anywhere in a tick (BrokerDisconnected) is
DEGRADED, never a halt.
"""

import asyncio
from datetime import date, datetime
from enum import Enum

from src import bs, costs, validation
from src.book import Book, Leg
from src.broker import BrokerDisconnected, BrokerError
from src.bs import Greeks
from src.clock import Clock, T_years, is_open, next_session_date, seconds_to_close, session_date
from src.config import Live
from src.costs import option_half_spread
from src.gate import Edge, Gate, edge_per_day
from src.hedge import HedgeState, hedge_all
from src.log import get_logger
from src.risk import Risk
from src.store import DECISION_COLUMNS, Store
from src.types import Account, Fill, OptionQuote, Order, OrderState, Quote
from src.vol import RealizedVol

log = get_logger("runner")
CHAIN_TTL_S = 300.0
CANDIDATE_MAX_SPREAD = 0.05  # (ask - bid) / mid of each candidate leg
SHUTDOWN_S = 30.0
IV_WARN_S = 600.0  # a held leg without tick-106 Greeks is warned about at most this often


class State(str, Enum):
    STARTING = "STARTING"
    RUNNING = "RUNNING"
    DEGRADED = "DEGRADED"
    HALTED = "HALTED"
    STOPPING = "STOPPING"
    STOPPED = "STOPPED"


TRANSITIONS = {
    (State.STARTING, "started"): State.RUNNING,
    (State.STARTING, "stop"): State.STOPPING,
    (State.STARTING, "halt"): State.HALTED,
    (State.RUNNING, "degrade"): State.DEGRADED,
    (State.DEGRADED, "recover"): State.RUNNING,
    (State.RUNNING, "halt"): State.HALTED,
    (State.DEGRADED, "halt"): State.HALTED,
    (State.RUNNING, "stop"): State.STOPPING,
    (State.DEGRADED, "stop"): State.STOPPING,
    (State.HALTED, "stop"): State.STOPPING,
    (State.STOPPING, "stopped"): State.STOPPED,
}


class IllegalTransition(RuntimeError):
    pass


class _Stopped(Exception):
    """stop() arrived during start(): straight to shutdown, no halt."""


class Runner:
    def __init__(
        self,
        cfg: Live,
        broker,
        store: Store,
        clock: Clock,
        execute: bool,
        contracts: int | None = None,
        force_open: bool = False,
        mode: str = "dry_run",
        sleep=asyncio.sleep,
    ):
        self.cfg, self.broker, self.store, self.clock = cfg, broker, store, clock
        self.execute, self.force_open, self.mode, self.sleep = execute, force_open, mode, sleep
        self.contracts = contracts if contracts is not None else cfg.position.contracts
        self.state = State.STARTING
        self.book = Book()
        self.vol = RealizedVol(cfg.vol)
        self.gate = Gate(cfg.gate)
        self.risk = Risk(cfg.risk, cfg.loop, execute)
        self.working: Order | None = None
        self.hedged_for_close_session: date | None = None
        self.session_id = ""
        self._stop = asyncio.Event()
        self._stock_id: int | None = None
        # every leg conId ever registered: Book.apply_fill books an unknown conId as shares
        self._leg_meta: dict[int, Leg] = {}
        self._last_reconcile: datetime | None = None
        self._chain: tuple[datetime, list[OptionQuote]] | None = None
        self._pnl_session: date | None = None
        # book.pnl at entry; 0 for a replayed ledger; None = take it on the next held tick
        self._entry_basis: float | None = 0.0
        self._observe_only = False  # dry-run with an unrecoverable book: never reconcile
        self._iv_warned: dict[int, datetime] = {}  # leg conId -> last "IV frozen" warning
        self._row: dict | None = None  # decisions row of the tick in progress
        self._opt_traded = False  # an option order was placed this tick: step-4 Greeks are stale
        self._cost_fill: Fill | None = None  # first broker STK fill; checks the cost model
        self._cost_checked = False
        self._shut = False

    @property
    def contracts_held(self) -> int:
        s = self.book.straddle()
        return min(s[0].qty, s[1].qty) if s else 0

    # --- state -----------------------------------------------------------------------------------

    def fire(self, event: str) -> None:
        try:
            new = TRANSITIONS[(self.state, event)]
        except KeyError:
            raise IllegalTransition(f"{self.state.value} --{event}") from None
        log.info(f"{self.state.value} --{event}--> {new.value}")
        self.state = new

    def stop(self, reason: str = "") -> None:
        """Idempotent; safe from a signal handler."""
        self._stop.set()
        if (self.state, "stop") in TRANSITIONS:
            log.warning(f"stop requested: {reason}")
            self.fire("stop")

    def _check_stop(self) -> None:
        if self._stop.is_set():
            raise _Stopped

    def _note(self, reason: str) -> None:
        if self._row is not None:
            self._row["reason"] = reason

    # --- lifecycle -------------------------------------------------------------------------------

    async def run(self) -> None:
        try:
            await self.start()
            await self.loop()
        except _Stopped:
            pass
        except Exception:
            log.exception("runner")
            await self.halt("exception")
        finally:
            await self.shutdown()

    async def start(self) -> None:
        cfg, b = self.cfg, self.broker
        symbol = cfg.instrument.symbol
        await b.connect()
        self._check_stop()
        now = self.clock.now()
        self.session_id = self.store.start_session(self.mode, cfg.model_dump_json(), now)
        if bind := getattr(b, "bind_session", None):
            bind(self.store, self.session_id)
        if on_reject := getattr(b, "on_reject", None):
            on_reject(self.risk.record_reject)
        for o in b.open_orders():  # rebound from an earlier session: persist, then cancel
            log.warning(f"cancelling stale order {o.id} {o.action} {o.qty} {o.sec_type}")
            self.store.save_order(self.session_id, o)
            self.store.update_order(self.session_id, await b.cancel(o))
        self._check_stop()
        pos = await b.positions()  # after the cancels: a fill racing a cancel is in this snapshot
        if b.fills or b.commission_updates:  # ... so booking it again would double it
            log.warning(f"start: {len(b.fills)} fills before the position snapshot not booked")
            b.fills.clear()
            b.commission_updates.clear()
        try:
            self.book = Book.from_positions(pos, symbol)
        except ValueError as e:
            if self.execute:
                await self.halt(str(e))
                return
            log.warning(f"{e}: dry-run observes with an empty book, no reconcile this session")
            self.book, self._observe_only = Book(), True
        self._last_reconcile = now
        for leg in self.book.legs:
            self._leg_meta[leg.con_id] = leg
            await b.stream_leg(leg.con_id)
        self._stock_id = await b.stream_stock(symbol)
        if self.book.straddle():
            self._replay_ledger(pos, symbol)
        self.book.premium_paid = sum(leg.avg_cost * leg.qty * leg.mult for leg in self.book.legs)
        self._check_stop()
        try:
            bars = await asyncio.wait_for(
                b.historical_bars(symbol, now, cfg.loop.warmup_sessions, 60),
                cfg.loop.warmup_max_s,
            )
        except (TimeoutError, BrokerDisconnected) as e:  # optional: never a halt
            log.warning(f"vol warm-up failed ({e!r}): continuing without it")
            bars = None
        if bars is not None and not bars.empty:
            self.vol.warm_from_bars(bars)
        self._check_stop()
        log.info(
            f"started {self.mode}: {self.contracts_held} straddles, {self.book.shares} shares, "
            f"vol sessions {self.vol.sessions()}"
        )
        self.fire("started")

    def _replay_ledger(self, pos, symbol: str) -> None:
        """Recovered straddle: replay the campaign's stored broker fills so realized hedge P&L,
        commissions and the entry premium survive a restart (max_loss needs them). A replay that
        does not rebuild the broker position is dropped: max_loss then runs from the next tick."""
        legs = {leg.con_id: leg for leg in self.book.legs}
        fills = self.store.campaign_fills(list(legs), self._stock_id)
        book = Book()
        for f in fills:
            book.apply_fill(f, legs.get(f.con_id))
        if book.diff(pos, symbol).ok:
            log.info(f"ledger: {len(fills)} stored fills replayed, realized {book.realized:+.2f}")
            self.book = book
        else:
            log.warning(
                f"ledger: {len(fills)} stored fills do not rebuild the broker position; "
                "max_loss is measured from the first tick after this restart"
            )
            self._entry_basis = None

    async def loop(self) -> None:
        while self.state in (State.RUNNING, State.DEGRADED) and not self._stop.is_set():
            t0 = self.clock.now()
            try:
                await self.tick()
            except BrokerDisconnected as e:
                log.warning(f"tick: {e}: DEGRADED until the link and a fresh quote are back")
                if self.state is State.RUNNING:
                    self.fire("degrade")
            except Exception:
                log.exception("tick")
                await self.halt("exception")
            finally:  # the sleep cannot be skipped: a blocked trade must not busy-loop
                elapsed = (self.clock.now() - t0).total_seconds()
                await self._sleep_or_stop(max(0.0, self.cfg.loop.interval_s - elapsed))

    async def _sleep_or_stop(self, s: float) -> None:
        tasks = [asyncio.create_task(self.sleep(s)), asyncio.create_task(self._stop.wait())]
        _, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for t in pending:
            t.cancel()
        await asyncio.gather(*pending, return_exceptions=True)

    async def halt(self, reason: str) -> None:
        """Cancel everything and stop deciding; never flattens. HALTED leaves only via stop()."""
        self.risk.halt(reason)
        self._stop.set()  # ends the loop's sleep now; the state stays HALTED
        try:
            n = await self.broker.cancel_all()
            log.warning(f"halt: {n} open orders cancelled")
        except Exception:
            log.exception("halt: cancel_all")
        if (self.state, "halt") in TRANSITIONS:
            self.fire("halt")
        if self._row is not None:  # the tick in progress writes its row
            self._row["reason"] = f"halt:{reason}"
        elif self.session_id:
            row = self._new_row(self.clock.now())
            row["reason"] = f"halt:{reason}"
            self.store.save_decision(self.session_id, row)

    async def shutdown(self) -> None:
        """Idempotent, never raises; positions stay in place (restart recovers them by conId)."""
        if self._shut:  # a second call (stop after halt) must still complete the transition
            if self.state is State.STOPPING:
                self.fire("stopped")
            return
        self._shut = True
        try:
            await asyncio.wait_for(self._shutdown(), SHUTDOWN_S)
        except Exception:
            log.exception("shutdown")
        if self.state is State.STOPPING:
            self.fire("stopped")

    async def _shutdown(self) -> None:
        b, now = self.broker, self.clock.now()

        async def attempt(name, fn, *a):
            try:
                await fn(*a)
            except Exception:
                log.exception(f"shutdown: {name}")

        await attempt("cancel_all", b.cancel_all)
        await attempt("final reconcile", self._final_check, now)
        cids = [leg.con_id for leg in self.book.legs] + ([self._stock_id] if self._stock_id else [])
        for cid in cids:
            await attempt(f"unstream {cid}", b.unstream, cid)
        await attempt("disconnect", b.disconnect)

    async def _final_check(self, now: datetime) -> None:
        self._drain()
        if not self.session_id:
            return
        try:
            d = self.book.diff(await self.broker.positions(), self.cfg.instrument.symbol)
            if not d.ok:
                log.warning(f"final reconcile: {d}")
        except Exception:
            log.exception("final reconcile")
        row = self._new_row(now)
        row["reason"] = "shutdown"
        self.store.save_decision(self.session_id, row)
        status = "halted" if (self.state is State.HALTED or self.risk.halted) else "stopped"
        self.store.end_session(self.session_id, status, now, self.book.shares, self.book.realized)

    async def flatten(self) -> int:
        """Operator path (run_paper flatten --execute), no tick loop: close the straddle (BAG SELL
        with the reprice loop) and the shares, verify, print. 0 clean, 1 residual positions,
        2 market closed. Unpaired legs: no single-leg order path exists, close them by hand.
        Every way out (returns, errors, the task cancelled by a signal) cancels our open orders,
        ends the session row and disconnects."""
        cfg, b = self.cfg, self.broker
        symbol = cfg.instrument.symbol
        await b.connect()
        now = self.clock.now()
        self.session_id = self.store.start_session("flatten", cfg.model_dump_json(), now)
        if bind := getattr(b, "bind_session", None):
            bind(self.store, self.session_id)
        try:
            await b.cancel_all()  # rebound orders first: a fill racing it is in the snapshot
            try:
                self.book = Book.from_positions(await b.positions(), symbol)
            except ValueError as e:
                print(f"{e}: unpaired leg, close it by hand in TWS")
                return 1
            b.fills.clear()  # ... so it must not be booked again
            b.commission_updates.clear()
            self._stock_id = await b.stream_stock(symbol)
            for leg in self.book.legs:
                self._leg_meta[leg.con_id] = leg
                await b.stream_leg(leg.con_id)
            if not is_open(now):
                print("market closed: flatten refused")
                return 2
            for _ in range(20):  # fresh subscriptions need a moment to tick
                if b.quote(self._stock_id, now, require_live=False) is not None and all(
                    b.leg_quote(leg.con_id, now) for leg in self.book.legs
                ):
                    break
                await self.sleep(0.5)
                now = self.clock.now()
            # options first: a straddle that will not close keeps its hedge (never naked options)
            closed = await self._exit(now, "flatten") if self.book.straddle() else True
            if closed and self.book.shares:
                await self._flatten_shares(now)
            elif not closed:
                print("straddle not closed: hedge shares left in place, retry")
            residual = [p for p in (await b.positions()).values() if p.symbol == symbol and p.qty]
            for p in residual:
                print(f"residual {p.sec_type} conId={p.con_id} qty={p.qty}")
            print("flat" if not residual else f"{len(residual)} residual positions")
            return 1 if residual else 0
        finally:  # every exit, incl. Ctrl-C/SIGTERM cancelling us mid-order: none left working
            try:
                await b.cancel_all()
            except Exception:
                log.exception("flatten: cancel_all")
            self.store.end_session(
                self.session_id, "flatten", self.clock.now(), self.book.shares, self.book.realized
            )
            await b.disconnect()

    # --- tick ------------------------------------------------------------------------------------

    def _new_row(self, now: datetime) -> dict:
        row = dict.fromkeys(DECISION_COLUMNS)
        row.update(
            ts=now.isoformat(),
            mode=self.mode,
            policy=self.cfg.hedge.policy,
            force_open=int(self.force_open),
            fill_source="broker" if self.execute else "sim",
            md_type=self.broker.market_data_type,
            clock_offset_s=self.broker.clock_offset_s,
            shares=self.book.shares,
            gate_state=self.gate.state,
            campaign="held" if self.book.straddle() else "flat",
        )
        return row

    async def tick(self) -> dict:
        """One decisions row per call, written even when a step raises."""
        now = self.clock.now()
        self._row = row = self._new_row(now)
        try:
            await self._tick(row, now)
        finally:
            row["shares"] = self.book.shares
            row["campaign"] = "held" if self.book.straddle() else "flat"
            self.store.save_decision(self.session_id, row)
            self._row = None
        return row

    async def _tick(self, row: dict, now: datetime) -> None:
        cfg, b, book = self.cfg, self.broker, self.book
        r, qy = cfg.instrument.risk_free_rate, cfg.instrument.dividend_yield
        self._opt_traded = False
        # 1-2 quote, then drain (SimBroker fills pending share orders inside quote()).
        # Never require_live: delayed data is refused at step 7 by risk.check_hedge ("delayed"),
        # it is not staleness (that would turn md_type 3 into DEGRADED -> HALT with no S logged).
        q = b.quote(self._stock_id, now, require_live=False)
        self._drain()
        age = None if q is None else validation.quote_age_s(q.ts, now)
        row["quote_age_s"] = age
        st = self.risk.check_stale(age, now)
        if st == "halt":
            await self.halt("stale quote")
            return
        if st == "stale" or not b.connected:
            if self.state is State.RUNNING:
                self.fire("degrade")
            row["reason"] = "disconnected" if not b.connected else "stale"
        elif self.state is State.DEGRADED and b.data_ok and b.connected:
            self.fire("recover")
            if n := await b.cancel_all():  # an order whose wait the outage cut may still rest
                log.warning(f"recover: {n} orders left working through the outage cancelled")
                self._drain()
            await self._maybe_reconcile(now)  # outage fills are in positions(), not the book
        if q is None or row["reason"]:  # no decisions without a fresh quote; P&L and reconcile run
            await self._pnl_check(row, await b.account(), None, now)
            await self._maybe_reconcile(now)
            return
        S = q.mid
        row.update(S=S, bid=q.bid, ask=q.ask, md_type=q.market_data_type)
        # 3 realized vol
        self.vol.update(q.ts, S)
        sigma_f = self.vol.forecast_sigma()
        row["sigma_f"] = sigma_f
        # 4 legs: tick-106 Greeks/IV when present, mids from the streamed leg quote
        for leg in book.legs:
            g = b.leg_greeks(leg.con_id, S, now)
            oq = b.leg_quote(leg.con_id, now)
            if g is not None:
                leg.ibkr, leg.iv = g, g.iv
            else:
                if leg.iv == 0 and oq is not None:  # restart seed: legs from positions carry iv=0
                    leg.iv = self._leg_iv(oq, S, now) or 0.0
                last = self._iv_warned.get(leg.con_id)
                if last is None or (now - last).total_seconds() >= IV_WARN_S:
                    self._iv_warned[leg.con_id] = now
                    log.warning(f"leg {leg.con_id}: no tick-106 Greeks, IV frozen at {leg.iv:.4f}")
            leg.mid = (
                oq.mid
                if oq is not None
                else bs.price(S, leg.strike, leg.T(now), r, qy, leg.iv, leg.right)
            )
        try:
            G = book.greeks(S, now, r, qy, cfg.greeks.source, cfg.greeks, cfg.greeks.max_age_s)
        except ValueError as e:
            log.warning(f"greeks: {e}")
            row["reason"], G = "greeks_nan", None
        if G is not None:
            row.update(
                delta_opt=G.delta - book.shares,
                gamma=G.gamma,
                theta_day=G.theta,
                vega=G.vega,
                delta_net=G.delta,
            )
        pnl = book.pnl(S)
        row["book_pnl"] = pnl
        if self._pnl_session != session_date(now):
            self._pnl_session, book.session_start_pnl = session_date(now), pnl
        # 5 account + kill switch
        acct = await b.account()
        if await self._pnl_check(row, acct, pnl - book.session_start_pnl, now):
            return
        # 6 campaign (gate, entry, roll, exit)
        if self.state is State.RUNNING and is_open(now) and G is not None:
            await self._campaign(row, S, now, G, sigma_f, acct)
        # 7 hedge; never on the tick an entry/exit/roll placed an option order: G describes the
        # pre-campaign book and leg reports may still trail (a recompute is unsafe too)
        if (
            self.state is State.RUNNING
            and is_open(now)
            and G is not None
            and (book.legs or book.shares)
            and not self._opt_traded
        ):
            await self._hedge_step(row, q, S, now, G, acct)
        # 8 reconcile
        await self._maybe_reconcile(now)

    async def _pnl_check(self, row: dict, acct: Account | None, book_daily, now) -> bool:
        """Daily-loss / reject-storm kill switch; True when this call halted."""
        if acct is not None:
            row.update(nlv=acct.nlv, daily_pnl=acct.daily_pnl)
        self.risk.on_pnl(acct.daily_pnl if acct else None, book_daily, acct, now)
        if self.risk.halted and self.state is not State.HALTED:
            await self.halt(self.risk.halt_reason)
            return True
        return False

    def _drain(self) -> None:
        b = self.broker
        while b.fills:
            f = b.fills.popleft()
            meta = self._leg_meta.get(f.con_id)
            if meta is None and f.con_id != self._stock_id:
                log.error(f"fill {f.exec_id}: unknown conId {f.con_id}, not booked")
                continue
            self.book.apply_fill(f, meta)
            if f.source == "sim":
                self.store.save_fill(self.session_id, f)
            elif (
                self._cost_fill is None
                and not self._cost_checked
                and f.con_id == self._stock_id
                and (o := b.orders.get(f.order_id)) is not None
                and abs(f.qty) == o.qty  # IBKR's minimum is per order: only a one-execution fill
            ):
                self._cost_fill = f
            log.info(f"fill {f.order_id} conId={f.con_id} {f.qty:+d} @ {f.price:.4f}")
        while b.commission_updates:
            exec_id, comm = b.commission_updates.popleft()
            self.book.set_commission(exec_id, comm)
            f = self._cost_fill
            if f is not None and exec_id == f.exec_id and not self._cost_checked:
                self._cost_checked = True
                c = self.cfg.costs
                modeled = costs.commission(f.qty, c) + costs.reg_fees(f.qty, f.price, c)
                if abs(comm - modeled) > 0.2 * modeled:
                    log.warning(
                        f"cost model mismatch: actual={comm:.2f} modeled={modeled:.2f}, "
                        "run scripts/calibrate_costs.py"
                    )
        if (w := self.working) is not None and not w.is_open:
            if w.reason == "close_hedge" and w.state is OrderState.FILLED:  # a cancel retries
                self.hedged_for_close_session = session_date(w.created)
            self.store.update_order(self.session_id, w)
            self.working = None

    async def _campaign(self, row, S, now, G: Greeks, sigma_f, acct) -> None:
        cfg, book = self.cfg, self.book
        held = book.straddle() is not None
        if book.legs and not held:
            row["reason"] = "unpaired"
            return
        if not held and book.shares and self.execute:  # shares with no straddle (crash mid-roll)
            await self._flatten_shares(now)
            return
        cand = edge = sigma_iv = None
        if held:
            if self._entry_basis is None:  # restart without a ledger: loss since this tick
                self._entry_basis = book.pnl(S)
            sigma_iv = sum(leg.iv for leg in book.legs) / len(book.legs)
            dte = book.dte(now)
            if sigma_f is not None:
                G_opt = G + Greeks(-float(book.shares), 0.0, 0.0, 0.0)
                edge = self._edge(
                    G_opt, book.shares, S, sigma_f, sigma_iv, self.contracts_held, dte, now, True
                )
        else:
            cand = await self._candidate(S, now)
            if cand is None:
                row["reason"] = "no_chain"
            elif (sigma_iv := self._iv_of(cand, S, now)) is None:
                row["reason"] = "no_iv"
            elif sigma_f is None:  # held: the hedge still runs, sigma_f NULL says why no gate
                row["reason"] = "no_vol"
            else:
                edge = self._flat_edge(cand, S, now, acct, sigma_f, sigma_iv)
        row["iv_atm"] = sigma_iv
        if edge is not None:
            row.update(edge_day=edge.edge_day, leland_a=edge.leland_a)
            if self.gate.due(now):
                self.gate.evaluate(edge, now, held)
        row["gate_state"] = self.gate.state
        if not held:
            if cand is not None and (self.gate.state == "OPEN" or self.force_open):
                if why := self._entry_guard(row, now, cand):
                    self._note(why)
                else:
                    await self._enter(cand, S, now, acct)
            return
        p = cfg.position
        if dte <= p.hard_exit_dte:
            await self._exit(now, "hard_exit")
        elif dte <= p.roll_dte:
            await self._roll(row, S, now, acct)
        elif (
            self.gate.state == "CLOSED"
            and self.gate.reason == "exit_confirmed"
            and not self.force_open
        ):
            await self._exit(now, "exit_confirmed")
        elif (
            book.premium_paid > 0
            and book.pnl(S) - self._entry_basis <= -p.max_loss_frac_premium * book.premium_paid
        ):
            await self._exit(now, "max_loss")

    def _entry_guard(self, row, now, cand) -> str | None:
        """Why no straddle may open now (a fresh entry or a roll's re-entry), else None."""
        # SPY and the legs the BAG is priced from; exits stay allowed: they reduce risk
        if self.execute and (row["md_type"] != 1 or any(q.market_data_type != 1 for q in cand)):
            return "delayed"
        if seconds_to_close(now) <= self.cfg.loop.no_entry_last_s:
            return "late"
        if next_session_date(session_date(now)) in self.cfg.instrument.ex_dividend_dates:
            return "ex_div"
        if self.working is not None:
            return "working_order"
        return None

    async def _hedge_step(self, row, q: Quote, S, now, G: Greeks, acct) -> None:
        cfg, book = self.cfg, self.book
        st = HedgeState(
            delta_net=G.delta,
            gamma=G.gamma,
            S=S,
            T=book.T(now),
            seconds_to_close=seconds_to_close(now),
            W_var=self.vol.horizon_variance(now, cfg.loop.interval_s, cfg.vol.overnight_frac),
        )
        dec = hedge_all(
            st, cfg.hedge, cfg.costs, cfg.instrument.risk_free_rate, cfg.loop.close_window_s
        )
        row.update(
            band_u=dec.band, qp_u=dec.qp, naive_u=dec.naive, band_h=dec.band_h, qp_h=dec.qp_h
        )
        row["reason"] = row["reason"] or dec.reason
        u, close_hedge = dec.chosen, dec.reason == "close_hedge"
        if close_hedge and self.hedged_for_close_session == session_date(now):
            u = 0
        if self.working is not None and self.working.is_open:  # at most one working share order
            if (now - self.working.created).total_seconds() > cfg.loop.fill_timeout_s:
                self.working = await self.broker.cancel(self.working)
                self.store.update_order(self.session_id, self.working)
            row["reason"], u = "working_order", 0
        u, row["risk_result"] = self.risk.check_hedge(
            u,
            G.delta,
            book.shares,
            self.contracts_held,
            acct,
            quote_ok=q.market_data_type == 1 or not self.execute,
        )
        if u:  # a close hedge counts once it fills (_drain): a timed-out one is re-sent
            await self._hedge(u, q, "close_hedge" if close_hedge else "hedge")
            row["u_sent"] = u

    # --- gate inputs -----------------------------------------------------------------------------

    def _edge(self, G_opt: Greeks, shares, S, sigma_f, sigma_iv, n, dte, now, sunk) -> Edge:
        """edge_per_day with the band hedger's (1-f)*H for this book as the crossing size."""
        cfg = self.cfg
        T = dte / 365
        st = HedgeState(
            delta_net=G_opt.delta + shares,
            gamma=G_opt.gamma,
            S=S,
            T=T,
            seconds_to_close=seconds_to_close(now),
            W_var=self.vol.horizon_variance(now, cfg.loop.interval_s, cfg.vol.overnight_frac),
        )
        band_h = hedge_all(
            st, cfg.hedge, cfg.costs, cfg.instrument.risk_free_rate, cfg.loop.close_window_s
        ).band_h
        return edge_per_day(
            G_opt,
            S,
            sigma_f,
            sigma_iv,
            cfg.costs,
            (1 - cfg.hedge.inner_fraction) * band_h,
            n,
            dte,
            cfg.gate.holding_sessions,
            cfg.gate.turnover_frac,
            cfg.loop.interval_s,
            sunk,
        )

    def _flat_edge(self, cand, S, now, acct, sigma_f, sigma_iv) -> Edge:
        cfg = self.cfg
        call, put = cand
        n = self.risk.size_contracts(
            call.mid + put.mid, acct, self.contracts, cfg.position.max_contracts
        )[0]
        T = T_years(call.expiry, now)
        r, qy = cfg.instrument.risk_free_rate, cfg.instrument.dividend_yield
        Gc = (
            bs.greeks(S, call.strike, T, r, qy, sigma_iv, "C")
            + bs.greeks(S, put.strike, T, r, qy, sigma_iv, "P")
        ).scale(n * 100)
        return self._edge(Gc, 0, S, sigma_f, sigma_iv, n, T * 365, now, False)

    def _leg_iv(self, oq: OptionQuote, S, now) -> float | None:
        """The quote's tick-106 IV, else implied_vol of its mid."""
        ins = self.cfg.instrument
        T = T_years(oq.expiry, now)
        return oq.iv or bs.implied_vol(
            oq.mid, S, oq.strike, T, ins.risk_free_rate, ins.dividend_yield, oq.right
        )

    def _iv_of(self, cand, S, now) -> float | None:
        """Mean of the legs' tick-106 IVs, implied_vol of the mid for a leg without one."""
        ivs = [v for oq in cand if (v := self._leg_iv(oq, S, now))]
        return sum(ivs) / len(ivs) if ivs else None

    async def _candidate(self, S, now) -> tuple[OptionQuote, OptionQuote] | None:
        """ATM pair of the (300 s cached) chain: min |K - S| with both rights, spread/mid <= 5%."""
        if self._chain is None or (now - self._chain[0]).total_seconds() >= CHAIN_TTL_S:
            cfg = self.cfg
            self._chain = (
                now,
                await self.broker.chain(cfg.instrument.symbol, cfg.position.entry_dte, now),
            )
        pairs: dict[tuple, dict[str, OptionQuote]] = {}
        for oq in self._chain[1]:
            if (oq.ask - oq.bid) / oq.mid <= CANDIDATE_MAX_SPREAD:
                pairs.setdefault((oq.expiry, oq.strike), {})[oq.right] = oq
        ok = [(k, v) for k, v in pairs.items() if "C" in v and "P" in v]
        if not ok:
            return None
        _, v = min(ok, key=lambda kv: abs(kv[0][1] - S))
        return v["C"], v["P"]

    # --- orders ----------------------------------------------------------------------------------

    async def _hedge(self, u: int, q: Quote, reason: str) -> None:
        limit = round(q.ask + 0.01 if u > 0 else q.bid - 0.01, 2)
        self.working = await self.broker.place_shares(u, limit, reason)
        self.store.save_order(self.session_id, self.working)
        log.info(f"{reason} {u:+d} @ {limit} ({self.working.id})")

    async def _enter(self, cand, S, now, acct) -> None:
        cfg, b, sid = self.cfg, self.broker, self.session_id
        call, put = cand
        premium = call.mid + put.mid
        n, why = self.risk.size_contracts(premium, acct, self.contracts, cfg.position.max_contracts)
        if n < 1:
            self._note(f"size:{why}")
            return
        dte = T_years(call.expiry, now) * 365
        limit = round(premium + option_half_spread(dte, cfg.costs), 2)
        try:
            margin = await b.whatif_straddle(call, put, n, limit)
        except BrokerDisconnected:
            raise
        except BrokerError as e:
            log.warning(f"whatif failed: {e}")
            self._note("whatif_failed")
            return
        ok, why = self.risk.check_straddle(n, premium, acct, margin, cfg.position.max_contracts)
        if not ok:
            self._note(f"straddle:{why}")
            return
        for oq in cand:  # registered BEFORE the order so apply_fill can open the legs
            self._leg_meta[oq.con_id] = Leg(
                oq.con_id,
                oq.right,
                oq.strike,
                oq.expiry,
                0,
                iv=self._leg_iv(oq, S, now) or 0.0,
                mid=oq.mid,
            )
        o = await b.place_straddle(call, put, n, "BUY", limit, "entry")
        self._opt_traded = True
        self.store.save_order(sid, o)
        o = await b.wait(o, cfg.loop.straddle_timeout_s)
        self.store.update_order(sid, o)
        self._drain()
        k_call, k_put = (abs(o.leg_fills.get(oq.con_id, 0)) for oq in cand)
        filled = o.state is OrderState.FILLED  # a guaranteed BAG never fills a naked leg
        if not filled and k_call != k_put:
            await self._unwind(o)
            return
        k = o.filled if filled else k_call or o.filled  # leg reports can trail the status
        if k < 1:
            if o.is_open:
                self.store.update_order(sid, await b.cancel(o))
            log.warning(f"entry unfilled: {o.state.value} {o.error}")
            self._note("entry_unfilled")
            return
        self.book.premium_paid = o.avg_price * k * 100
        self._entry_basis = self.book.pnl(S)
        self._note("entry")
        for oq in cand:
            await b.stream_leg(oq.con_id)
        log.info(f"entered {k} straddles K={call.strike} @ {o.avg_price:.2f} ({o.state.value})")

    async def _exit(self, now: datetime, reason: str) -> bool:
        """BAG SELL at net mid - half-spread, two reprices, then the touch. False = retry next
        tick (a held leg has no usable quote); units left after the touch halt."""
        cfg, b, sid = self.cfg, self.broker, self.session_id
        call, put = self.book.straddle()
        cids = (call.con_id, put.con_id)
        left = min(call.qty, put.qty)
        hs = option_half_spread(self.book.dte(now), cfg.costs)
        for attempt in range(4):
            cq, pq = b.leg_quote(call.con_id, now), b.leg_quote(put.con_id, now)
            if cq is None or pq is None:
                self._note("spread_wide")
                return False
            limit = cq.bid + pq.bid if attempt == 3 else cq.mid + pq.mid - hs * (attempt + 1)
            o = await b.place_straddle(cq, pq, left, "SELL", round(limit, 2), reason)
            self._opt_traded = True
            self.store.save_order(sid, o)
            o = await b.wait(o, cfg.loop.straddle_timeout_s)
            if o.is_open:
                o = await b.cancel(o)
            self.store.update_order(sid, o)
            self._drain()
            left -= o.filled
            log.info(
                f"{reason} attempt {attempt}: {o.filled}/{o.qty} @ {limit:.2f} {o.state.value}"
            )
            if left <= 0:
                break
        if left > 0:
            await self.halt(f"exit failed {reason}: {left} units left on {cids}")
            return False
        self.gate.force_closed(now, reason)
        self._note(f"exit:{reason}")
        for cid in cids:
            await b.unstream(cid)
        await self._flatten_shares(now)
        return True

    async def _roll(self, row, S, now, acct) -> None:
        """Roll only when the gate is OPEN for the new candidate (evaluated now) and its expiry is
        strictly later; otherwise exit. The re-entry obeys the entry guards; a blocked or failed
        re-entry degrades to an exit."""
        cand = await self._candidate(S, now)
        sigma_f = self.vol.forecast_sigma()
        ok = False
        if cand is not None and cand[0].expiry > self.book.legs[0].expiry:
            sigma_iv = self._iv_of(cand, S, now)
            if sigma_f is not None and sigma_iv is not None:
                edge = self._flat_edge(cand, S, now, acct, sigma_f, sigma_iv)
                ok = self.gate.evaluate(edge, now, held=False) == "OPEN"
            ok = ok or self.force_open
        if not ok:
            await self._exit(now, "roll_refused")
        elif await self._exit(now, "roll"):
            if why := self._entry_guard(row, now, cand):
                log.info(f"roll re-entry refused ({why}): exited")
                self.gate.reason = "roll_refused"
                self._note("exit:roll_refused")
                return
            await self._enter(cand, S, now, acct)
            if self.book.straddle():
                self.gate.state, self.gate.reason = "OPEN", "rolled"
                self._note("roll")

    async def _unwind(self, o: Order) -> None:
        """Unequal leg fills on a guaranteed BAG are a broker surprise, not a code path: both
        brokers fill complete units only and there is no single-leg order (deferred). Halt with
        the conIds; the operator closes the naked leg in TWS and runs `flatten`."""
        if o.is_open:
            self.store.update_order(self.session_id, await self.broker.cancel(o))
        await self.halt(f"unwind needed: leg fills {o.leg_fills} on order {o.id}")

    async def _flatten_shares(self, now: datetime) -> None:
        b = self.broker
        if self.working is not None and self.working.is_open:
            self.working = await b.cancel(self.working)
            self.store.update_order(self.session_id, self.working)
            self._drain()
        u = -self.book.shares
        if not u:
            return
        q = b.quote(self._stock_id, now, require_live=False)
        if q is None:
            log.warning("flatten shares: no quote, retrying next tick")
            return
        await self._hedge(u, q, "flatten")
        await b.wait(self.working, self.cfg.loop.fill_timeout_s)
        b.quote(
            self._stock_id, now, require_live=False
        )  # SimBroker fills on the quote after placement
        self._drain()

    # --- reconcile -------------------------------------------------------------------------------

    async def _maybe_reconcile(self, now: datetime) -> None:
        if self._observe_only or not self.broker.connected:  # a reconnect sets needs_reconcile
            return
        age = (now - self._last_reconcile).total_seconds() if self._last_reconcile else None
        if age is None or age >= self.cfg.loop.reconcile_s or self.broker.needs_reconcile:
            await self._reconcile(now)

    async def _reconcile(self, now: datetime) -> None:
        b, cfg = self.broker, self.cfg
        symbol = cfg.instrument.symbol
        self._last_reconcile = now
        b.needs_reconcile = False
        pos = await b.positions()
        d = self.book.diff(pos, symbol)
        if d.share_drift:
            if abs(d.share_drift) > cfg.risk.max_share_drift:
                await self.halt(f"share drift {d.share_drift:+d}")
                return
            log.warning(f"reconcile: adopting share drift {d.share_drift:+d}")
            self.book.shares += d.share_drift
            self._note("reconcile_adopt")
        if d.missing_legs or d.extra_legs or d.qty_mismatch:
            msg = (
                f"leg mismatch: missing {d.missing_legs} extra {d.extra_legs} qty {d.qty_mismatch}"
            )
            if self.execute:
                await self.halt(msg)
                return
            log.warning(f"reconcile (dry-run): {msg}")
            try:
                legs = Book.from_positions(pos, symbol).legs
            except ValueError:
                return  # cannot adopt what is not one straddle; keep observing
            old, new = {leg.con_id for leg in self.book.legs}, {leg.con_id for leg in legs}
            for cid in old - new:
                await b.unstream(cid)
            for cid in new - old:
                await b.stream_leg(cid)
            for leg in legs:
                self._leg_meta[leg.con_id] = leg
            self.book.legs = legs
            self.book.premium_paid = sum(leg.avg_cost * leg.qty * leg.mult for leg in legs)
