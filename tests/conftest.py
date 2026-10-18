from datetime import datetime, timedelta

import pytest
from hypothesis import settings

from src import bs
from src.book import Book, Leg
from src.clock import NY, SimClock, T_years, expiry_at, session_date
from src.config import load_live
from src.sim_broker import SimBroker
from src.types import Account, OptionQuote, Quote

settings.register_profile("ci", deadline=None, max_examples=200)
settings.load_profile("ci")


@pytest.fixture
def sim_clock():
    return SimClock(datetime(2026, 3, 10, 10, 0, tzinfo=NY))


@pytest.fixture(scope="session")
def live_cfg():
    return load_live("config/live.yaml")  # the REAL file, so YAML and tests cannot drift


@pytest.fixture
def straddle_book():
    """Long n ATM straddles bought at BS mid (r=0.04, q=0.0099) `dte` calendar days out."""

    def make(n=5, dte=30, S=600.0, iv=0.15, now=None, K=None):
        now = now or datetime(2026, 3, 10, 10, 0, tzinfo=NY)
        exp = expiry_at(session_date(now) + timedelta(days=dte))
        K = K or round(S)
        T = T_years(exp, now)
        legs = [
            Leg(con_id=cid, right=right, strike=K, expiry=exp, qty=n, iv=iv, mid=mid, avg_cost=mid)
            for cid, right in ((1001, "C"), (1002, "P"))
            for mid in (bs.price(S, K, T, 0.04, 0.0099, iv, right),)
        ]
        return Book(legs=legs, premium_paid=sum(leg.mid for leg in legs) * n * 100)

    return make


# --- SimBroker/Runner test infrastructure ----------------------------------------------------


class FakeMarket:
    """Scriptable market for Runner/SimBroker tests. Records every async call in .calls."""

    def __init__(self, clock, S=600.0, half_spread=0.005, md_type=1, con_id=756733):
        self.clock, self.S, self.h = clock, S, half_spread
        self.market_data_type, self._con_id = md_type, con_id
        self.connected = self.data_ok = True
        self.clock_offset_s, self.needs_reconcile = 0.0, False
        self.quote_ts_lag_s, self.quote_none = 0.0, False  # staleness / no-quote switches
        self.greeks = {}  # con_id -> IbkrGreeks
        self.leg_quotes = {}  # con_id -> OptionQuote (held legs; set from make_chain entries)
        self.chain_quotes = []
        self.acct = Account(100_000, 60_000, 200_000, None, -1, clock.now())
        self.pos = {}  # con_id -> Position (broker truth, e.g. recovered state)
        self.calls = []

    def stock_con_id(self, symbol):
        return self._con_id

    def quote(self, con_id, now, require_live):
        if self.quote_none or not self.connected or (require_live and self.market_data_type != 1):
            return None
        ts = now - timedelta(seconds=self.quote_ts_lag_s)
        return Quote(self.S - self.h, self.S + self.h, self.S, ts, self.market_data_type)

    def leg_greeks(self, con_id, S, now):
        return self.greeks.get(con_id)

    def leg_quote(self, con_id, now):
        return self.leg_quotes.get(con_id)

    async def chain(self, symbol, target_dte, now):
        return self.chain_quotes

    async def account(self):
        return self.acct

    async def positions(self):
        return dict(self.pos)

    async def connect(self):
        self.calls.append("connect")

    async def disconnect(self):
        self.calls.append("disconnect")

    async def stream_stock(self, symbol):
        self.calls.append(("stream_stock", symbol))
        return self._con_id

    async def stream_leg(self, con_id):
        self.calls.append(("stream_leg", con_id))

    async def unstream(self, con_id):
        self.calls.append(("unstream", con_id))

    def set_spot(self, S):
        self.S = S


def make_chain(S, dte, iv, now, call_id=1001, put_id=1002):
    """ATM +/-2 strikes, both rights, BS mids -/+ 0.06, iv and delta set, ts = now.
    ATM ids are call_id/put_id; the other strikes offset them by 10 per strike step."""
    exp = expiry_at(session_date(now) + timedelta(days=dte))
    T = T_years(exp, now)
    out = []
    for k in range(-2, 3):
        K = float(round(S) + k)
        for right, cid in (("C", call_id), ("P", put_id)):
            mid = bs.price(S, K, T, 0.04, 0.0099, iv, right)
            delta = bs.greeks(S, K, T, 0.04, 0.0099, iv, right).delta
            out.append(
                OptionQuote(
                    cid + 10 * k,
                    right,
                    K,
                    exp,
                    max(mid - 0.06, 0.01),
                    mid + 0.06,
                    iv,
                    delta,
                    now,
                    1,
                )
            )
    return out


@pytest.fixture
def fake_market(sim_clock):
    return FakeMarket(sim_clock)


@pytest.fixture
def sim_broker(fake_market, sim_clock, live_cfg):
    b = SimBroker(fake_market, live_cfg.costs, sim_clock)
    b.stock_con_id(
        live_cfg.instrument.symbol
    )  # bind symbol/conId as Runner.start's stream_stock does
    return b


@pytest.fixture
def tmp_store():
    from src.store import Store

    s = Store(":memory:")
    yield s
    s.close()


# --- runner ----------------------------------------------------------------------------------


@pytest.fixture
def runner(live_cfg, sim_clock, tmp_store):
    """Runner over SimBroker(market) with a fake sleep that records durations and advances the
    SimClock by them (`runner.sleeps`). Loop fields override via **loop_overrides."""
    from src.runner import Runner

    def make(market, execute=False, policy="band", contracts=None, force_open=False, **loop):
        cfg = live_cfg.model_copy(
            update={
                "hedge": live_cfg.hedge.model_copy(update={"policy": policy}),
                "loop": live_cfg.loop.model_copy(update=loop),
            }
        )
        sleeps = []

        async def _fake_sleep(s):
            sleeps.append(s)
            sim_clock.advance(s)

        r = Runner(
            cfg,
            SimBroker(market, cfg.costs, sim_clock),
            tmp_store,
            sim_clock,
            execute=execute,
            contracts=contracts,
            force_open=force_open,
            mode="paper" if execute else "dry_run",
            sleep=_fake_sleep,
        )
        r.sleeps = sleeps
        return r

    return make
