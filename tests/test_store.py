"""Store: FKs enforced, created_at never reset, exec_id dedupe, campaign fills, unique ids."""

import sqlite3
from dataclasses import replace
from datetime import datetime, timedelta

import pytest

from src.clock import NY, aware
from src.store import DECISION_COLUMNS, Store
from src.types import Fill, Order, OrderState

T0 = datetime(2026, 3, 10, 10, 0, tzinfo=NY)


@pytest.fixture
def store():
    s = Store(":memory:")
    yield s
    s.close()


@pytest.fixture
def sid(store):
    return store.start_session("dry_run", "{}", T0)


def bag(id="bag1", state=OrderState.SUBMITTED):
    return Order(
        id=id,
        con_id=0,
        sec_type="BAG",
        action="BUY",
        qty=5,
        limit=12.35,
        reason="entry",
        state=state,
        broker_order_id=77,
        perm_id=123456,
        created=T0,
        updated=T0,
    )


def stk(id="stk1", qty=100):
    return Order(id, 756733, "STK", "BUY", qty, 600.06, "hedge", created=T0, updated=T0)


def fill(exec_id, order_id, con_id=756733, qty=100, commission=None):
    return Fill(exec_id, order_id, con_id, qty, 600.05, T0, commission, "sim")


def orders(store, sid):
    return store._db.execute(
        "SELECT * FROM orders WHERE session_id=? ORDER BY rowid", (sid,)
    ).fetchall()


def test_file_store_uses_wal(tmp_path):
    p = tmp_path / "sub" / "t.db"  # parent created on demand
    s = Store(p)
    s.close()
    assert sqlite3.connect(p).execute("PRAGMA journal_mode").fetchone()[0] == "wal"


def test_fill_for_unknown_session_raises(store):
    with pytest.raises(sqlite3.IntegrityError):
        store.save_fill("nope", fill("e1", "o1"))


def test_order_for_unknown_session_raises(store):
    with pytest.raises(sqlite3.IntegrityError):
        store.save_order("nope", stk())


def test_bag_order_round_trip_and_update_keeps_created(store, sid):
    o = bag()
    store.save_order(sid, o)
    (back,) = orders(store, sid)
    assert (back["id"], back["sec_type"], back["qty"], back["limit_price"]) == (
        "bag1",
        "BAG",
        5,
        12.35,
    )
    assert (back["broker_order_id"], back["perm_id"], back["state"]) == (77, 123456, "SUBMITTED")

    later = T0 + timedelta(seconds=20)
    o2 = replace(o, state=OrderState.PARTIAL, filled=3, avg_price=12.3, updated=later)
    store.update_order(sid, o2)
    store.save_order(store.start_session("dry_run", "{}", T0), stk("other"))
    (back,) = orders(store, sid)
    assert (back["state"], back["filled"], back["avg_price"]) == ("PARTIAL", 3, 12.3)
    assert back["created_at"] == aware(T0).isoformat()
    assert back["updated_at"] == aware(later).isoformat()


def test_update_unknown_order_is_a_warning_not_an_error(store, sid):
    store.update_order(sid, stk("ghost"))


def test_duplicate_save_order_is_idempotent(store, sid):
    """Broker._place and the Runner both save a new order: one row, the later state wins."""
    store.save_order(sid, stk())
    store.save_order(sid, replace(stk(), state=OrderState.FILLED, filled=100))
    rows = store._db.execute("SELECT state FROM orders WHERE session_id=?", (sid,)).fetchall()
    assert [r["state"] for r in rows] == ["FILLED"]


def test_exec_id_dedupe_and_commission_patch(store, sid):
    store.save_order(sid, stk())
    store.save_fill(sid, fill("e1", "stk1"))
    store.save_fill(sid, fill("e1", "stk1", qty=999))  # duplicate execId ignored
    assert store.set_commission("e1", 0.35) == 1
    assert store.set_commission("zzz", 0.35) == 0
    rows = store._db.execute("SELECT qty, commission FROM fills").fetchall()
    assert [tuple(r) for r in rows] == [(100, 0.35)]


def test_campaign_fills_replay_order_and_scope(store, sid):
    """Broker fills only, across sessions, oldest first: the current opening of the legs onward
    (an earlier campaign on the same contract is dropped) plus the stock fills since it."""
    other = store.start_session("paper", "{}", T0)

    def bf(sess, exec_id, con_id, qty, minutes, source="broker"):
        f = Fill(exec_id, "o", con_id, qty, 1.0, T0 + timedelta(minutes=minutes), 0.5, source)
        store.save_fill(sess, f)

    bf(sid, "old-c", 1001, 5, 0)  # an earlier campaign on the same call, closed below
    bf(sid, "old-h", 756733, -30, 1)
    bf(sid, "old-x", 1001, -5, 2)
    bf(sid, "between", 756733, 30, 3)  # before the opening: not this campaign
    bf(other, "c", 1001, 5, 10)
    bf(other, "p", 1002, 5, 10)
    bf(other, "sim", 756733, -99, 11, source="sim")  # never broker truth
    bf(sid, "h2", 756733, 20, 30)  # inserted out of ts order: sorted by ts
    bf(other, "h1", 756733, -50, 20)
    bf(other, "q", 9999, 1, 21)  # another contract
    got = store.campaign_fills([1001, 1002], 756733)
    assert [f.exec_id for f in got] == ["c", "p", "h1", "h2"]
    assert got[0] == Fill("c", "o", 1001, 5, 1.0, aware(T0 + timedelta(minutes=10)), 0.5, "broker")
    assert store.campaign_fills([1003, 1004], 756733) == []  # no leg fill, no campaign


def test_session_ids_distinct_in_tight_loop(store):  # no second-resolution ids
    ids = {store.start_session("dry_run", "{}", T0) for _ in range(100)}
    assert len(ids) == 100


def test_decision_round_trip(store, sid):
    row = {c: f"v_{c}" for c in DECISION_COLUMNS if c not in ("ts", "shares", "force_open")}
    row |= {"ts": T0, "shares": -120, "force_open": 1}
    store.save_decision(sid, row)
    (back,) = store.last_decisions(sid)
    assert back["ts"] == aware(T0).isoformat()
    assert back["shares"] == -120 and back["force_open"] == 1
    for c in row.keys() - {"ts"}:
        assert back[c] == row[c], c
    assert back["session_id"] == sid


def test_decision_table_matches_columns(store):
    cols = [r["name"] for r in store._db.execute("PRAGMA table_info(decisions)")]
    assert tuple(cols) == ("id", "session_id", *DECISION_COLUMNS)


def test_decision_missing_keys_null_unknown_key_raises(store, sid):
    store.save_decision(sid, {"ts": T0, "reason": "stale"})
    (back,) = store.last_decisions(sid)
    assert back["S"] is None and back["reason"] == "stale"
    with pytest.raises(ValueError, match="bogus"):
        store.save_decision(sid, {"ts": T0, "bogus": 1})


def test_last_decisions_newest_first(store, sid):
    for r in "abc":
        store.save_decision(sid, {"reason": r})
    assert [d["reason"] for d in store.last_decisions(sid, 2)] == ["c", "b"]
    assert len(store.last_decisions(sid)) == 3


def test_end_session_accepts_none_realized(store, sid):
    store.end_session(sid, "stopped", T0 + timedelta(hours=6), 0, None, "clean")
    row = store._db.execute("SELECT * FROM sessions WHERE id=?", (sid,)).fetchone()
    assert row["realized"] is None and row["status"] == "stopped" and row["final_shares"] == 0
    assert row["ended_at"] == aware(T0 + timedelta(hours=6)).isoformat()
