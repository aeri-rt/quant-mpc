"""SQLite audit trail: sessions, orders (STK and BAG alike), fills by execId, decision rows.

Autocommit connection, WAL for file-backed stores, foreign keys on. Every timestamp is passed in
by the caller and stored as ISO-8601 UTC text; session ids are uuid4 hex.
"""

import sqlite3
import uuid
from datetime import datetime
from pathlib import Path

from src.clock import aware
from src.log import get_logger
from src.types import Fill, Order

log = get_logger("store")

_DECISION_TYPES = {
    "ts": "TEXT",
    "mode": "TEXT",
    "md_type": "INTEGER",
    "quote_age_s": "REAL",
    "clock_offset_s": "REAL",
    "S": "REAL",
    "bid": "REAL",
    "ask": "REAL",
    "iv_atm": "REAL",
    "sigma_f": "REAL",
    "delta_opt": "REAL",
    "gamma": "REAL",
    "theta_day": "REAL",
    "vega": "REAL",
    "shares": "INTEGER",
    "delta_net": "REAL",
    "band_h": "REAL",
    "qp_h": "REAL",
    "band_u": "INTEGER",
    "qp_u": "INTEGER",
    "naive_u": "INTEGER",
    "u_sent": "INTEGER",
    "policy": "TEXT",
    "risk_result": "TEXT",
    "gate_state": "TEXT",
    "edge_day": "REAL",
    "leland_a": "REAL",
    "campaign": "TEXT",
    "reason": "TEXT",
    "nlv": "REAL",
    "daily_pnl": "REAL",
    "book_pnl": "REAL",
    "force_open": "INTEGER",
    "fill_source": "TEXT",
}
DECISION_COLUMNS = tuple(_DECISION_TYPES)  # the runner builds its row from these keys

_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS sessions (
  id TEXT PRIMARY KEY, started_at TEXT NOT NULL, ended_at TEXT, mode TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'running', cfg_json TEXT, final_shares INTEGER, realized REAL,
  notes TEXT);
CREATE TABLE IF NOT EXISTS orders (
  session_id TEXT NOT NULL REFERENCES sessions(id), id TEXT NOT NULL, con_id INTEGER,
  sec_type TEXT, action TEXT, qty INTEGER, limit_price REAL, reason TEXT, state TEXT,
  filled INTEGER, avg_price REAL, broker_order_id INTEGER, perm_id INTEGER, error TEXT,
  created_at TEXT, updated_at TEXT, PRIMARY KEY (session_id, id));
CREATE TABLE IF NOT EXISTS fills (
  exec_id TEXT PRIMARY KEY, session_id TEXT NOT NULL REFERENCES sessions(id), order_id TEXT,
  con_id INTEGER, qty INTEGER, price REAL, commission REAL, source TEXT, ts TEXT);
CREATE TABLE IF NOT EXISTS decisions (
  id INTEGER PRIMARY KEY, session_id TEXT NOT NULL REFERENCES sessions(id),
  {", ".join(f"{c} {t}" for c, t in _DECISION_TYPES.items())});
"""
_INSERT_DECISION = (
    f"INSERT INTO decisions (session_id, {', '.join(DECISION_COLUMNS)}) "
    f"VALUES ({', '.join('?' * (len(DECISION_COLUMNS) + 1))})"
)


def _iso(t: datetime | None) -> str | None:
    return aware(t).isoformat() if t is not None else None


def _dt(s: str | None) -> datetime | None:
    return datetime.fromisoformat(s) if s else None


class Store:
    def __init__(self, path: str | Path):
        path = str(path)
        memory = path == ":memory:"
        if not memory:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path, isolation_level=None, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        if not memory:
            self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA foreign_keys=ON")
        self._db.execute("PRAGMA busy_timeout=5000")
        self._db.executescript(_SCHEMA)

    def close(self) -> None:
        self._db.close()

    def start_session(self, mode: str, cfg_json: str, started_at: datetime) -> str:
        sid = uuid.uuid4().hex
        self._db.execute(
            "INSERT INTO sessions (id, started_at, mode, cfg_json) VALUES (?, ?, ?, ?)",
            (sid, _iso(started_at), mode, cfg_json),
        )
        return sid

    def end_session(
        self,
        session_id: str,
        status: str,
        ended_at: datetime,
        final_shares: int,
        realized: float | None,
        notes: str = "",
    ) -> None:
        self._db.execute(
            "UPDATE sessions SET status=?, ended_at=?, final_shares=?, realized=?, notes=? "
            "WHERE id=?",
            (status, _iso(ended_at), final_shares, realized, notes, session_id),
        )

    def save_order(self, session_id: str, o: Order) -> None:
        """Idempotent: Broker._place saves at placement and the Runner saves the same Order again."""
        self._db.execute(
            "INSERT OR REPLACE INTO orders (session_id, id, con_id, sec_type, action, qty,"
            " limit_price, reason,"
            " state, filled, avg_price, broker_order_id, perm_id, error, created_at, updated_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                session_id,
                o.id,
                o.con_id,
                o.sec_type,
                o.action,
                o.qty,
                o.limit,
                o.reason,
                o.state.value,
                o.filled,
                o.avg_price,
                o.broker_order_id,
                o.perm_id,
                o.error,
                _iso(o.created),
                _iso(o.updated),
            ),
        )

    def update_order(self, session_id: str, o: Order) -> None:
        """State change on a saved order; created_at is never touched."""
        n = self._db.execute(
            "UPDATE orders SET state=?, filled=?, avg_price=?, broker_order_id=?, perm_id=?,"
            " error=?, updated_at=? WHERE session_id=? AND id=?",
            (
                o.state.value,
                o.filled,
                o.avg_price,
                o.broker_order_id,
                o.perm_id,
                o.error,
                _iso(o.updated),
                session_id,
                o.id,
            ),
        ).rowcount
        if n == 0:
            log.warning("update_order: order {id} not in session {sid}", id=o.id, sid=session_id)

    def save_fill(self, session_id: str, f: Fill) -> None:
        self._db.execute(
            "INSERT OR IGNORE INTO fills (exec_id, session_id, order_id, con_id, qty, price,"
            " commission, source, ts) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                f.exec_id,
                session_id,
                f.order_id,
                f.con_id,
                f.qty,
                f.price,
                f.commission,
                f.source,
                _iso(f.ts),
            ),
        )

    def set_commission(self, exec_id: str, commission: float) -> int:
        """Patch from commissionReport (arrives after execDetails); rows updated (0 = unknown)."""
        return self._db.execute(
            "UPDATE fills SET commission=? WHERE exec_id=?", (commission, exec_id)
        ).rowcount

    def save_decision(self, session_id: str, row: dict) -> None:
        unknown = row.keys() - _DECISION_TYPES.keys()
        if unknown:
            raise ValueError(f"unknown decision columns: {sorted(unknown)}")
        vals = [row.get(c) for c in DECISION_COLUMNS]
        self._db.execute(
            _INSERT_DECISION,
            [session_id, *(_iso(v) if isinstance(v, datetime) else v for v in vals)],
        )

    def last_decisions(self, session_id: str, n: int = 20) -> list[dict]:
        rows = self._db.execute(
            "SELECT * FROM decisions WHERE session_id=? ORDER BY id DESC LIMIT ?", (session_id, n)
        )
        return [dict(r) for r in rows]

    def campaign_fills(self, leg_con_ids: list[int], stock_con_id: int) -> list[Fill]:
        """Broker fills of the campaign holding these legs, oldest first, across sessions: the
        leg fills since the legs were last all flat, plus the stock fills from that opening on."""
        held = dict.fromkeys(leg_con_ids, 0)
        marks = ",".join("?" * (len(held) + 1))
        rows = self._db.execute(
            f"SELECT * FROM fills WHERE source='broker' AND con_id IN ({marks}) ORDER BY ts, rowid",
            (*held, stock_con_id),
        ).fetchall()
        start = len(rows)
        for i, r in enumerate(rows):
            if r["con_id"] in held:
                if not any(held.values()):  # an opening fill: an earlier campaign ended before it
                    start = i
                held[r["con_id"]] += r["qty"]
        return [
            Fill(
                r["exec_id"],
                r["order_id"],
                r["con_id"],
                r["qty"],
                r["price"],
                _dt(r["ts"]),
                r["commission"],
                r["source"],
            )
            for r in rows[start:]
        ]
