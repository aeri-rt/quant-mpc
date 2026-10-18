"""The one position/P&L ledger: option legs + hedge shares + cash. `now` is always injected.

Canonical units table (referenced by docs/decisions/004):

| Quantity            | Unit / convention                                                      | Where     |
|---------------------|------------------------------------------------------------------------|-----------|
| timestamps          | tz-aware; UTC internally, America/New_York for sessions; naive raises | clock     |
| option expiry       | 16:00:00 America/New_York of the listed date                           | clock     |
| T (pricing, theta)  | calendar years = max(0, seconds to expiry)/(365*86400)                 | clock     |
| dt (hedging, RV)    | trading years = seconds/(252*6.5*3600); from timestamp differences     | clock/vol |
| sigma (every sigma) | annualized decimal (0.15); variance = sigma^2                          | vol/gate  |
| per-share Greeks    | delta in [-1,1]; gamma 1/$; vega $ per vol-point; theta $/calendar day | bs        |
| portfolio Greeks    | per-share x qty x mult (qty signed); delta includes `shares`           | book      |
| theta in the gate   | per TRADING day = theta_day * 365/252 (same basis as Gamma S^2 s^2/252)| gate      |
| IBKR modelGreeks    | delta/gamma as per-share; vega/theta only if cfg flags are set         | book      |
|                     | (True = already per vol-pt / per day; False = /100, /365; None = BS)   |           |
| prices              | $; shares int; contracts int; multiplier 100                           | types     |
| P&L                 | realized + sum(mid - avg_cost)*qty*mult + (S - share_avg_cost)*shares  | book      |
|                     | - commissions; fills at bid/ask carry the spread in the price          |           |
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import TYPE_CHECKING

from src import bs
from src.bs import Greeks
from src.clock import T_years, aware
from src.types import Fill, IbkrGreeks, Position

if TYPE_CHECKING:
    from src.config import GreeksCfg


def _ibkr_unit(v: float | None, flag: bool | None, divisor: float, fallback: float) -> float:
    """IBKR vega/theta: verified per-pt/per-day (True) -> as is; False -> divide; None -> BS."""
    if v is None or flag is None:
        return fallback
    return v if flag else v / divisor


def _ledger(qty: int, avg: float, q: int, px: float) -> tuple[int, float, float]:
    """One average-cost step: (qty, avg, realized) after a signed fill of q @ px."""
    if qty == 0 or (qty > 0) == (q > 0):  # open or add: blend the average
        return qty + q, (avg * abs(qty) + px * abs(q)) / (abs(qty) + abs(q)), 0.0
    closed = min(abs(qty), abs(q))
    realized = (px - avg) * closed * (1 if qty > 0 else -1)
    new = qty + q
    if abs(q) > abs(qty):  # flipped: the remainder opens at px
        return new, px, realized
    return new, avg if new else 0.0, realized


@dataclass
class Leg:
    con_id: int
    right: str
    strike: float
    expiry: datetime
    qty: int
    mult: int = 100
    iv: float = 0.0
    mid: float = 0.0
    avg_cost: float = 0.0  # per share of underlying (option price)
    ibkr: IbkrGreeks | None = None

    def T(self, now: datetime) -> float:
        return T_years(self.expiry, now)

    def greeks(
        self,
        S: float,
        now: datetime,
        r: float,
        q: float,
        source: str,
        flags: GreeksCfg | None,
        max_age_s: float,
    ) -> Greeks:
        """Portfolio-scaled Greeks (per-share x qty x mult)."""
        ib = self.ibkr
        fresh = (
            source == "ibkr"
            and ib is not None
            and (aware(now) - ib.ts).total_seconds() <= max_age_s
        )
        if fresh:
            g = bs.greeks(S, self.strike, self.T(now), r, q, ib.iv, self.right)
            per_share = Greeks(
                delta=ib.delta + ib.gamma * (S - ib.S_ref),  # Gamma*dS carry-forward
                gamma=ib.gamma,
                vega=_ibkr_unit(ib.vega, flags and flags.ibkr_vega_per_volpt, 100, g.vega),
                theta=_ibkr_unit(ib.theta, flags and flags.ibkr_theta_per_day, 365, g.theta),
            )
        else:
            iv = self.iv or (ib.iv if ib is not None else 0.0)
            if not iv:
                raise ValueError("no iv")
            per_share = bs.greeks(S, self.strike, self.T(now), r, q, iv, self.right)
        return per_share.scale(self.qty * self.mult)


@dataclass
class BookDiff:
    share_drift: int
    missing_legs: list[int]
    extra_legs: list[int]
    qty_mismatch: dict[int, tuple[int, int]]

    @property
    def ok(self) -> bool:
        return not (self.missing_legs or self.extra_legs or self.qty_mismatch) and (
            self.share_drift == 0
        )


@dataclass
class Book:
    legs: list[Leg] = field(default_factory=list)
    shares: int = 0
    share_avg_cost: float = 0.0
    realized: float = 0.0
    commissions: float = 0.0
    premium_paid: float = 0.0  # set by the runner from the entry fills
    session_start_pnl: float = 0.0
    _fill_commissions: dict[str, float] = field(default_factory=dict, repr=False)

    def greeks(
        self,
        S: float,
        now: datetime,
        r: float,
        q: float,
        source: str = "bs",
        flags: GreeksCfg | None = None,
        max_age_s: float = 120.0,
    ) -> Greeks:
        g = sum(
            (leg.greeks(S, now, r, q, source, flags, max_age_s) for leg in self.legs),
            Greeks(float(self.shares), 0.0, 0.0, 0.0),
        )
        if not g.finite():
            raise ValueError(f"non-finite greeks: {g}")
        return g

    def dte(self, now: datetime) -> float:
        """Calendar days to the nearest leg expiry (float); inf with no legs."""
        if not self.legs:
            return math.inf
        return (min(leg.expiry for leg in self.legs) - aware(now)).total_seconds() / 86400

    def T(self, now: datetime) -> float:
        return min((leg.T(now) for leg in self.legs), default=0.0)

    def straddle(self) -> tuple[Leg, Leg] | None:
        if len(self.legs) != 2:
            return None
        a, b = self.legs
        if (
            a.expiry != b.expiry
            or a.strike != b.strike
            or a.qty <= 0
            or b.qty <= 0
            or {a.right, b.right} != {"C", "P"}
        ):
            return None
        return (a, b) if a.right == "C" else (b, a)

    def is_flat(self) -> bool:
        return not self.legs and self.shares == 0

    def leg(self, con_id: int) -> Leg | None:
        return next((leg for leg in self.legs if leg.con_id == con_id), None)

    def apply_fill(self, f: Fill, leg_meta: Leg | None = None) -> None:
        leg = self.leg(f.con_id)
        if leg is None and leg_meta is None:  # STK
            self.shares, self.share_avg_cost, r = _ledger(
                self.shares, self.share_avg_cost, f.qty, f.price
            )
        else:  # OPT
            if leg is None:
                leg = replace(leg_meta, con_id=f.con_id, qty=0, avg_cost=0.0)
                self.legs.append(leg)
            leg.qty, leg.avg_cost, r = _ledger(leg.qty, leg.avg_cost, f.qty, f.price)
            r *= leg.mult
            if leg.qty == 0:
                self.legs.remove(leg)
        self.realized += r
        c = f.commission or 0.0
        self.commissions += c
        self._fill_commissions[f.exec_id] = c

    def set_commission(self, exec_id: str, commission: float) -> None:
        """Late commissionReport: replace whatever was booked for this execution."""
        self.commissions += commission - self._fill_commissions.get(exec_id, 0.0)
        self._fill_commissions[exec_id] = commission

    def pnl(self, S: float) -> float:
        return (
            self.realized
            + sum((leg.mid - leg.avg_cost) * leg.qty * leg.mult for leg in self.legs)
            + (S - self.share_avg_cost) * self.shares
            - self.commissions
        )

    @classmethod
    def from_positions(cls, positions: dict[int, Position], symbol: str) -> Book:
        """Rebuild from broker positions; raises unless the option legs form one straddle."""
        b = cls()
        for p in positions.values():
            if p.symbol != symbol or p.qty == 0:
                continue
            if p.sec_type == "STK":
                b.shares, b.share_avg_cost = p.qty, p.avg_cost
            elif p.sec_type == "OPT":
                b.legs.append(
                    Leg(p.con_id, p.right, p.strike, p.expiry, p.qty, p.mult, avg_cost=p.avg_cost)
                )
        if b.legs and b.straddle() is None:
            raise ValueError(f"unpaired legs: {[leg.con_id for leg in b.legs]}")
        return b

    def diff(self, positions: dict[int, Position], symbol: str) -> BookDiff:
        """Broker minus book: share drift, legs only on one side, qty mismatches."""
        ours = {leg.con_id: leg.qty for leg in self.legs}
        theirs = {
            p.con_id: p.qty
            for p in positions.values()
            if p.symbol == symbol and p.sec_type == "OPT" and p.qty
        }
        broker_shares = sum(
            p.qty for p in positions.values() if p.symbol == symbol and p.sec_type == "STK"
        )
        return BookDiff(
            share_drift=broker_shares - self.shares,
            missing_legs=[c for c in ours if c not in theirs],
            extra_legs=[c for c in theirs if c not in ours],
            qty_mismatch={
                c: (ours[c], theirs[c]) for c in ours if c in theirs and ours[c] != theirs[c]
            },
        )
