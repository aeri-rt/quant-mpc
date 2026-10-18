"""Alpha gate: expected edge of a straddle in $ per TRADING day, and the once-per-session
OPEN/CLOSED decision with entry/exit hysteresis and a re-entry cooldown.

Pure: no wall-clock reads, no vol estimation, no sizing. Sign-agnostic: Γ and θ enter signed,
so a short straddle is gated by the same code (asserted in tests, not wired).
"""

import math
from dataclasses import dataclass
from datetime import date, datetime, time

from src.bs import Greeks
from src.clock import dt_years, next_session_date, ny, session_date
from src.config import Costs
from src.config import Gate as GateCfg
from src.costs import option_cost, per_share_rate


@dataclass(frozen=True)
class Edge:
    edge_day: float  # gamma_day + theta_tday - hedge_cost_day - option_cost_day
    gamma_day: float  # ½ Γ S² σ_f² / 252
    theta_tday: float  # θ ($/calendar day) · 365/252
    hedge_cost_day: float
    option_cost_day: float  # 0 when sunk (exit test)
    leland_a: float  # diagnostic only
    sigma_f: float
    sigma_iv: float
    shares_day: float  # expected |Δ change| per trading day
    crossings_day: float  # shares_day / band_trade_shares


NAN_EDGE = Edge(-math.inf, *[math.nan] * 9)  # any non-finite input; caller logs reason "nan"


def edge_per_day(
    g: Greeks,
    S: float,
    sigma_f: float,
    sigma_iv: float,
    c: Costs,
    band_trade_shares: float,
    n_contracts: int,
    dte: float,
    holding_sessions: int,
    turnover_frac: float,
    interval_s: float,
    sunk: bool,
) -> Edge:
    """$ per trading day of holding (sunk=True) or entering (sunk=False) the book `g`.

    `band_trade_shares` is the hedger's (1 − inner_fraction)·H; `holding_sessions` (XNYS sessions
    from entry to roll) amortises the option round trip (2 × one order) per trading day.
    """
    if not (g.finite() and all(map(math.isfinite, (S, sigma_f, sigma_iv, band_trade_shares, dte)))):
        return NAN_EDGE
    gamma_day = 0.5 * g.gamma * S**2 * sigma_f**2 / 252
    theta_tday = g.theta * 365 / 252
    shares_day = abs(g.gamma) * S * sigma_f * math.sqrt(2 / (math.pi * 252))
    c_bar = 0.5 * (per_share_rate(S, 1, c) + per_share_rate(S, -1, c))
    crossings_day = shares_day / max(band_trade_shares, 1.0)
    hedge_cost_day = c_bar * shares_day * turnover_frac + c.min_commission * crossings_day
    option_cost_day = (
        0.0 if sunk else 2 * option_cost(n_contracts, dte, c) / max(holding_sessions, 1)
    )
    denom = sigma_f * math.sqrt(dt_years(interval_s))
    leland_a = math.sqrt(2 / math.pi) * (c_bar / S) / denom if denom > 0 else math.inf
    return Edge(
        gamma_day + theta_tday - hedge_cost_day - option_cost_day,
        gamma_day,
        theta_tday,
        hedge_cost_day,
        option_cost_day,
        leland_a,
        sigma_f,
        sigma_iv,
        shares_day,
        crossings_day,
    )


class Gate:
    """OPEN/CLOSED with hysteresis. The runner calls `evaluate` once per session (when `due()`,
    at/after `eval_time` NY) and at roll time for the new candidate.

    The runner may set `reason` directly (no_chain, no_iv, no_vol, spread_wide, ...) and calls
    `force_closed` after an executed exit or a refused roll.
    """

    def __init__(self, cfg: GateCfg):
        self.cfg = cfg
        self._eval_time = time.fromisoformat(cfg.eval_time)
        self.state = "CLOSED"
        self.reason = "init"
        self.last_eval_session: date | None = None
        self.exit_streak = 0
        self.cooldown_until: date | None = None
        self.last_edge: Edge | None = None

    def due(self, now: datetime) -> bool:
        return ny(now).time() >= self._eval_time and self.last_eval_session != session_date(now)

    def evaluate(self, e: Edge, now: datetime, held: bool) -> str:
        """Held: `e` is the book's edge with sunk=True. Flat: `e` is the candidate's.
        Caller guards with due() or roll time."""
        today, cfg = session_date(now), self.cfg
        self.last_eval_session, self.last_edge = today, e
        if held:
            self.exit_streak = self.exit_streak + 1 if e.edge_day < cfg.exit_edge_usd_per_day else 0
            if self.exit_streak >= cfg.exit_confirm_sessions:
                self.state, self.reason = "CLOSED", "exit_confirmed"
                self.close_cooldown(now)
            else:
                self.state, self.reason = "OPEN", "holding"
        elif self.cooldown_until and today < self.cooldown_until:
            self.state, self.reason = "CLOSED", "cooldown"
        elif e.sigma_f - e.sigma_iv < cfg.min_vol_margin:  # NaN margin falls through
            self.state, self.reason = "CLOSED", "vol_margin"
        elif e.edge_day >= cfg.min_edge_usd_per_day:
            self.state, self.reason, self.exit_streak = "OPEN", "gate_open", 0
        else:
            self.state, self.reason = "CLOSED", "edge_below_min"
        return self.state

    def force_closed(self, now: datetime, reason: str) -> None:
        self.state, self.reason, self.exit_streak = "CLOSED", reason, 0
        self.close_cooldown(now)

    def close_cooldown(self, now: datetime) -> None:
        d = session_date(now)
        for _ in range(self.cfg.reentry_cooldown_sessions):
            d = next_session_date(d)
        self.cooldown_until = d
