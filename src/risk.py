"""Risk limits and the kill switch: flags, counters and pure checks.

Single-threaded asyncio: no locks, no callbacks, no threads. The runner reads `halted` and
calls `runner.halt()` itself. Every limit is a fraction of NetLiquidation (ADR-004 §4).
"""

from collections import deque
from datetime import datetime, timedelta
from math import floor

from src.clock import aware, is_open, session_open
from src.config import Loop
from src.config import Risk as RiskCfg
from src.log import get_logger
from src.types import Account

log = get_logger("risk")
PDT_NLV = 25_000  # FINRA pattern-day-trader equity threshold ($)
REJECT_WINDOW = timedelta(seconds=600)


class Risk:
    def __init__(self, cfg: RiskCfg, loop: Loop, execute: bool):
        self.cfg, self.loop, self.execute = cfg, loop, execute
        self.halted = False
        self.halt_reason = ""
        self._loss_streak = 0
        self._rejects: deque[datetime] = deque()
        self._stale_since: datetime | None = None

    def halt(self, reason: str) -> None:
        if self.halted:
            return
        self.halted, self.halt_reason = True, reason
        log.critical("HALT {reason}", reason=reason)

    def size_contracts(
        self,
        premium_per_straddle: float,
        acct: Account | None,
        requested: int | None,
        max_contracts: int,
    ) -> tuple[int, str]:
        """Straddles affordable under max_premium_frac_nlv; premium is $/share (call mid + put mid)."""
        if acct is None:
            return 0, "no_account"
        if acct.nlv < self.cfg.min_nlv:
            return 0, "min_nlv"
        if premium_per_straddle <= 0:
            return 0, "no_premium"
        n = floor(self.cfg.max_premium_frac_nlv * acct.nlv / (premium_per_straddle * 100))
        n = min(n, max_contracts)
        if requested:
            n = min(n, requested)
        return (n, "ok") if n >= 1 else (0, "premium_frac")

    def check_hedge(
        self,
        u: int,
        delta_net: float,
        shares: int,
        contracts: int,
        acct: Account | None,
        quote_ok: bool,
    ) -> tuple[int, str]:
        """First failing check wins. Never blocks a trade that reduces |delta| or |shares|."""
        if self.halted:
            return 0, "halted"
        if u == 0:
            return 0, "none"
        if self.execute and not quote_ok:
            return 0, "delayed"
        if self.execute and acct is None:
            return 0, "no_account"
        if acct is not None:
            if acct.nlv < self.cfg.min_nlv:
                return 0, "min_nlv"
            if acct.day_trades_remaining == 0 and acct.nlv < PDT_NLV:  # IBKR: -1 = unlimited
                return 0, "pdt"
        if abs(delta_net + u) > abs(delta_net) + 1e-9:
            return 0, "increases_exposure"
        after = abs(shares + u)
        if after > abs(shares) and after > contracts * 100 + self.cfg.max_net_delta_shares:
            return 0, "book_bound"
        return u, "ok"

    def check_straddle(
        self,
        n: int,
        premium_per_straddle: float,
        acct: Account | None,
        init_margin_change: float | None,
        max_contracts: int,
    ) -> tuple[bool, str]:
        if self.halted:
            return False, "halted"
        if acct is None:
            return False, "no_account"
        if n < 1 or n > max_contracts:
            return False, "contracts"
        if acct.nlv < self.cfg.min_nlv:
            return False, "min_nlv"
        if premium_per_straddle * 100 * n > self.cfg.max_premium_frac_nlv * acct.nlv:
            return False, "premium_frac"
        if (
            init_margin_change is not None
            and init_margin_change > self.cfg.max_margin_frac_excess * acct.excess_liquidity
        ):
            return False, "margin"
        return True, "ok"

    def on_pnl(
        self,
        broker_daily: float | None,
        book_daily: float | None,
        acct: Account | None,
        now: datetime,
    ) -> None:
        """Daily-loss kill switch: min(broker, book) below -frac*NLV for loss_persist_ticks in a row,
        ignored inside open_grace_s after the open (pre-open marks)."""
        if self.halted or acct is None:
            return
        losses = [x for x in (broker_daily, book_daily) if x is not None]
        if not losses:
            return
        loss, limit = min(losses), self.cfg.max_daily_loss_frac_nlv * acct.nlv
        self._loss_streak = self._loss_streak + 1 if loss < -limit else 0
        if self._loss_streak < self.cfg.loss_persist_ticks:
            return
        opened = session_open(now)
        if opened is None or (aware(now) - opened).total_seconds() >= self.cfg.open_grace_s:
            self.halt(f"daily loss {loss:.0f} < -{limit:.0f}")

    def record_reject(self, now: datetime) -> None:
        now = aware(now)
        self._rejects.append(now)
        while now - self._rejects[0] > REJECT_WINDOW:
            self._rejects.popleft()
        if len(self._rejects) >= self.cfg.max_rejects_per_10min:
            self.halt("reject storm")

    def check_stale(self, quote_age_s: float | None, now: datetime) -> str:
        """'fresh' | 'stale' (age > stale_quote_s) | 'halt' (stale for > stale_halt_s while the
        market is open; no SPY ticks after hours or on a holiday is not a fault)."""
        if quote_age_s is not None and quote_age_s <= self.loop.stale_quote_s:
            self._stale_since = None
            return "fresh"
        now = aware(now)
        if not is_open(now):
            self._stale_since = None
            return "stale"
        if self._stale_since is None:
            self._stale_since = now
        if (now - self._stale_since).total_seconds() > self.loop.stale_halt_s:
            self.halt("stale quote")
            return "halt"
        return "stale"
