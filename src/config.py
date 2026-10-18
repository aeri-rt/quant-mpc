"""Configuration: pydantic v2 models (frozen, unknown keys rejected) loaded from config/*.yaml.

Field defaults are the documented starting values; config/live.yaml must list every field (tests
check keys, not values) and is where the owner tunes them. No env-var layer.
"""

from datetime import date
from pathlib import Path
from typing import Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Broker(_Base):
    host: str = "127.0.0.1"
    port: int = 7497
    client_id: int = 10
    market_data_type: Literal[1, 3] = 3
    timeout_s: float = 10
    account: str = ""


class Instrument(_Base):
    symbol: str = "SPY"
    risk_free_rate: float = 0.04
    dividend_yield: float = 0.0099
    ex_dividend_dates: list[date] = []


class Position(_Base):
    contracts: int | None = None
    max_contracts: int = 10
    entry_dte: int = 30
    prefer_monthly: bool = True
    roll_dte: int = 7
    hard_exit_dte: int = 1
    max_loss_frac_premium: float = 0.5


class Loop(_Base):
    interval_s: float = 60
    reconcile_s: float = 300
    close_window_s: float = 300
    stale_quote_s: float = 15
    stale_halt_s: float = 600
    fill_timeout_s: float = 20
    straddle_timeout_s: float = 30
    no_entry_last_s: float = 1800
    warmup_sessions: int = 30
    warmup_max_s: float = 120


class GreeksCfg(_Base):
    source: Literal["bs", "ibkr"] = "bs"
    max_age_s: float = 120
    ibkr_vega_per_volpt: bool | None = None
    ibkr_theta_per_day: bool | None = None


class Vol(_Base):
    ewma_lambda: float = 0.94
    har_min_sessions: int = 60
    profile_sessions: int = 60
    overnight_frac: float = 0.33


class Gate(_Base):
    eval_time: str = Field("10:00", pattern=r"^([01]\d|2[0-3]):[0-5]\d$")  # NY, HH:MM
    min_edge_usd_per_day: float = 10.0
    min_vol_margin: float = 0.01
    exit_edge_usd_per_day: float = -25.0
    exit_confirm_sessions: int = 2
    reentry_cooldown_sessions: int = 2
    holding_sessions: int = 16  # XNYS sessions from entry (30 DTE) to roll (7 DTE)
    turnover_frac: float = 2.0


class Hedge(_Base):
    policy: Literal["band", "qp", "naive"] = "band"
    inner_fraction: float = 0.5
    band_min_shares: float = 5
    band_max_shares: float = 300
    max_trade_shares: int = 300
    band_risk_aversion: float = 1e-4  # γ, 1/$ (band)
    qp_risk_aversion: float = 3e-6  # λ, 1/$ (qp) — a separate axis from γ


class Costs(_Base):
    commission_per_share: float = 0.0035
    min_commission: float = 0.35
    half_spread: float = 0.005
    sec_fee_rate: float = 2.06e-5
    finra_taf_per_share: float = 0.0
    finra_taf_cap: float = 9.79
    option_commission: float = 0.65
    option_min_commission: float = 1.0
    option_fees_per_contract: float = 0.05
    option_half_spread_by_dte: list[tuple[float, float]] = [
        (5, 0.015),
        (14, 0.04),
        (45, 0.06),
        (90, 0.07),
    ]


class Risk(_Base):
    max_premium_frac_nlv: float = 0.15
    max_margin_frac_excess: float = 0.5
    max_daily_loss_frac_nlv: float = 0.02
    loss_persist_ticks: int = 3
    open_grace_s: float = 300
    max_net_delta_shares: int = 150
    max_share_drift: int = 50
    max_rejects_per_10min: int = 3
    min_nlv: float = 25_000


class StoreCfg(_Base):
    path: str = "data/trades.db"


class Log(_Base):
    dir: str = "logs"
    level: str = "INFO"


class Live(_Base):
    broker: Broker
    instrument: Instrument
    position: Position
    loop: Loop
    greeks: GreeksCfg
    vol: Vol
    gate: Gate
    hedge: Hedge
    costs: Costs
    risk: Risk
    store: StoreCfg
    log: Log

    @model_validator(mode="after")
    def _cross_checks(self) -> Self:
        p, h, lp, g = self.position, self.hedge, self.loop, self.gate
        rules = {
            "position.roll_dte < position.entry_dte": p.roll_dte < p.entry_dte,
            "position.hard_exit_dte < position.roll_dte": p.hard_exit_dte < p.roll_dte,
            "0 <= hedge.inner_fraction <= 1": 0 <= h.inner_fraction <= 1,
            "loop.stale_quote_s < loop.stale_halt_s": lp.stale_quote_s < lp.stale_halt_s,
            "gate.exit_confirm_sessions >= 1": g.exit_confirm_sessions >= 1,
        }
        if broken := [r for r, ok in rules.items() if not ok]:
            raise ValueError("violated: " + "; ".join(broken))
        return self


class Synthetic(_Base):
    sigma: float = 0.16
    seed: int = 42
    S0: float = 600.0
    iv: float = 0.15


class Sweep(_Base):
    band: list[float] = [1e-3, 3e-4, 1e-4, 3e-5, 1e-5]
    qp: list[float] = [3e-5, 1e-5, 3e-6, 1e-6, 3e-7]


class Backtest(_Base):
    live_config: str = "config/live.yaml"
    start: date
    end: date
    bar_s: int = 60
    source: Literal["synthetic", "ibkr"] = "synthetic"
    hedger: Literal["band", "qp", "naive", "all"] = "all"
    cache_dir: str = "data/cache"
    out_dir: str = "data/backtest_results"
    cache_version: int = 2
    skew_b: float = 1.0
    initial_nlv: float = 100_000
    synthetic: Synthetic = Synthetic()
    sweep_risk_aversion: Sweep = Sweep()


def _read(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def load_live(path: str | Path = "config/live.yaml") -> Live:
    return Live.model_validate(_read(path))


def load_backtest(path: str | Path = "config/backtest.yaml") -> tuple[Backtest, Live]:
    """Load the backtest config and the live config it names (relative paths resolve from the repo root)."""
    bt = Backtest.model_validate(_read(path))
    live = Path(bt.live_config)
    if not live.is_absolute():
        live = Path(path).resolve().parent.parent / live
    return bt, load_live(live)
