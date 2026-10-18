import copy

import pytest
import yaml
from pydantic import ValidationError

from src.clock import TRADING_SECONDS_PER_YEAR, dt_years
from src.config import Backtest, Gate, Live, Position, load_backtest, load_live


@pytest.fixture(scope="module")
def live_dict():
    with open("config/live.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_live_loads(live_cfg):
    assert isinstance(live_cfg, Live)
    assert live_cfg.hedge.policy == "band"
    assert live_cfg.position.contracts is None
    assert live_cfg.greeks.ibkr_vega_per_volpt is None
    assert live_cfg.costs.option_half_spread_by_dte[0] == (5, 0.015)
    assert live_cfg.instrument.ex_dividend_dates[0].isoformat() == "2026-12-18"
    assert dt_years(live_cfg.loop.interval_s) == 60 / TRADING_SECONDS_PER_YEAR


def test_yaml_lists_every_field(live_dict):
    # Keys, not values: the owner tunes live.yaml (runbook); extra="forbid" rejects unknown keys.
    fields = {(s, k) for s, f in Live.model_fields.items() for k in f.annotation.model_fields}
    assert {(s, k) for s, v in live_dict.items() for k in v} == fields


def test_holding_sessions_default_is_trading_sessions():
    # gate amortises per TRADING day: 30 → 7 DTE is 23 calendar days ≈ 16 XNYS sessions
    p = Position()
    assert Gate().holding_sessions == round((p.entry_dte - p.roll_dte) * 252 / 365) == 16


def test_extra_key_rejected(live_dict):
    d = copy.deepcopy(live_dict)
    d["hedge"]["foo"] = 1
    with pytest.raises(ValidationError, match="foo"):
        Live.model_validate(d)


def test_missing_section_rejected(live_dict):
    d = {k: v for k, v in live_dict.items() if k != "risk"}
    with pytest.raises(ValidationError, match="risk"):
        Live.model_validate(d)


@pytest.mark.parametrize(
    "section, key, value, msg",
    [
        ("position", "roll_dte", 30, "roll_dte < position.entry_dte"),
        ("position", "hard_exit_dte", 7, "hard_exit_dte < position.roll_dte"),
        ("hedge", "inner_fraction", 1.5, "inner_fraction"),
        ("loop", "stale_quote_s", 600, "stale_quote_s < loop.stale_halt_s"),
        ("gate", "exit_confirm_sessions", 0, "exit_confirm_sessions"),
        ("gate", "eval_time", "10:00:00", "pattern"),
        ("gate", "eval_time", "25:00", "pattern"),
        ("hedge", "policy", "mpc", "policy"),
    ],
)
def test_cross_checks(live_dict, section, key, value, msg):
    d = copy.deepcopy(live_dict)
    d[section][key] = value
    with pytest.raises(ValidationError, match=msg):
        Live.model_validate(d)


def test_backtest_loads():
    bt, live = load_backtest("config/backtest.yaml")
    assert isinstance(bt, Backtest) and isinstance(live, Live)
    assert bt.start < bt.end
    assert bt.bar_s == live.loop.interval_s
    assert bt.sweep_risk_aversion.band[2] == live.hedge.band_risk_aversion


def test_backtest_absolute_live_path(tmp_path):
    cfg = tmp_path / "cfg" / "bt.yaml"
    cfg.parent.mkdir()
    live_abs = (tmp_path.parent / "x").resolve()  # any absolute path is used verbatim
    cfg.write_text(f"live_config: {live_abs}\nstart: 2025-01-02\nend: 2025-01-03\n")
    with pytest.raises(FileNotFoundError):
        load_backtest(cfg)
    with pytest.raises(FileNotFoundError):
        load_live(tmp_path / "missing.yaml")


def test_frozen(live_cfg):
    with pytest.raises(ValidationError):
        live_cfg.hedge.policy = "qp"
    with pytest.raises(ValidationError):
        live_cfg.loop = live_cfg.loop
