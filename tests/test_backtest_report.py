"""backtest/report.py tables, plots and CSVs + scripts/run_backtest.py argparse.

Hand-built Results (SimpleNamespace with the sim.Result fields): 10 sessions, 3 hedgers with
different numbers; backtest.sim.run is never called, so this file is independent of the simulator.
"""

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from backtest import report
from scripts.run_backtest import load, parse_args

ROOT = Path(__file__).resolve().parent.parent
DATES = pd.bdate_range("2025-03-03", periods=10).date
GATE = np.array([0, 0, 1, 1, 1, 1, 1, 1, 0, 1], bool)
SEEDS = {"band": 1, "qp": 2, "naive": 3}


def make_result(hedger: str, risk_aversion: float, gate=GATE) -> SimpleNamespace:
    rng = np.random.default_rng(SEEDS[hedger])
    n = len(DATES)
    daily = pd.DataFrame(
        {
            "date": DATES,
            "delta": rng.normal(0, 20, n),
            "gamma": rng.uniform(10, 60, n),
            "theta": -rng.uniform(20, 40, n),
            "vega": rng.normal(0, 10, n),
            "hedge_spread": rng.uniform(1, 5, n),
            "commissions": rng.uniform(0.5, 2, n),
            "residual": rng.normal(0, 3, n),
            "trades": rng.integers(0, 8, n),
            "shares": rng.integers(0, 200, n),
            "mean_abs_delta": rng.uniform(5, 40, n),
            "sigma_f": rng.uniform(0.12, 0.2, n),
            "sigma_iv": rng.uniform(0.13, 0.19, n),
            "gate_open": gate,
        }
    )
    daily["pnl"] = (
        daily[["delta", "gamma", "theta", "vega", "residual"]].sum(axis=1)
        - daily["hedge_spread"]
        - daily["commissions"]
    )
    ts = pd.DatetimeIndex(
        [
            pd.Timestamp(f"{d} 09:30", tz="America/New_York") + pd.Timedelta(minutes=10 * k)
            for d in DATES
            for k in range(39)
        ]
    )
    m = len(ts)
    decisions = pd.DataFrame(
        {
            "ts": ts,
            "S": 600 + rng.normal(0, 2, m).cumsum(),
            "delta_net": rng.normal(0, 30, m),
            "band_h": rng.uniform(20, 50, m),
            "u_sent": rng.integers(-10, 10, m),
            "reason": "hedge",
            "pnl": rng.normal(0, 5, m).cumsum(),
        }
    )
    fills = pd.DataFrame(
        {
            "ts": ts[:4],
            "con_id": 756733,
            "qty": [10, -10, 5, -5],
            "price": 600.0,
            "commission": 0.35,
            "source": "sim",
        }
    )
    pnl, cum = daily["pnl"], daily["pnl"].cumsum()
    summary = {
        "total_pnl": pnl.sum(),
        "days": n,
        "daily_std": pnl.std(),
        "sharpe": pnl.mean() / pnl.std() * np.sqrt(252),
        "max_drawdown": float((cum.cummax() - cum).max()),
        "trades_per_day": daily["trades"].mean(),
        "hedge_cost_per_day": (daily["hedge_spread"] + daily["commissions"]).mean(),
        "hedge_friction_per_day": 1.0 + risk_aversion * 1e4,
        "hedge_error_std": (daily["delta"] + daily["residual"]).std(),
        "mean_abs_delta": daily["mean_abs_delta"].mean(),
        "turnover_frac_realised": 0.3 + SEEDS[hedger] / 10,
        "gamma_pnl": daily["gamma"].sum(),
        "theta_pnl": daily["theta"].sum(),
        "gate_open_frac": gate.mean(),
    }
    return SimpleNamespace(
        hedger=hedger,
        risk_aversion=risk_aversion,
        decisions=decisions,
        fills=fills,
        daily=daily,
        summary=summary,
    )


@pytest.fixture
def results() -> dict[str, SimpleNamespace]:
    return {
        "band": make_result("band", 1e-4),
        "qp": make_result("qp", 3e-6),
        "naive": make_result("naive", 0.0),
    }


# --- tables --------------------------------------------------------------------------------------


def test_summary_table_one_line_per_hedger_with_sharpe(results):
    txt = report.summary_table(results, gate_turnover=0.5)
    assert "DK" not in txt.splitlines()[0]
    lines = {ln.split()[0]: ln for ln in txt.splitlines()[2:]}
    assert set(lines) == {"band", "qp", "naive"}
    for name, r in results.items():
        assert f"{r.summary['sharpe']:.2f}" in lines[name]
        assert lines[name].split()[-1] == f"{100 * GATE.mean():.0f}%"  # gate% is the last column
    assert "0.40/0.50" in lines["band"]  # realised vs configured turnover, side by side


def test_attribution_total_row_sums_and_last_n(results):
    r = results["band"]
    txt = report.attribution_table(r)
    total = txt.splitlines()[-1].split()
    assert total[0] == "TOTAL"
    d = r.daily
    assert total[2] == f"{d['gamma'].sum():.0f}"
    assert total[5] == f"{d['hedge_spread'].sum():.2f}"
    assert total[8] == f"{d['pnl'].sum():.0f}"
    assert total[9] == f"{d['trades'].sum():.0f}"
    assert total[-1] == f"{100 * GATE.mean():.0f}%"
    short = report.attribution_table(r, last_n=3).splitlines()
    assert len(short) == 2 + 3 + 1  # header, rule, 3 sessions, TOTAL
    assert short[2].startswith(str(DATES[-3]))


def test_comparison_table_requires_identical_gate(results):
    txt = report.comparison_table(results)
    for tok in ("band", "qp", "naive", "gamma/theta", "residual share"):
        assert tok in txt
    rows = {ln.rsplit(maxsplit=3)[0]: ln.split()[-3:] for ln in txt.splitlines()[2:]}
    s = [r.summary for r in results.values()]
    assert rows["hedge friction/day"] == [f"{x['hedge_friction_per_day']:.2f}" for x in s]
    assert rows["std hedge-err P&L"] == [f"{x['hedge_error_std']:.0f}" for x in s]
    bad = dict(results)
    bad["qp"] = make_result("qp", 3e-6, gate=~GATE)
    with pytest.raises(ValueError, match="gate_open"):
        report.comparison_table(bad)


def test_frontier_table_sorted_with_adr_rule():
    sweep = {
        "band": [make_result("band", ra) for ra in (1e-4, 1e-3, 1e-5)],
        "qp": [make_result("qp", ra) for ra in (3e-6, 3e-5)],
    }
    txt = report.frontier_table(sweep)
    rows = [ln.split() for ln in txt.splitlines()[2:-1]]
    assert [r[0] for r in rows] == ["band"] * 3 + ["qp"] * 2
    assert [float(r[1]) for r in rows[:3]] == [1e-5, 1e-4, 1e-3]
    # axes: hedge friction (shares only) and hedging-error std, not total cost / total std
    want = sorted(sweep["band"], key=lambda r: r.risk_aversion)[0].summary
    assert rows[0][2:4] == [
        f"{want['hedge_friction_per_day']:.2f}",
        f"{want['hedge_error_std']:.0f}",
    ]
    assert txt.splitlines()[-1] == report.ADR_RULE
    assert "p<0.05 on paired daily differences" in txt


# --- files ---------------------------------------------------------------------------------------


def test_plots_write_three_pngs(results, tmp_path):
    pytest.importorskip("matplotlib")
    paths = report.plots(results, tmp_path)
    assert [p.name for p in paths] == ["equity.png", "attribution.png", "delta.png"]
    for p in paths:
        assert p.stat().st_size > 1024


def test_write_csvs_per_hedger(results, tmp_path):
    paths = report.write_csvs(results, tmp_path)
    names = {p.name for p in paths}
    assert {"daily_band.csv", "decisions_band.csv", "daily_naive.csv"} <= names
    assert len(pd.read_csv(tmp_path / "daily_qp.csv")) == 10


# --- scripts/run_backtest.py ---------------------------------------------------------------------


def test_script_help_exits_zero():
    r = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "run_backtest.py"), "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert r.returncode == 0, r.stderr
    for tok in ("--synthetic", "--ibkr", "--sweep", "--force-open", "--interval"):
        assert tok in r.stdout


def test_parse_args_and_overrides():
    a = parse_args(["--synthetic", "--hedger", "band"])
    assert a.synthetic and a.hedger == "band" and not a.sweep
    cfg = str(ROOT / "config" / "backtest.yaml")
    bt, live = load(parse_args(["--config", cfg, "--interval", "300", "--hedger", "qp"]))
    assert bt.bar_s == 300 and live.loop.interval_s == 300 and bt.hedger == "qp"
    with pytest.raises(SystemExit):
        parse_args(["--hedger", "mpc"])
    assert load(parse_args(["--config", cfg, "--ibkr"]))[0].source == "ibkr"
    assert load(parse_args(["--config", cfg, "--synthetic"]))[0].source == "synthetic"
    with pytest.raises(SystemExit):
        parse_args(["--synthetic", "--ibkr"])
