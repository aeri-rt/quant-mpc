"""scripts/run_paper.py + scripts/check_ibkr.py: argparse, wiring, the live guard, exit codes.
No broker: nothing here connects, and Broker.connect is patched to assert it is never reached."""

import asyncio
import math
import signal
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from scripts import run_paper
from scripts.check_ibkr import unit_flag
from scripts.run_paper import build_components, exit_code, main, parse_args
from src.broker import Broker
from src.config import load_live
from src.runner import State
from src.sim_broker import SimBroker

ROOT = Path(__file__).resolve().parent.parent
RUN_PAPER = ROOT / "scripts" / "run_paper.py"
CHECK_IBKR = ROOT / "scripts" / "check_ibkr.py"


def run_script(script: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(script), *args], cwd=ROOT, capture_output=True, text=True, timeout=120
    )


@pytest.fixture
def cfg_path(tmp_path) -> str:
    """The real live.yaml with store/log paths moved under tmp_path."""
    cfg = yaml.safe_load((ROOT / "config" / "live.yaml").read_text(encoding="utf-8"))
    cfg["store"] = {"path": str(tmp_path / "trades.db")}
    cfg["log"] = {"dir": str(tmp_path / "logs"), "level": "INFO"}
    p = tmp_path / "live.yaml"
    p.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return str(p)


@pytest.fixture
def no_connect(monkeypatch):
    async def boom(self):
        raise AssertionError("Broker.connect must not be reached")

    monkeypatch.setattr(Broker, "connect", boom)


# --- subprocess level ---------------------------------------------------------------------------


def test_help_mentions_subcommands_and_switches():
    r = run_script(RUN_PAPER, "--help")
    assert r.returncode == 0, r.stderr
    for tok in ("flatten", "--execute", "--force-open"):
        assert tok in r.stdout


def test_check_ibkr_help():
    r = run_script(CHECK_IBKR, "--help")
    assert r.returncode == 0, r.stderr
    assert "--port" in r.stdout and "--client-id" in r.stdout


def test_flatten_requires_execute():
    r = run_script(RUN_PAPER, "flatten")
    assert r.returncode == 1
    assert "--execute" in r.stderr


# --- argparse -----------------------------------------------------------------------------------


def test_run_is_the_default_subcommand():
    a = parse_args([])
    assert a.cmd == "run" and not a.execute and not a.live and not a.force_open
    a = parse_args(["--execute", "--contracts", "2", "--interval", "5"])
    assert (a.cmd, a.execute, a.contracts, a.interval) == ("run", True, 2, 5.0)
    f = parse_args(["flatten", "--execute", "--client-id", "7"])
    assert (f.cmd, f.execute, f.client_id, f.contracts, f.force_open) == (
        "flatten",
        True,
        7,
        None,
        False,
    )


# --- wiring (no connection) ---------------------------------------------------------------------


def test_dry_run_wraps_sim_broker(cfg_path, no_connect):
    w = build_components(parse_args(["run", "--config", cfg_path]))
    assert isinstance(w.broker, SimBroker) and w.mode == "dry_run"
    assert w.broker._m.readonly is True  # the market Broker underneath is read-only


@pytest.mark.parametrize("monthly", [True, False])
def test_prefer_monthly_reaches_the_broker(tmp_path, cfg_path, no_connect, monthly):
    cfg = yaml.safe_load(Path(cfg_path).read_text(encoding="utf-8"))
    cfg["position"]["prefer_monthly"] = monthly
    p = tmp_path / "monthly.yaml"
    p.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    for argv in (["run", "--config", str(p)], ["run", "--execute", "--config", str(p)]):
        b = build_components(parse_args(argv)).broker
        assert getattr(b, "_m", b).prefer_monthly is monthly


def test_execute_uses_bare_broker(cfg_path, no_connect):
    w = build_components(parse_args(["run", "--execute", "--config", cfg_path]))
    assert isinstance(w.broker, Broker) and w.mode == "paper"
    assert w.broker.readonly is False and not w.broker.connected


def test_cli_overrides_nested_config(cfg_path, no_connect):
    base = load_live(cfg_path)
    argv = ["run", "--config", cfg_path, "--interval", "5", "--client-id", "99", "--port", "4002"]
    w = build_components(parse_args(argv))
    assert w.cfg.loop.interval_s == 5 and w.cfg.broker.client_id == 99 and w.cfg.broker.port == 4002
    assert w.cfg.loop.stale_quote_s == base.loop.stale_quote_s  # siblings survive the nested copy
    assert w.cfg.broker.host == base.broker.host


# --- live guard: every refusal happens before any connection ------------------------------------


@pytest.mark.parametrize(
    "extra, why",
    [
        (["--execute", "--port", "7496", "--force-open"], "--force-open"),
        (["--execute", "--port", "7497"], "--port 7496"),
        (["--execute"], "--port 7496"),
        (["--port", "7496"], "--execute"),
        (["--execute", "--port", "7496"], "not confirmed"),  # input() != CONFIRM
    ],
)
def test_live_refused_before_connect(cfg_path, no_connect, monkeypatch, capsys, extra, why):
    monkeypatch.setattr("builtins.input", lambda *_: "no")
    assert main(["run", "--live", "--config", cfg_path, *extra]) == 1
    err = capsys.readouterr().err
    assert "refused" in err and why in err


def test_live_confirm_proceeds(cfg_path, no_connect, monkeypatch, capsys):
    prompts = []

    def fake_input(prompt):
        prompts.append(prompt)
        return "CONFIRM"

    async def fake_run(w):
        assert w.mode == "live" and isinstance(w.broker, Broker) and w.broker.readonly is False
        return 0

    monkeypatch.setattr("builtins.input", fake_input)
    monkeypatch.setattr(run_paper, "run", fake_run)
    assert main(["run", "--live", "--execute", "--port", "7496", "--config", cfg_path]) == 0
    assert prompts == ["Type CONFIRM to trade LIVE: "]
    assert "mode=live" in capsys.readouterr().out


# --- exit codes ---------------------------------------------------------------------------------


def _runner(state, halted=False, reason=""):
    return SimpleNamespace(state=state, risk=SimpleNamespace(halted=halted, halt_reason=reason))


def test_exit_codes(capsys):
    assert exit_code(_runner(State.STOPPED)) == 0
    assert exit_code(_runner(State.STOPPING)) == 1
    assert exit_code(_runner(State.HALTED, True, "stale quote")) == 3
    assert exit_code(_runner(State.STOPPED, True, "reject storm")) == 3  # shutdown after a halt
    assert "reject storm" in capsys.readouterr().err


async def test_flatten_signal_cancels_into_the_runner_finally(monkeypatch, capsys):
    """SIGINT/SIGTERM during `flatten --execute` cancel the task, so Runner.flatten's finally
    (cancel our orders, end the session, disconnect) runs before the process exits."""
    handlers, ran = {}, []
    loop = asyncio.get_running_loop()
    monkeypatch.setattr(loop, "add_signal_handler", lambda sig, fn, *a: handlers.update({sig: fn}))

    async def flatten():
        try:
            handlers[signal.SIGTERM]()  # the signal arrives mid-flatten
            await asyncio.sleep(10)
        finally:
            ran.append("finally")

    w = SimpleNamespace(runner=SimpleNamespace(flatten=flatten))
    assert await asyncio.create_task(run_paper.flatten(w)) == 1  # its own task, as asyncio.run
    assert ran == ["finally"] and set(handlers) == {signal.SIGINT, signal.SIGTERM}
    assert "interrupted" in capsys.readouterr().err


# --- check_ibkr: the vega/theta unit classifier -------------------------------------------------


@pytest.mark.parametrize(
    "ratio, scale, flag",
    [
        (1.0, 100, True),
        (0.75, 100, True),
        (100.0, 100, False),
        (380.0, 365, False),
        (0.5, 100, None),
        (10.0, 100, None),
        (-1.0, 365, None),
        (math.nan, 100, None),
    ],
)
def test_unit_flag(ratio, scale, flag):
    assert unit_flag(ratio, scale) is flag
