"""src/log.py: no import-time side effects; setup() gives one JSON-lines file."""

import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from loguru import logger


@pytest.fixture
def clean_logger():
    yield
    logger.remove()  # leave no sinks behind for other tests


def test_import_has_no_side_effects(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    sys.modules.pop("src.log", None)
    before = len(logger._core.handlers)
    importlib.import_module("src.log")
    assert list(tmp_path.iterdir()) == []
    assert len(logger._core.handlers) == before


def test_setup_writes_one_jsonl_line(tmp_path, clean_logger):
    from src.log import get_logger, setup

    setup(tmp_path)
    get_logger("x").info("hi")
    logger.remove()  # flush + close the file sink before reading
    files = list(tmp_path.glob("quant-mpc_*.jsonl"))
    assert len(files) == 1
    lines = files[0].read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])["record"]
    assert rec["extra"]["component"] == "x"
    assert rec["message"] == "hi"


def test_setup_without_dir_adds_no_file_sink(tmp_path, clean_logger, monkeypatch):
    from src.log import setup

    monkeypatch.chdir(tmp_path)
    setup(None, "WARNING")
    assert len(logger._core.handlers) == 1 and list(tmp_path.iterdir()) == []


def test_unbound_logger_prints_dash(clean_logger, capsys):
    from src.log import get_logger, setup

    setup(None)
    logger.info("plain")
    get_logger("x").info("bound")
    err = capsys.readouterr().err
    assert "| - | plain" in err and "| x | bound" in err


def test_setup_makes_redirected_console_utf8():
    # a redirected Windows console defaults to cp1252; the backtest tables print γ/Δ/σ
    r = subprocess.run(
        [sys.executable, "-c", "from src.log import setup; setup(None); print('γΔσ→')"],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        env={**os.environ, "PYTHONIOENCODING": "cp1252"},
        check=False,
    )
    assert r.returncode == 0, r.stderr
    assert r.stdout.decode("utf-8").strip() == "γΔσ→"
