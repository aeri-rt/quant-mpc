# Repository Guidelines

## Project Structure & Module Organization

quant-mpc is a Python 3.12/3.13 gated long-gamma SPY straddle, delta-hedged by a Whalley–Wilmott band, with a closed-form receding-horizon QP as the research hedger (`docs/decisions/004-v2-thesis-and-hedger.md`).

- `src/`: 16 flat modules — `clock`, `types`, `config`, `log`, `costs`, `bs`, `book`, `vol`, `gate`, `hedge`, `risk`, `validation`, `store`, `sim_broker`, `broker`, `runner`. No subpackages.
- `backtest/`: `data`, `iv`, `sim`, `report` — drives the same `Runner.tick` per bar through `SimBroker` + `SimClock`; a peer package to `src/`.
- `scripts/`: `run_paper.py` (run | flatten), `run_backtest.py`, `fetch_data.py` (--cboe | --ibkr), `check_ibkr.py` (read-only paper checklist), `calibrate_costs.py`, setup scripts.
- `tests/`: offline pytest suite (`conftest.py`, `fake_ib.py`, `fixtures/`). `config/`: `live.yaml`, `backtest.yaml`.
- `docs/`: decisions, design (one spec kept for reference), guides, runbooks, research, templates. `data/` and `logs/`: ignored runtime artifacts.

## Build, Test, and Development Commands

Run from the repository root with the virtual environment activated (`source venv/bin/activate`; Windows `.\venv\Scripts\Activate.ps1`).

- Install: `pip install $(python -c "import tomllib;d=tomllib.load(open('pyproject.toml','rb'))['project'];print(' '.join(d['dependencies']+d['optional-dependencies']['dev']))")` or `pip install -r requirements.lock`.
- `pytest -q`: the whole suite, offline, ≈20 s.
- `ruff check src backtest scripts tests` / `ruff format src backtest scripts tests`.
- `python scripts/run_backtest.py --synthetic --hedger all --force-open`: offline end-to-end run (`--ibkr` replays the bars cached by `fetch_data.py --ibkr`).
- `python scripts/check_ibkr.py`: connectivity, clock offset, quote freshness, Greek-unit ratios, whatIf; never orders.
- `python scripts/run_paper.py`: dry-run through SimBroker (default); `--execute` places paper orders; `flatten --execute` liquidates.
- `python scripts/calibrate_costs.py`: booked commissions vs the cost model after an execute session.

## Coding Style & Naming Conventions

Four-space indentation, type annotations, `snake_case` functions/modules, `PascalCase` classes, `UPPER_SNAKE_CASE` constants. Ruff targets Python 3.12 with a 100-character line length, import sorting, and the `DTZ` (no naive datetimes) and `ASYNC` rule sets. Configuration is pydantic v2 with `extra="forbid"`: a new knob goes in the YAML and the model. Inject `now`; the only wall-clock read is `Clock.now()` in `src/clock.py`. Marketable limit orders only, positions by conId, no simulate-on-failure. Typed exceptions are `BrokerError`, `BrokerDisconnected` and `IllegalTransition`; component loggers come from `src.log.get_logger`. Mark deliberate ceilings with a `# ponytail:` comment naming the upgrade path. Shortest correct code, stdlib first, no speculative abstractions.

## Testing Guidelines

pytest with `asyncio_mode = "auto"` and `pythonpath = ["."]`. Name files `test_*.py`, functions `test_*`. Everything runs offline: `tests/fake_ib.py` and `conftest.py::FakeMarket` replace TWS; Runner-driven sim tests use 5-min bars. Focused runs: `pytest tests/test_runner.py -q`, `pytest -k flatten -q`. Add a regression test for changed behaviour (order states, risk limits, state transitions, numeric boundaries); hypothesis is available for identities. Never run anything against a broker from the test suite.

## Commit & Pull Request Guidelines

Short imperative subjects (`Add platform-specific setup scripts`). Keep commits focused. PRs explain the problem, the behaviour change, the validation commands and results. Update `config/*.yaml` comments, the README status table and the relevant ADR/runbook when behaviour changes.

## Configuration & Generated Files

Keep credentials, market-data caches, databases and logs out of commits (`data/`, `logs/` are ignored except `.gitkeep`). Document substantial architectural changes with `docs/templates/DECISION_RECORD.md` as a numbered file in `docs/decisions/`.
