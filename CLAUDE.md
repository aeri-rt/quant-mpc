# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A Model Predictive Control (MPC) gamma-scalping system for SPY options/equities. The thesis is that delta hedging is an *optimal control problem* — solve a finite-horizon NLP each tick, apply the first action, re-solve next tick. Based on Clinet, Perreton, Reydellet (2021).

## Commands

Always activate the venv first: `source venv/bin/activate` (macOS/Linux) or `.\venv\Scripts\Activate.ps1` (Windows). All commands assume venv-active.

```bash
# Verify deps + IPOPT/do-mpc work end-to-end
python scripts/verify_setup.py

# Lint + format (line length 100, target py311)
ruff check src/ backtest/ scripts/ tests/
ruff format src/ backtest/ scripts/ tests/

# Tests — pytest is configured with asyncio_mode=auto, so async tests don't need @pytest.mark.asyncio
pytest                                          # all tests
pytest tests/test_backtest.py                   # one file
pytest tests/test_backtest.py::TestPnLEngineEquity::test_buy_sell_realized_pnl  # one test
pytest -k "kappa"                               # by name pattern
pytest --tb=long -x                             # stop on first failure with full traceback

# Paper trading (requires TWS/IB Gateway running on port 7497)
python scripts/test_ibkr_connection.py --paper  # connectivity smoke test first
python scripts/run_paper.py --dry-run           # default: log recommendations, do not trade
python scripts/run_paper.py --execute           # actually submit orders
python scripts/run_paper.py --execute --interval 30 --dte 45 --contracts 5

# Backtest (offline, no broker needed)
python scripts/fetch_backtest_data.py --synthetic        # populate data/cache/ with synthetic bars+VIX
python scripts/run_backtest.py --synthetic               # run engine with synthetic data
python scripts/run_backtest.py --config config/backtest.yaml  # with cached real data
```

`tests/test_integration.py` requires a live TWS/Gateway connection — the rest of the test suite is offline. There is no pytest marker for this distinction; check the module docstring before running.

## Architecture

### Two top-level Python packages

`src/` and `backtest/` are **peer packages**, both rooted at the project directory. Scripts and tests put the project root on `sys.path` (`sys.path.insert(0, str(Path(__file__).parent.parent))`) so both `from src.X` and `from backtest.X` resolve. There's no enclosing namespace — don't add one.

- `src/` — live & paper trading: data, models, control, strategy, execution, risk
- `backtest/` — offline simulation: replays historical bars through `GammaScalpingStrategy` + `DeltaHedgeMPC` with VIX-derived IV and a fill simulator

### Live tick pipeline (per-iteration, in `scripts/run_paper.py::run`)

```
streaming SPY quote (fallback to snapshot)
  → OptionsChainManager.update_spot
  → every 5 ticks: refresh IBKR Greeks (preferred over BS calc)
  → every 10 ticks: PositionManager.check_positions → ROLL / EXIT_PROFIT / EXIT_LOSS signals
  → every 60 ticks: reconcile internal positions vs IBKR (auto-correct equity drift)
  → GammaScalpingStrategy.update(spot)
        ├─ kappa_estimator.update → (κ, S̄, σ)
        ├─ PortfolioGreeksCalculator.calculate → (Δ, Γ, ν, Θ)
        ├─ build MPCState(S, Δ_opt, q, Γ_opt, σ, S̄)
        └─ DeltaHedgeMPC.solve → HedgeRecommendation(trade_quantity, direction, ...)
  → RiskManager.validate_trade (may adjust quantity or block)
  → OrderManager.submit_hedge_order → TradeStore (SQLite audit trail)
```

The MPC state is `[S, Δ_opt, q]`, control is `u = Δq` (shares to trade). The objective penalises delta exposure (scaled by σ), execution cost (quadratic + linear |u| via auxiliary `z_abs ≥ |u|` smooth approximation), and trade smoothness. `_apply_direction_regime()` clamps `u` bounds before each solve based on the sign of net delta vs `direction_threshold` — this is what produces three regimes (`BUY_ONLY`, `SELL_ONLY`, `BOTH`) and prevents round-trips. See `docs/design/architecture-overview.md` and `docs/decisions/002-greeks-mpc-integration.md` for full design rationale.

### Time units

**The MPC `dt` is in years, not seconds.** The YAML `mpc.time.dt` is in seconds; `MPCParams.dt` (the property) converts via `dt_seconds / (252 * 6.5 * 3600)`. The constant `TRADING_MINUTES_PER_YEAR` in `src/utils/helpers.py` is the canonical conversion factor — use it instead of hard-coding `252 * 390`.

### Configuration layering

Three YAMLs in `config/` are loaded by `src/utils/config.py::load_config()` into a single cached `AppConfig`:

- `settings.yaml` → `BrokerConfig`, `InstrumentConfig`, app mode
- `mpc_params.yaml` → `MPCParams` (and `MHE` settings, currently unused)
- `strategy_params.yaml` → `GammaScalpingParams`, `RiskParams`

`MPCParams` (the YAML-shaped dataclass) is **distinct from** `MPCConfig` (the runtime controller config in `src/control/mpc_controller.py`). The bridge is `MPCConfig.from_yaml_config(mpc_params)`. Both exist because the YAML schema is more nested than the controller wants.

The backtest has its own flat `BacktestConfig` (`backtest/config.py`) loaded from `config/backtest.yaml` — it doesn't reuse `AppConfig`.

### State machine — the only legal way to change runtime state

`src/core/state_machine.py` enforces transitions via a `VALID_TRANSITIONS` whitelist:

```
INITIALIZING → CONNECTING → POSITIONING → RUNNING ⇄ PAUSED → STOPPING → STOPPED
                    ↓             ↓           ↓
                  ERROR         ERROR       ERROR  (RESUME from ERROR re-enters INITIALIZING)
```

Always call `state_machine.trigger(StateEvent.X, details=...)` — never mutate `_state` directly. The trading loop's `while self._state_machine.can_trade:` is the kill-switch hook; `RiskManager` calls `runner.stop()` on limit breach, which fires `StateEvent.STOP`.

### Position recovery and reconciliation

On startup, `_recover_existing_positions()` reads IBKR positions before opening a new straddle — this prevents doubled exposure on restart. If options exist, it rebuilds `OptionPosition` objects from IBKR data; if only equity exists, it adopts that quantity as the current hedge. There's a 5-minute roll cooldown (`_roll_cooldown_seconds`) to prevent flapping.

Mid-session, `_reconcile_positions()` runs every 60 iterations. On equity-position drift it auto-corrects internal state to match IBKR; on option-position drift it logs a warning only.

### Validation discipline

Quotes from IBKR can arrive with NaN, Inf, zero, or inverted bid/ask (especially on delayed feeds for paper). `src/utils/validation.py` returns *new* `Quote`/`OptionQuote` instances (frozen dataclasses) with sanitised values. Greeks are clamped to physically-meaningful ranges. Validate at the IBKR-provider boundary, not deeper.

### Logging

Loguru-based, with component-tagged loggers (`get_logger("mpc")`, `get_logger("strategy")`, `get_logger("ibkr")`, etc.). Console is human-readable; file logs are JSON-serialised. Use the structured helpers (`log_trade`, `log_mpc_solve`, `log_greeks`, `log_metrics`) for events that need to be machine-parsed later — they emit typed records (`TradeLog`, `MPCSolveLog`, etc.). `Timer("op_name")` context manager records `elapsed_ms` for perf-sensitive sections.

### Custom exceptions drive recovery

The paper-trading loop catches `BrokerConnectionError`, `BrokerOrderError`, and `QuantMPCError` (`src/core/exceptions.py`) separately and fires the appropriate `StateEvent` — connection errors trigger `CONNECTION_LOST` (back to CONNECTING), recoverable errors stay in RUNNING, generic exceptions become `FATAL_ERROR`. When adding new error paths, raise the typed exception rather than a bare `Exception`.

## Conventions

- Status indicators in `README.md`'s "Current Status (V1)" table use ✅/🚧 — keep them in sync when shipping a feature.
- Configs are YAML; runtime types are dataclasses. New tunable knobs go in YAML *and* in the matching dataclass — don't hardcode in Python.
- `data/` and `logs/` are gitignored (with `.gitkeep` exceptions). Don't commit `.db`, `.sqlite`, or `.log` files.
- The codebase uses `ib_insync` for IBKR, not the raw IB API. Async functions; `await provider.connect()`.
