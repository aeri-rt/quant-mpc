# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A gated long-gamma SPY straddle, delta-hedged by a cost-aware Whalley–Wilmott band. The thesis (ADR-004, `docs/decisions/004-v2-thesis-and-hedger.md`) is that the edge is the *gate* — long gamma only when the realized-variance forecast beats the candidate's implied variance by enough to pay theta, hedging friction and the option round-trip — and the hedger only decides how cheaply to carry it. A receding-horizon QP ships as its closed-form first action (research path, shadow-logged every tick). The hedger has no casadi/do-mpc/scipy optimizer or QP solver and none will be added (the N-step program exists only as a scipy SLSQP test oracle); scipy in production is limited to `norm` and the `brentq` implied-vol root in `src/bs.py`.

## Commands

Always activate the venv first: `source venv/bin/activate` (macOS/Linux) or `.\venv\Scripts\Activate.ps1` (Windows).

```bash
# Lint + format (line length 100, target py312; ruff DTZ and ASYNC rules are on)
ruff check src backtest scripts tests
ruff format src backtest scripts tests

# Tests — all offline (no TWS, no network), ≈20 s; asyncio_mode=auto, pythonpath=["."]
pytest -q
pytest tests/test_hedge.py -q
pytest tests/test_hedge.py::test_qp_closed_form_equals_program -q
pytest -k "flatten" -q
pytest --tb=long -x

# Backtest (offline; synthetic bars are generated from config/backtest.yaml's seed)
python scripts/run_backtest.py --synthetic --hedger all --force-open      # ≈180 s at 1-min bars
python scripts/run_backtest.py --synthetic --hedger band --interval 300    # ≈13 s (≈32 s with --hedger all --force-open)
python scripts/run_backtest.py --ibkr --hedger all --sweep --start 2024-10-07 --end 2025-03-31   # real bars cached by fetch_data --ibkr; the first 60 sessions are vol warm-up, P&L from 2025-01-02
python scripts/fetch_data.py --cboe                                        # VIX9D/VIX/VIX3M closes → data/cache
python scripts/fetch_data.py --ibkr --start 2024-10-07 --end 2025-03-31    # 1-min MIDPOINT bars via TWS, cached per session

# Paper (TWS/Gateway on 7497, Master Client ID = broker.client_id)
python scripts/check_ibkr.py                      # read-only checklist; prints the greeks.ibkr_* lines for live.yaml
python scripts/run_paper.py                       # dry-run (default): orders go to SimBroker
python scripts/run_paper.py --execute --contracts 1 --force-open
python scripts/run_paper.py flatten --execute     # liquidate; the only liquidation path after a HALT
python scripts/calibrate_costs.py                 # booked commissions vs costs model after an execute session
```

`--live` needs `--execute`, `--port 7496` on the command line and a typed `CONFIRM`; `--force-open` is refused with `--live`. Exit codes: 0 STOPPED, 3 HALTED (reason printed), 1 otherwise.

## Architecture

### Flat peer packages

`src/` (16 modules) and `backtest/` (4) are peer packages rooted at the repo; `scripts/` and `tests/` are packages too and `[tool.pytest.ini_options] pythonpath = ["."]` makes `from src.x`, `from backtest.x`, `from scripts.run_paper` and `from tests.conftest` resolve. There is no enclosing namespace and no subpackages — do not add either. The README's module table is the map.

### Tick pipeline (`Runner.tick`)

```
quote + drain fills → staleness (DEGRADED / HALT) → RealizedVol.update → leg Greeks/mids → Book.greeks
→ account + Risk.on_pnl (kill switch) → campaign: candidate, σ_iv, edge_per_day, Gate.evaluate once per session,
  enter / roll / exit → hedge_all → band_u, qp_u, naive_u; chosen → Risk.check_hedge → one working marketable limit
→ reconcile by conId → one decisions row (always written, even when a step raises)
```

`Runner.tick` has no sleeps and no wall clock — the backtest drives it per bar through `SimBroker` + `SimClock` (`backtest/sim.py`), and `tests/test_backtest_sim.py::test_live_and_backtest_identical_decision` asserts that live-shaped and backtest-shaped markets produce the same `band_u/qp_u/naive_u`. `SimBroker` is the only order path outside `--execute` (dry-run wraps `Broker(readonly=True)` in it, so a dry-run never cancels real orders a crashed `--execute` session left resting); fills reach the `Book` only by draining `broker.fills`. Reconcile is time-based (`loop.reconcile_s`) plus after a reconnect (`broker.needs_reconcile`) and is skipped while disconnected; it deliberately does not run after every fill, because IBKR positions can lag the fills. A restart with a held straddle replays the campaign's stored broker fills (`Store.campaign_fills`) into the ledger; when they do not rebuild the broker position, one WARNING is logged and max_loss is measured from the first held tick.

### States

`STARTING → RUNNING ⇄ DEGRADED → HALTED → STOPPING → STOPPED`; `TRANSITIONS` in `src/runner.py` is the whole machine and `Runner.fire(event)` raises `IllegalTransition` on anything else. `halt()` cancels our orders, sets the stop event and never flattens (a halted session is stored as `halted` even if a stop arrives afterwards); `flatten()` returns 0 clean / 1 residual / 2 market closed, and every way out (including Ctrl-C / SIGTERM, which cancel it: exit 1, `flatten interrupted` on stderr) cancels our orders, ends the sessions row as `flatten` and disconnects. A lost broker link anywhere in a tick (`BrokerDisconnected`) is DEGRADED, never a halt; DEGRADED → RUNNING calls `cancel_all` first, then the reconcile a reconnect scheduled, before any decision (ib_async does not replay executions it re-fetches). Staleness halts (`loop.stale_halt_s`) only while the market is open. `--force-open` disables the `exit_confirmed` exit, so a forced campaign holds until roll, hard exit or max loss. A fresh entry and a roll's re-entry pass the same guards (`_entry_guard`: delayed data in execute, `no_entry_last_s`, ex-dividend tomorrow, a working share order). A roll re-evaluates the gate on the new candidate and degrades to an exit when it is CLOSED or the re-entry is blocked.

### Time units

Option `T` is **calendar** years to 16:00 America/New_York (`clock.T_years`); hedging / RV / QP `dt` is **trading** years, seconds / 5,896,800 (`clock.dt_years`). Every datetime is tz-aware (`clock.aware` raises on naive); the only wall-clock read in the codebase is `Clock.now()` in `src/clock.py`, and everything else takes `now` as an argument.

### Configuration

`src/config.py` holds pydantic v2 models with `extra="forbid"` and `frozen=True`; `config/live.yaml` must contain every section and `Live` cross-checks (roll_dte < entry_dte, stale_quote_s < stale_halt_s, …). Defaults live in the models; the YAML lists every key (`tests/test_config.py::test_yaml_lists_every_field` checks keys, not values, so values are tuned per the runbook). `config/backtest.yaml` → `Backtest` plus the `live_config` it validates. No env-var layer. Risk fractions are of NetLiquidation, so the paper account's NLV must be reset to the intended live size (runbook).

### Costs

One convention: a fill executes at bid/ask (the spread is inside the price) and `commission()` + `reg_fees()` ride on `Fill.commission`; `per_share_rate()` and `option_cost()` are planning inputs for the band, QP and gate, never a charge. `option_cost` charges the option half-spread on each leg — deliberately twice the one half-spread per straddle that the BAG limits (net mid ± one half-spread) and SimBroker's fill at the limit pay; backtest P&L uses the fills. `tests/test_backtest_sim.py::test_c5_pnl_regressions` pins the share round trip (2·half-spread + commissions + reg fees); `tests/test_sim_broker.py::test_straddle_fill_prices_and_ledger` pins the BAG fill at its limit.

### Greeks and IBKR units

`bs.greeks` returns per-share delta, gamma (1/$), vega ($/vol-point), theta ($/calendar day); `Book.greeks` scales by qty × 100 and adds shares to delta; the gate converts theta to per trading day. With `greeks.source: ibkr`, IBKR delta/gamma (tick 106) are used per share with Γ·dS carry-forward, and IBKR vega/theta only when `greeks.ibkr_vega_per_volpt` / `ibkr_theta_per_day` are `true`/`false` (`false` = divide by 100 / 365; `null` = Black-Scholes). The shipped default `greeks.source: bs` prices all four Greeks with Black-Scholes at the tick-106 IV; a leg without tick 106 keeps its last IV (WARNING `IV frozen` at most every 10 min per leg). `scripts/check_ibkr.py` measures the ratios and prints the YAML lines.

### Validation

`src/validation.py` returns a typed value or `None`, never a repair: no bid/ask swap, no back-fill, no clamping. A one-sided, crossed, stale or too-wide quote is `None` and the caller decides what "no data" means this tick.

### Store

`src/store.py`: sqlite3, WAL for file-backed stores, foreign keys on, UTC ISO text. Tables `sessions`, `orders` (STK and BAG alike, `INSERT OR REPLACE` on save, `update_order` on state change), `fills` (execId primary key, commission patched by execId from commissionReport), `decisions` (one row per tick; `DECISION_COLUMNS` is the schema). `gate_state` is the only gate column. The only fill getter is `campaign_fills` (restart ledger replay) and there is none for sessions; scripts read with `pd.read_sql_query(sql, store._db)`. The decisions `reason` vocabulary: stale, disconnected, greeks_nan, unpaired, no_chain, no_iv, no_vol, spread_wide, entry_unfilled, whatif_failed, size:<why>, straddle:<why>, the entry guards delayed / late / ex_div / working_order, entry, exit:<reason> (max_loss, hard_exit, exit_confirmed, roll_refused; a roll's exit:roll is overwritten by its re-entry), roll, reconcile_adopt, halt:<reason>, shutdown, plus hedge_all's nan / no_vol / close_hedge / increases_exposure / in_band / hedge; `risk_result`: ok, none, halted, delayed, no_account, min_nlv, pdt, increases_exposure, book_bound.

### Logging

Loguru with two sinks set up once from the entrypoint by `src.log.setup(dir, level)`: human console at `level`, DEBUG JSONL file under `logs/` (the backtest passes `None`: console WARNING, no file). Component loggers via `get_logger("runner")`; structured events are `log.bind(**fields).info("event")`.

## Conventions

- `README.md`'s "Current status (V2)" table uses ✅ / 🚧 / ❌ with an evidence column — no ✅ without a named test; keep it in sync when shipping.
- New tunable knobs go in the YAML **and** the matching pydantic model (`extra="forbid"` rejects anything else). A few operational constants deliberately stay module-level because none has needed tuning: `CHAIN_TTL_S`, `CANDIDATE_MAX_SPREAD`, `SHUTDOWN_S`, `IV_WARN_S` (runner), `STOCK_MAX_SPREAD`, `OPTION_MAX_SPREAD` (validation), `ACK_S`, `OFFSET_INTERVAL_S` (broker), `PDT_NLV` (risk); promote one to the YAML when a paper session shows it binding.
- No wall-clock reads outside `src/clock.py` (`tests/test_clock.py::test_only_wall_clock_read_is_clock_py` enforces it; ruff DTZ forbids naive datetimes); inject `now`.
- No market orders — marketable DAY limits only; no symbol-keyed positions — conId everywhere; no simulate-on-failure — an order `placeOrder` refuses is a REJECTED order, not a pretend fill; with the socket down nothing is sent (`BrokerDisconnected`, not REJECTED, no reject count).
- Typed exceptions are only `BrokerError`, `BrokerDisconnected` (`src/broker.py`) and `IllegalTransition` (`src/runner.py`); the tick loop maps `BrokerDisconnected` to DEGRADED and anything else to HALT.
- `# ponytail:` comments mark deliberate ceilings and their upgrade path (currently two, in `src/vol.py`); add one when you cut a known corner.
- Tests are offline: `tests/fake_ib.py` and `tests/conftest.py::FakeMarket` stand in for TWS; Runner-driven sim tests use 5-min bars to keep the suite at ≈20 s.
- Nothing in `data/` or `logs/` is committed (`.gitkeep` only); the pre-refactor V1 tree is commit `713e934` ("Snapshot the V1 working tree before the V2 refactor"); `git show 713e934:<path>` or `git worktree add ../v1 713e934`.
