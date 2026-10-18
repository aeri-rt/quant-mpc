# quant-mpc

A gated long-gamma SPY straddle, delta-hedged by a cost-aware Whalley–Wilmott band. The system is long gamma only when its own realized-variance forecast beats the candidate straddle's implied variance by enough to pay theta, hedging friction and the option round-trip; it is expected to be flat most of the time while writing one `decisions` row per tick. A receding-horizon QP hedger ships as its closed-form first action for research and is shadow-logged on every tick.

## Quickstart

```bash
python3 -m venv venv && source venv/bin/activate        # Windows: .\venv\Scripts\Activate.ps1
pip install $(python -c "import tomllib;d=tomllib.load(open('pyproject.toml','rb'))['project'];print(' '.join(d['dependencies']+d['optional-dependencies']['dev']))")
# or the pinned set: pip install -r requirements.lock
pytest -q                                               # 420 offline tests, ≈20 s, no broker

python scripts/run_backtest.py --synthetic --hedger all --force-open   # bars are generated from backtest.yaml's seed
python scripts/fetch_data.py --cboe                     # Cboe VIX9D/VIX/VIX3M closes → data/cache (real-bar backtests also need --ibkr)
python scripts/run_backtest.py --ibkr --hedger all --sweep --start 2024-10-07 --end 2025-03-31   # real bars cached by fetch_data.py --ibkr; 60 warm-up sessions, P&L from 2025-01-02

python scripts/check_ibkr.py                            # TWS paper on 7497: read-only checklist, never orders
python scripts/run_paper.py                             # dry-run (default): quotes from TWS, orders to SimBroker
python scripts/run_paper.py --execute --contracts 1     # paper orders through TWS
python scripts/run_paper.py flatten --execute           # liquidate straddle + shares (the only liquidation path after a HALT; a plain restart resumes with the held book)
```

`--live` additionally requires `--execute`, `--port 7496` on the command line and a typed `CONFIRM`; `--force-open` is refused with `--live`. Exit codes: 0 stopped, 3 halted (reason printed), 1 anything else. `flatten`: 0 flat, 1 residual or interrupted, 2 market closed.

## Architecture

`src/` and `backtest/` are flat peer packages rooted at the repo (`pytest` uses `pythonpath = ["."]`).

| module | one line |
|---|---|
| `src/clock.py` | tz-aware time base, XNYS sessions, `T_years` (calendar) vs `dt_years` (trading); the project's only wall-clock read |
| `src/types.py` | frozen dataclasses Quote, OptionQuote, IbkrGreeks, Fill, Position, Account; mutable Order; OrderState enum |
| `src/config.py` | pydantic v2 models (`extra="forbid"`, frozen) for `config/live.yaml` and `config/backtest.yaml`; defaults live here |
| `src/log.py` | loguru: console sink + JSONL file sink via `setup(dir, level)` |
| `src/costs.py` | the one cost model: `commission`, `reg_fees` (charged on fills); `per_share_rate`, `option_half_spread`, `option_cost` (planning inputs only) |
| `src/bs.py` | Black-Scholes price, Greeks (per-share, documented units), implied vol |
| `src/book.py` | the one ledger: legs + shares + cash, Greeks at an injected `now`, `from_positions`/`diff` by conId; units table |
| `src/vol.py` | realized vol from 1-min mids: overnight² + 5-min subsampled RV, diurnal profile, EWMA → log-HAR forecast |
| `src/gate.py` | `edge_per_day` ($/trading day) and the once-per-session OPEN/CLOSED gate with hysteresis and cooldown |
| `src/hedge.py` | Whalley–Wilmott band (production), closed-form QP (research), naive −Δ; `hedge_all` returns all three |
| `src/risk.py` | NLV-fraction limits, sizing, PDT guard, daily-loss kill switch with persistence and open grace |
| `src/validation.py` | broker-boundary checks that return a typed value or `None`, never a repair |
| `src/store.py` | sqlite WAL audit trail: `sessions`, `orders`, `fills` (by execId), `decisions` (one row per tick); `campaign_fills` replays a recovered straddle's ledger |
| `src/sim_broker.py` | the only order path outside `--execute`: fills share orders on the next quote at bid/ask |
| `src/broker.py` | ib_async: conId positions, tick-106 Greeks, one-expiry chain, BAG + whatIf, typed order states, reconnect |
| `src/runner.py` | lifecycle STARTING→RUNNING⇄DEGRADED→HALTED→STOPPING→STOPPED, `tick()`, entry/roll/exit, reconcile, `flatten()` |
| `backtest/data.py` | Cboe CSV + parquet cache, IBKR 1-min MIDPOINT bars per session, synthetic GBM bars on XNYS sessions |
| `backtest/iv.py` | prior-day ATM IV from the VIX term structure with the DDKZ haircut |
| `backtest/sim.py` | `SyntheticMarket` + `SimBroker` + `SimClock` driving the same `Runner.tick` per bar; per-session attribution |
| `backtest/report.py` | summary / attribution / comparison / frontier tables, three PNGs, CSVs |

### Tick pipeline (`Runner.tick`, every `loop.interval_s`; identical in dry-run, `--execute` and backtest)

```
1. q = broker.quote(SPY); drain broker.fills → Book.apply_fill (+ commission updates by execId)
2. quote age → Risk.check_stale: fresh | stale → DEGRADED (no decisions, fills still drained) | > 600 s stale while the
   market is open → HALT (after hours / holidays: DEGRADED only); DEGRADED → RUNNING cancels orders the outage left
   working, then reconciles if due (before any decision)
3. S = q.mid; RealizedVol.update(ts, S); σ_f = forecast_sigma()  (None until 5 sessions)
4. per held leg: tick-106 IV (and Greeks under greeks.source: ibkr) and the leg mid — without tick 106 the IV stays
   frozen (WARNING at most every 10 min per leg); Greeks = Book.greeks(S, now)  (ValueError → reason greeks_nan)
5. account; Risk.on_pnl → daily-loss kill switch (3-tick persistence, no halt before 09:35)
6. campaign (RUNNING and market open): ATM candidate from the 300 s chain cache, σ_iv, edge_per_day →
   Gate.evaluate once per session (10:00) or at roll; enter if OPEN (or --force-open) unless an entry guard
   blocks it (reason delayed [execute, SPY or candidate-leg md_type ≠ 1] | late | ex_div | working_order); hard_exit ≤ 1 DTE,
   roll ≤ 7 DTE (gate re-evaluated on the new candidate and the re-entry guarded, else exit), exit_confirmed,
   max_loss; completed actions stamp reason entry | exit:<reason> | roll
7. hedge: hedge_all → band_u, qp_u, naive_u (all logged); chosen → Risk.check_hedge →
   one working marketable DAY limit (ask+0.01 / bid−0.01); older than fill_timeout_s → cancel
8. reconcile by conId every reconcile_s or after a reconnect (skipped while disconnected; not after each fill, positions can
   lag fills): adopt share drift ≤ 50, halt above or on a leg mismatch (execute)
9. one decisions row per tick, written even when a step raises
```

### Backtest bar pipeline (`backtest/sim.run`)

```
bars (1-min XNYS: synthetic GBM, or cached IBKR MIDPOINT) + term structure (prior-day Cboe; synthetic: nodes
  iv·√(1+3·T·skew_b²), so atm_iv = iv at 9/30/91 d and drifts slightly between nodes)
first vol.har_min_sessions (60) sessions → RealizedVol warm-up only, excluded from P&L
per bar: SimClock.set(ts) → SimBroker fills last bar's share orders at open ± half_spread → Runner.tick() at close ± half_spread
  (an order placed on a session's last bar present in the data fills at that bar's close ± half_spread, not overnight)
options are Black-Scholes at the prior session's haircut ATM IV; expiries are XNYS sessions, monthlies preferred;
  sizing uses the constant backtest.initial_nlv, so every hedger holds the same contract count
attribute() per session (delta, gamma, theta, vega, hedge_spread, commissions, hedge_friction, residual) → tables, equity/attribution/delta PNGs, daily_*.csv + decisions_*.csv
```

`run_backtest.py` flags: `--synthetic` | `--ibkr` (force `source:`; `--ibkr` reads the bars cached by `fetch_data.py --ibkr` and exits with that hint when none exist), `--hedger band|qp|naive|all`, `--sweep` (γ/λ frontier), `--force-open`, `--interval S`, `--start/--end`, `--out DIR`. Synthetic Jan–Jun (measured 2026-10-18): `--hedger all --force-open` at 1-min bars ≈ 180 s, `--hedger all --interval 300 --force-open` ≈ 32 s, `--hedger band --interval 300` ≈ 13 s.

The summary's frontier axes are `hedge_friction_per_day` (stock half-spread + stock commissions; option round trips are common to every hedger and excluded) and `hedge_error_std` (std of the daily delta + residual P&L); `comparison_table` and the `--sweep` frontier print both. Option fills: the runner sends BAG limits at net mid ± one option half-spread and SimBroker fills a BAG at its limit; the gate's `option_cost` charges a half-spread on each leg (deliberately conservative: twice the spread the fills pay); backtest P&L uses the fills.

### Units (full table in `src/book.py` and ADR-004)

| quantity | convention |
|---|---|
| timestamps | tz-aware; UTC internally, America/New_York for sessions; naive raises |
| T (pricing, theta) | calendar years to 16:00 ET of the expiry date |
| dt (hedging, RV, QP) | trading years = seconds / (252·6.5·3600) |
| σ | annualized decimal; variance = σ² |
| Greeks | per share: delta ∈ [−1,1], gamma 1/$, vega $/vol-point, theta $/calendar day; portfolio = × qty × 100, delta includes shares |
| theta in the gate | per trading day (× 365/252) |
| IBKR Greeks | only with `greeks.source: ibkr` (default `bs`: all four are Black-Scholes at the tick-106 IV); vega/theta then also need `greeks.ibkr_vega_per_volpt` / `ibkr_theta_per_day` set from `check_ibkr.py`; `null` = Black-Scholes |
| P&L | fills at bid/ask (spread in the price) − commissions and reg fees on the fill; `per_share_rate()` / `option_cost()` are planning inputs, never a charge |

Config: `config/live.yaml` (every section required, unknown keys rejected) and `config/backtest.yaml` (backtest-only keys + the `live_config` it validates). No solver: the hedger has no casadi/do-mpc/scipy optimizer or QP solver and none will be added (the N-step program exists only as a scipy SLSQP test oracle); scipy in production is limited to `norm` and the `brentq` implied-vol root in `src/bs.py`.

## Current status (V2)

| Component | Status | Evidence |
|---|---|---|
| Clock / calendar (XNYS, 16:00 ET expiry, tz-aware) | ✅ | tests/test_clock.py |
| Config (pydantic, extra=forbid, one live.yaml) | ✅ | tests/test_config.py |
| Black-Scholes + units table | ✅ | tests/test_bs.py (hypothesis), tests/test_book.py |
| Book ledger (live = backtest) | ✅ | tests/test_book.py |
| Realized vol (EWMA, log-HAR, diurnal profile) | ✅ | tests/test_vol.py null tests |
| Alpha gate (IV vs forecast RV, once per session) | ✅ code / 🚧 edge unproven | tests/test_gate.py; owner's decisions dataset |
| Whalley–Wilmott band hedger (production) | ✅ | tests/test_hedge.py |
| Receding-horizon QP (closed form, research) | ✅ shadow-logged / 🚧 acceptance pending | test_qp_closed_form_equals_program; ADR 004 rule |
| Risk (NLV fractions, kill switch w/ persistence, PDT guard) | ✅ | tests/test_risk.py |
| Broker (ib_async, conId positions, BAG+whatIf, order states) | ✅ offline / 🚧 paper-verified | tests/test_broker.py; paper checklist |
| SimBroker (single order path for dry-run/backtest) | ✅ | tests/test_sim_broker.py |
| Runner (lifecycle, DEGRADED/HALT, reconcile, flatten) | ✅ offline / 🚧 paper-verified | tests/test_runner.py; paper checklist |
| Store (WAL, FKs, decisions dataset) | ✅ | tests/test_store.py |
| Backtest (sim-clock Greeks, attribution, band/qp/naive) | ✅ synthetic / 🚧 real bars | tests/test_backtest_sim.py; `run_backtest.py --synthetic` |
| IBKR Greek units verified | 🚧 | scripts/check_ibkr.py (manual) |
| Short gamma | ❌ not implemented | ADR 004 |

Minimum viable live size: ≥ 5 straddles at 30 DTE (≈ $10k premium at 15% IV, ≈ $14k at 20%; NLV ≈ $70–95k at `risk.max_premium_frac_nlv: 0.15`); below that, paper research only.

## Safety

- `SimBroker` is the only order path without `--execute`; there is no "simulate on failure" branch, and `readonly=True` is not the safety belt.
- Positions are keyed by conId; `Book.from_positions` refuses anything that is not exactly one straddle (execute halts at start, dry-run observes with an empty book and never reconciles that session).
- The straddle is one guaranteed SMART BAG with `whatIf` first; unequal leg fills halt with the conIds listed.
- Order states are typed: 10147/10148 → re-query, Inactive → cancel + re-query, unknown TWS status → UNKNOWN and re-queried; every order is persisted at creation and on every change.
- Share hedges are marketable DAY limits (ask+0.01 / bid−0.01), never market orders; at most one working hedge, cancelled after `fill_timeout_s`.
- Every order carries `orderRef qmpc:<session>`; rebind and `cancel_all` touch only those.
- Quotes carry the broker arrival timestamp, are never back-filled, and are rejected when crossed, one-sided or wider than 0.5% (10% for options). Stale > 15 s → DEGRADED (no orders), > 600 s while the market is open → HALT (after hours a stale feed never halts). In execute mode delayed data (`marketDataType ≠ 1`) on the SPY ticker means no hedge (`risk_result = delayed`); delayed data on the SPY ticker or the candidate legs means no entry (`reason = delayed`) and no roll re-entry (the roll exits and the row reads `reason = exit:roll_refused`; the log says `roll re-entry refused (delayed): exited`); exits still go out.
- Kill switch: min(broker dailyPnL, book P&L since open) below −2% NLV for 3 consecutive ticks and not before 09:35; also stale > 600 s during regular hours, leg mismatch at reconcile (execute; dry-run adopts), unpaired legs at start (execute), an exit with units left after the touch, share drift > 50, ≥ 3 rejects in 10 min, unwind needed, unhandled exception (a lost broker link is DEGRADED, never a halt).
- HALT cancels our open orders and **never flattens**; liquidation is `run_paper.py flatten --execute` (unpaired legs must be closed by hand in TWS). A halted session is stored with status `halted`.
- Ctrl-C / SIGTERM is a clean STOP: orders cancelled, positions left in place, restart recovers them by conId, replays the campaign's stored broker fills so realized P&L, commissions and premium survive (a WARNING says when they do not rebuild the position), and never opens a second straddle. In `flatten --execute`, Ctrl-C / SIGTERM cancel the flatten and its open orders (exit 1).
- **Dedicated account assumption:** `Book.from_positions` adopts every SPY share in the account as the hedge, and an execute-mode start flattens shares that have no straddle — run it in an account (or sub-account) that holds nothing else in SPY.

Sign-off before `--live`: [docs/runbooks/paper-checklist.md](docs/runbooks/paper-checklist.md); day-to-day operation: [docs/runbooks/runbook.md](docs/runbooks/runbook.md).

## Documentation

[docs/README.md](docs/README.md) — ADRs ([004](docs/decisions/004-v2-thesis-and-hedger.md) is the current thesis), [setup guide](docs/guides/setup.md), runbooks, the [Clinet 2021 paper note](docs/research/papers/clinet-2021-mpc.md).

## Disclaimer

Research software. Trading involves substantial risk of loss; nothing here is investment advice. MIT License.
