# ADR-004: V2 thesis — the edge is the gate, the hedger is a band, the QP ships in closed form

## Status
Accepted. Supersedes ADR-002 `002-greeks-mpc-integration.md` (deleted) and the deleted `docs/design/mpc-controller-design*.md` / `architecture-overview.md`. The owner's intent document [gamma-scalping-v2-alpha-spec.md](../design/gamma-scalping-v2-alpha-spec.md) is kept for reference with a header listing what this ADR corrects.

## Date
2026-10-17

## Context
V1 bought an ATM SPY straddle at startup and delta-hedged it with a 20-step do-mpc/IPOPT program. The verified facts (analysis probes C1–C16, pre-refactor tree committed as `713e934`): the optimizer was numerically "hedge ≈97% to flat every tick" (shares² risk against $ costs; the linear cost term was inert at ×1000); it labelled Infeasible/NaN/Max_Iter solves `optimal` and traded them; κ sat at its clamp and was dead; the backtest computed Greeks at wall-clock T=0 (gamma ≡ 0), omitted hedge mark-to-market and double-counted slippage; the live path keyed positions by symbol (stock + call + put collapsed into one entry), fetched 0–2 DTE chains regardless of `--dte`, legged the straddle with no cancel, and had an unreachable kill switch that deadlocked. The cited paper (Clinet–Perreton–Reydellet 2021) is an optimal-execution paper, not a hedging one ([summary](../research/papers/clinet-2021-mpc.md)).

## Decision

### 1. The edge is the gate, not the controller
Expected P&L of a delta-hedged straddle is Σ ½·Γ·S²·(σ_realized² − σ_implied²)·dt − theta carry − hedging costs − option round-trip; no hedging policy changes its sign. SPX realized variance averages ≈55% of implied (Carr–Wu), so the system is **long gamma only when** a forecast of close-to-close realized variance exceeds the candidate straddle's ATM implied variance by enough: `gate.edge_per_day` ≥ $10 per trading day **and** σ_f − σ_iv ≥ 1 vol point (`config/live.yaml: gate`). The gate is evaluated **once per session** at/after 10:00 ET (`Gate.due`) and again at roll time for the new candidate; it is expected to be CLOSED most of the time. While CLOSED the paper loop keeps running and writes one `decisions` row per tick (quote, σ_f, `iv_atm` = the candidate's σ_iv, `edge_day`, `leland_a`, `gate_state`; the Greeks and `band_u/qp_u/naive_u` columns describe the held book, so they are 0/NULL while flat) — the owner's own point-in-time IV-vs-RV dataset. Short gamma is **not** implemented: Greeks, book, gate, hedger, risk and attribution are signed and sign-agnostic (`tests/test_gate.py` asserts the inversion), so it is a later config flip, after that dataset exists.

### 2. The production hedger is a Whalley–Wilmott band; the QP is the research path and ships as its closed-form first action — final design
Band half-width `H = ((3/2)·e^{−rT}·k·S·Γ_p²/γ)^{1/3}` shares, `k` = one-way per-share cost ÷ S for the side about to trade (sells include the SEC fee), with one fixed-point step that folds the $0.35 per-order minimum into `k` at the expected trade size (1−f)·H₀; interior target sign(Δ)·f·H with f = `hedge.inner_fraction` = 0.5; H clipped to [5, 300] shares (`src/hedge.py: band_half_width, band_k`).

The receding-horizon QP — risk λ·Σ_k σ_k²·S²·dt·D_k² with the diurnal profile, asymmetric linear costs, overnight terminal weight, N := steps to the session close — has the closed-form first action **u₀ = −sign(D)·max(0, |D| − c_side/(2·λ·S²·W_var))**, W_var = Σ_k σ_k²·dt_k + overnight_frac·σ_f²/252 (`vol.horizon_variance`). The repo guards it with `tests/test_hedge.py::test_qp_closed_form_equals_program` — a scipy SLSQP oracle of the full N-step program over 40 random cases (diurnal weights, asymmetric costs, u_max, q_max, terminal weight), |u₀ − u_cf| < 1e-3 — and `test_qp_single_step_textbook`, the one-step soft-threshold identity. (Historical: the pre-refactor analysis also matched it to 1e-10 against qpOASES over 200 random cases; qpOASES is not a dependency, so that check is not reproducible from this repo.) **This closed form is the final design, not a provisional one: no numerical solver path (casadi, do-mpc, scipy or otherwise) will be added to production code** (owner decision, 2026-10-17). If a future term ever breaks the closed form, that test says so and the term is the thing to argue about. Both policies and the naive −Δ baseline are shadow-logged on every tick (`band_u, qp_u, naive_u`); `hedge.policy` picks which one trades. All policies hedge to centre once inside the last `loop.close_window_s` (300 s) of the session (Muravyev–Ni: overnight gap variance ∝ Δ²).

**Acceptance rule for the QP.** `hedge.policy: qp` replaces `band` only if, on real 1-min SPY bars over two disjoint ≥3-month windows out of sample, the QP's (hedge friction/day, hedge-error std) frontier lies below the band's: lower `hedge_friction_per_day` (stock half-spread + stock commissions; option round trips excluded, they are common to every hedger) at equal-or-lower `hedge_error_std` (std of the daily delta + residual P&L), p < 0.05 on paired daily differences. Until then the band is production and the QP is research.

### 3. One time base, one cost function, one lifecycle, one ledger, one tick
Option T is calendar years to 16:00 America/New_York; hedging/RV dt is trading years (seconds/5,896,800); XNYS calendar for sessions and early closes; a `Clock` is injected everywhere and the project's only wall-clock read is `Clock.now()` in `src/clock.py`. `src/costs.py` is the single cost model for band, QP, gate, risk and SimBroker: fills execute at bid/ask (the spread is in the fill price), commission + regulatory fees are charged on the fill, `per_share_rate()`/`option_cost()` are planning inputs and never charge anything. The gate's `option_cost` deliberately charges the option half-spread on **each leg** (two half-spreads per straddle per order), while the runner's BAG limits (entry at net mid + one half-spread, first exit attempt at net mid − one) and SimBroker, which fills a BAG at its limit, pay one half-spread per straddle: the gate over-states option friction (conservative), and backtest P&L comes from the fills, never from `option_cost`. One lifecycle: enter at ~30 DTE (the monthly within ±5 days when `position.prefer_monthly`, live and backtest alike), roll (gate OPEN) or exit (CLOSED) at ≤7 DTE, hard exit at ≤1 DTE, exit at −50% of premium. One `Book` ledger and one `Runner.tick` shared by live, dry-run and the backtest (`SimBroker` + `SimClock`; `tests/test_backtest_sim.py::test_live_and_backtest_identical_decision`).

### 4. Safety before strategy
ib_async; positions keyed by conId (`Book.from_positions` refuses anything that is not one straddle); `SimBroker` is the only order path outside `--execute` and there is no "simulate on failure" branch; straddle as one guaranteed SMART BAG with `whatIf` first, unequal leg fills halt with the conIds listed (UNWIND); typed order states incl. 10147/10148 → re-query (never "rejected"), Inactive → cancel + re-query, unknown TWS status → UNKNOWN; cancel on timeout; at most one working hedge (cancel-and-replace); marketable DAY limits only, never a market order; every order carries `orderRef qmpc:<session>` and rebind/cancel_all touch only those; quotes carry the broker arrival timestamp, never now(), never back-filled; stale → DEGRADED (no orders, fills still drained, reconcile runs while the socket is up) → HALT after 600 s stale during regular hours (never while the market is closed); a lost broker link anywhere in a tick is DEGRADED, never a halt, and DEGRADED → RUNNING first cancels any order the outage left working and reconciles before any decision; reconcile is time-based (`loop.reconcile_s`) plus after a reconnect, deliberately not after every fill, because IBKR positions can lag the fills; in execute mode delayed data (`marketDataType ≠ 1`) on the SPY ticker refuses hedges (`risk_result = delayed`), and on the SPY ticker or the candidate legs refuses entries (`reason = delayed`) and roll re-entries (the roll exits and the row reads `reason = exit:roll_refused`; the log says `roll re-entry refused (delayed): exited`), never exits; the kill switch is a flag with 3-tick persistence and a 09:35 open grace; HALT cancels our orders and **never flattens** — liquidation is the operator's `run_paper.py flatten --execute`; `readonly=True` is not a safety belt.

## Consequences

### Positive
- `src/` went from ≈7,500 LOC in 7 subpackages (the archived tree: 24 files, 7,465 lines in control, core, execution, models, risk, strategy, utils) to 3,754 LOC in 16 flat modules (`wc -l` on 2026-10-18: backtest 849, scripts 789, tests 5,422); do_mpc, casadi, ib_insync, yfinance, pydantic-settings and sqlalchemy are gone from the venv; `pip check` is clean; 419 offline tests run in ≈20 s (22 s wall on 2026-10-18).
- The hedger has one parameter per policy (γ for the band, λ for the QP) and both are compared on the same (hedge friction/day, hedge-error std) frontier (`run_backtest.py --hedger all --sweep`).
- Every tick leaves a `decisions` row, so the slow part (the gate being closed most of the time) still produces the dataset the next decision needs.

### Negative
- Live validation is slow because the gate is closed most of the time — that is the finding, not a bug; `--force-open` exists for paper plumbing checks only.
- The backtest validates the hedger and the accounting, not the gate's edge: its ATM IV is a prior-day Cboe VIX9D/VIX/VIX3M proxy with the DDKZ haircut (`backtest/iv.py`).
- A halted backtest returns a normal-looking `Result` (a WARNING is logged, there is no halted flag yet).

### Risks / open items
- IBKR Greek units are unverified until `python scripts/check_ibkr.py` is run and `greeks.ibkr_vega_per_volpt` / `ibkr_theta_per_day` are set; until then vega/theta are Black-Scholes. IBKR tick-106 Greeks are used only under `greeks.source: ibkr` (delta/gamma per share with Γ·dS carry-forward, vega/theta only with the flags set); the shipped default `bs` prices all four Greeks with Black-Scholes at the tick-106 IV, so the flags do not matter there.
- Minimum viable live size is ≥ 5 straddles: at 30 DTE that is ≈ $10k premium at 15% IV (≈ $14k at 20%), so NLV ≈ $70–95k at `risk.max_premium_frac_nlv: 0.15`; below that the $0.35 minimum and option spreads dominate and the gate shows it as negative edge.
- `gate.turnover_frac` (2.0) is the gate's assumption about realised band turnover; the backtest reports `summary["turnover_frac_realised"]` (1.6–2.4 on synthetic Jan–Jun 5-min bars, band, `--force-open`, seeds 1–7; re-measured 2026-10-18) and `tests/test_backtest_sim.py::test_turnover_sanity` bounds it to [tf/2, 2·tf]. Re-measure on real bars.
- The band-vs-QP decision on real bars is pending. Commands: `python scripts/fetch_data.py --cboe`, `python scripts/fetch_data.py --ibkr --start 2024-10-07 --end 2025-03-31` and again for a disjoint second window, then `python scripts/run_backtest.py --ibkr --hedger all --sweep --start … --end …` on each (the first `vol.har_min_sessions` (60) sessions of every window are RealizedVol warm-up only, so fetch ≈3 months of bars before each ≥3-month evaluation window; 2024-10-07 → 2025-03-31 trades from 2025-01-02) (`--ibkr` forces `source: ibkr`; with nothing cached it exits with the `fetch_data.py --ibkr` hint); apply the acceptance rule above.
- Cost constants drift (Section 31 rate, FINRA TAF from 2027-01-01): `scripts/calibrate_costs.py` compares booked commissions with the model after an execute session.

## Alternatives Considered

### Keep do-mpc and fix the objective
Rejected: with E[dS] = 0 the program's optimum is a threshold on |Δ|, which the band and the closed form already are; the solver only added failure modes (mislabelled solves, 20-step horizons with nothing to predict).

### A numerical QP solver in production (qpoases / scipy) behind `hedge_all`
Rejected by owner decision: the closed form is exact for this program and the SLSQP oracle in the test suite proves it; a solver would be a second implementation of the same number.

### Zakamouline's full fixed-plus-proportional approximation
Not chosen: adds σ-dependent exponents and more fitted constants for a question the frontier sweep answers empirically; Whalley–Wilmott with the min-commission fixed point is the one-parameter version.

### Short gamma now
Not chosen: the IV-vs-RV dataset does not exist yet; the code is sign-agnostic so the flip is a config change later.

## Units and time base (canonical; the same table heads `src/book.py`)

| Quantity | Unit / convention | Where enforced |
|---|---|---|
| timestamps | tz-aware; UTC internally, America/New_York for session logic; naive datetime → `ValueError` (ruff DTZ on) | clock.aware |
| option expiry | 16:00:00 America/New_York of the listed date | clock.expiry_at |
| T (option pricing, theta) | calendar years = max(0, seconds to expiry)/(365·86400) | clock.T_years |
| dt (hedging, RV, QP) | trading years = seconds/(252·6.5·3600) = seconds/5,896,800; RV from the minute-of-session grid of actual timestamps, the QP horizon from `loop.interval_s` steps to the close | clock.dt_years, vol |
| σ (every σ) | annualized decimal (0.15); variance = σ² | vol, gate, hedge |
| per-share Greeks | delta ∈ [−1,1]; gamma 1/$; vega $ per 1 vol-point (= ∂V/∂σ / 100); theta $ per **calendar day** (= ∂V/∂t_year / 365) | bs.greeks |
| portfolio Greeks | per-share × qty × mult, qty signed; delta includes `shares` | book.greeks |
| theta in the gate | per **trading** day = theta_day · 365/252 (same basis as ½ΓS²σ²/252) | gate.edge_per_day |
| IBKR modelGreeks | only with `greeks.source: ibkr` (default `bs`: all four Greeks are BS at the tick-106 IV): delta, gamma per share with Γ·dS carry-forward; vega/theta only if `greeks.ibkr_vega_per_volpt` / `ibkr_theta_per_day` are set (true = already per vol-pt / per day; false = ÷100 / ÷365; null = unverified → BS) | book.Leg.greeks, validation.ibkr_greeks |
| prices | $; shares int; contracts int; multiplier 100 | types |
| costs | $; band k = per_share_rate/S (dimensionless); W_var in years·σ² | costs, hedge |
| P&L | realized + Σ(mid − avg_cost)·qty·mult + (S − share_avg_cost)·shares − commissions (reg fees ride on `Fill.commission`); fills at bid/ask carry the spread in the price | book, sim_broker |
| session | XNYS (holidays, 13:00 early closes); `seconds_to_close` = ∞ when closed | clock |
| risk fractions | of NetLiquidation from accountSummary; paper NLV must be reset to the intended live size (runbook) | risk |

## References
- Whalley, A. E. & Wilmott, P. (1997). An asymptotic analysis of an optimal hedging model for option pricing with transaction costs. *Mathematical Finance* 7(3). — the band.
- Zakamouline, V. (2006). European option pricing and hedging with both fixed and proportional transaction costs. *Journal of Economic Dynamics and Control* 30. — the fixed-cost (min-commission) correction.
- Derman, E. & Kamal, M. (1999). When you cannot hedge continuously: the corrections of Black–Scholes. *Risk* 12(1). — hedging-error std √(π/4N), the Monte-Carlo oracle `derman_kamal` in `tests/test_backtest_sim.py`.
- Demeterfi, K., Derman, E., Kamal, M. & Zou, J. (1999). More than you ever wanted to know about volatility swaps. Goldman Sachs QS. — the DDKZ haircut in `backtest/iv.py`.
- Carr, P. & Wu, L. (2009). Variance risk premiums. *Review of Financial Studies* 22(3). — why the gate is closed most of the time.
- Muravyev, D. & Ni, X. (2020). Why do option returns change sign from day to night? *Journal of Financial Economics*. — the close-window hedge.
- Clinet, S., Perreton, J.-F. & Reydellet, S. (2021). Optimal trading: a model predictive control approach. arXiv:2110.11008. — what V1 cited; [summary](../research/papers/clinet-2021-mpc.md).
- Implementation: `src/hedge.py` (band, QP closed form, naive), `src/gate.py` (edge, hysteresis), `src/runner.py` (lifecycle, safety); operations: [docs/runbooks/runbook.md](../runbooks/runbook.md), [paper checklist](../runbooks/paper-checklist.md).
