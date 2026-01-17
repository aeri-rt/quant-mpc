# Design Document: Gamma Scalping V2 Alpha Spec (Selective Long Gamma + Cost-Aware MPC)

## Overview
This document specifies **V2 alpha logic** for the repo’s gamma scalping system: **selectively acquire long gamma** only when the expected **realized variance** exceeds **implied variance** by a cost-adjusted margin, then **harvest gamma** using a **cost-aware MPC hedger** with an explicit no-trade region / hedge cadence control.


**Template source:** `docs/templates/DESIGN_DOC.md`  
**Primary V1 references:**  
- `docs/decisions/002-greeks-mpc-integration.md`  
- `docs/design/mpc-controller-design-v1-revised.md`  
- `src/strategy/gamma_scalping.py`  
- `src/control/mpc_controller.py`  
- `src/models/greeks.py`

---

## Goals
- Turn the current system from “always-on hedging” into **selective, positive-expectancy long gamma campaigns**.
- Make hedging **explicitly cost-aware** (spread/slippage/commissions) so the system avoids churn.
- Add a **position selection and timing policy** (“when/what to buy”) that can plausibly create alpha.
- Provide a measurable framework: **PnL attribution**, cost accounting, and criteria to evaluate edge.

## Non-Goals
- Not attempting full vol-surface modeling, deep neural forecasting, or multi-asset hedging in V2.
- Not implementing market-making or intraday latency competition.
- Not promising a standalone money printer. This is a research-grade V2 spec aimed at *measurable* improvement.

---

## Background

### Key identity: hedger vs alpha
- **MPC** is the **hedging controller** (engine efficiency).
- **Alpha** comes from **gamma acquisition**: *which options, when, and at what implied vol / liquidity conditions*.

### Why selective long gamma matters on SPY
SPY index options often embed a volatility risk premium (IV > RV on average). A naive “always long straddle + hedge” tends to lose due to **theta + costs**, unless you consistently buy *cheap* gamma or enter ahead of underpriced movement.

### Core V2 thesis
Run long-gamma campaigns only when:
1) **Forecast realized variance > implied variance** by a margin, and  
2) The expected gamma capture **exceeds theta + costs**, and  
3) Liquidity is sufficient to hedge at a sustainable cadence.

---

## Design

## Definitions & Units
From `src/models/greeks.py` (`PortfolioGreeks`):
- `delta`: share-equivalent delta (shares to hedge)
- `gamma`: change in delta per **$1 move** in underlying
- `theta`: **$ decay per day** (already scaled by position sizes)

Volatility conventions:
- `sigma_fore`: forecast realized vol (annualized)
- `sigma_imp`: implied vol (annualized)
- 252 trading days per year

---

## System Components (V2 additions)

### 1) Gamma Campaign Manager (state machine)
Add a small state machine that decides whether the system should be long gamma at all.

**States**
- `OFF`: no long gamma positions (or minimal baseline)
- `ARMED`: watching for entry conditions (data valid, liquidity OK)
- `ACTIVE`: long gamma positions on; hedging + monitoring
- `WINDDOWN`: exiting positions due to time, edge decay, or risk triggers

**Responsibilities**
- Enforce entry/exit gates (edge, liquidity, risk)
- Track “campaign metadata” (entry time, entry IV, target horizon, etc.)
- Decide whether to *roll* or *close* as DTE decays

**Suggested file**
- `src/strategy/campaign.py`  
  - `GammaCampaignManager`
  - `CampaignState` enum
  - `CampaignDecision` dataclass

---

### 2) RV Forecast + IV Snapshot (signals)
V2 requires robust estimates:
- Forecast realized volatility over the intended holding/hedging horizon
- Current implied vol (ATM proxy) from mid quotes

**RV forecast (V2 default): EWMA realized vol**
- Use log returns: `r_t = ln(S_t / S_{t-1})`
- EWMA variance: `v_t = λ v_{t-1} + (1-λ) r_t^2`
- Annualize: `sigma_fore = sqrt(v_t * 252)`

**IV snapshot**
- Use ATM straddle mid IV, or (call+put) ATM IV average.
- Smooth with EMA to reduce microstructure noise.
- Reject stale data (timestamp checks).

**Suggested file**
- `src/alpha/signals.py`  
  - `EWMARealizedVol`
  - `ImpliedVolSnapshot`
  - `MarketQuality` (spread, depth proxies)

---

### 3) Option Selector (what to buy)
Select option package(s) that maximize **gamma efficiency** subject to liquidity and risk.

**Candidate set (V2)**
- Underlying: SPY
- Expiries: {0DTE, 1DTE, weekly nearest} OR configurable (e.g., 1–7 DTE)
- Strikes: ATM ± small band (e.g., closest strike to spot)

**Scoring metrics**
For each candidate package (e.g., ATM straddle):
- `Γ_pkg`: portfolio gamma (from Greeks calc)
- `Θ_pkg`: portfolio theta (from Greeks calc, $/day)
- `liq_score`: based on bid/ask spreads, volume, open interest, quote freshness
- `vega_guard`: penalize excessive vega if IV is elevated / event risk

**Primary efficiency scores**
- **Gamma/theta efficiency**: `E1 = Γ_pkg / |Θ_pkg|`  
  (higher is better, but not sufficient)
- **Cost-adjusted edge per day** (see below): `E_edge_day`

**Suggested file**
- `src/alpha/option_selector.py`  
  - `OptionCandidate`
  - `OptionPackage`
  - `OptionSelector.select_package(market_state) -> OptionPackage`

---

### 4) Cost Model (make the controller honest)
Add an explicit transaction cost model used by:
- Entry/exit gating
- Hedge cadence decisions
- MPC parameterization

**V2 cost components**
- **Stock hedge cost per share**:  
  `tc_share = half_spread_stock + est_slippage + commissions_per_share`
- **Option entry/exit cost**:  
  `tc_opt_roundtrip ≈ (half_spread_option * 2) * contract_multiplier * n_contracts`
  (amortize over expected holding days if needed)

**Expected hedge turnover proxy**
A simple analytic approximation for expected hedge shares per day:
- `E|ΔS| ≈ S * sigma_fore * sqrt(2 / (π*252))`
- `E|ΔΔ| ≈ |Γ| * E|ΔS|`
- `E[shares_traded/day] ≈ E|ΔΔ|`  (if trying to stay near delta-neutral)

Then:
- `tc_hedge_day ≈ tc_share * |Γ| * S * sigma_fore * sqrt(2 / (π*252))`

This is a **first-order** approximation—good enough to prevent fantasy backtests.

**Suggested file**
- `src/alpha/costs.py`  
  - `CostModel.estimate_daily_costs(greeks, market_quality, sigma_fore)`

---

### 5) Alpha Gate (entry/exit) — the core V2 logic

#### 5.1 Variance edge condition
Compute:
- `var_edge = sigma_fore^2 - sigma_imp^2`

A pure variance premium condition:
- Enter only if `var_edge > var_edge_min` (cost-adjusted)

But to incorporate *position-specific gamma*, compute expected edge per day:

#### 5.2 Cost-adjusted expected edge per day (position-aware)
For a delta-hedged long gamma package, a useful proxy is:

\[
E[\text{PnL/day}] \approx \frac{1}{2}\Gamma S^2 \cdot \frac{(\sigma_{fore}^2 - \sigma_{imp}^2)}{252} \;-\; tc_{hedge/day} \;-\; tc_{opt/day}
\]

Where:
- `Γ` is from the package Greeks (already scaled by contracts)
- `S` is spot
- `tc_hedge/day` from the cost model
- `tc_opt/day` amortizes option spread costs over expected holding days

**Entry rule (V2)**
- Require `E[PnL/day] > edge_min_dollars_per_day`
- AND `liq_score > liq_min`
- AND `sigma_imp` not extreme vs historical percentile (optional guard)
- AND market is open / data fresh

**Exit rules (V2)**
Exit (close or wind down) if any:
- `E[PnL/day] < exit_edge_floor` for `N_exit_bars`
- `DTE < dte_min_to_hold` (avoid last-minute theta cliff if not intended)
- campaign PnL drawdown hits stop (risk guard)
- liquidity degrades (spreads widen above max)
- major event window ends (if event-driven campaign)

**Suggested file**
- `src/alpha/gates.py`  
  - `AlphaGate.should_enter(...)`
  - `AlphaGate.should_exit(...)`

---

### 6) Hedging Policy V2: Cost-aware MPC + no-trade region
Your current MPC (`src/control/mpc_controller.py`) penalizes delta exposure and trading, but V2 should add **explicit cadence/no-trade logic** so you don’t solve/trade every tick.

#### 6.1 Dynamic deadband (no-trade region)
Theory-guided rule of thumb (proportional costs): optimal delta policy often has a **no-trade band** whose width grows with costs and shrinks with gamma/vol.

V2 practical approximation:
- Compute a **delta deadband** in shares:
  - `band_base = config.delta_band_base`
  - `band_cost = k_band * (tc_share / (abs(gamma_opt) * S * sigma_fore^2 + eps))^(1/3)`
  - `delta_band = clip(max(band_base, band_cost), band_min, band_max)`

Then:
- Only hedge if `|net_delta| > delta_band`

Integrations:
- Use this to set:
  - `MPCConfig.direction_threshold` (dynamic)
  - `GammaScalpingConfig.min_trade_size` (dynamic)
  - or an outer “skip MPC solve” gate

#### 6.2 Hedge cadence controller
Add:
- `min_seconds_between_hedges`
- `min_price_move_to_hedge` (in $ or bps)
- `max_hedges_per_hour` (hard throttle)

This prevents “microstructure overfitting” and keeps costs bounded.

#### 6.3 MPC parameter scheduling
Make `w_delta`, `w_exec`, `c_lin` dynamic based on:
- liquidity regime (spread width)
- forecast sigma (expected movement)
- campaign state (ACTIVE vs WINDDOWN)

Example schedule:
- wide spreads ⇒ increase `c_lin`, increase `w_exec`, widen deadband
- high forecast sigma ⇒ allow more frequent hedging (tighten band modestly)

**Suggested file**
- `src/alpha/hedge_policy.py`  
  - `HedgeCadenceController`
  - `MPCScheduler` (updates MPCConfig per tick)

---

## Data Flow (V2)

```
Market tick (S, quotes, chain) 
   → Signals (sigma_fore, sigma_imp, liquidity) 
      → CampaignManager (OFF/ARMED/ACTIVE/WINDDOWN)
         → if ENTER: OptionSelector → open package (paper/live)
         → if ACTIVE: GreeksCalculator → net_delta, gamma, theta
             → HedgePolicy gate (no-trade band + cadence)
                 → MPCState → MPC solve → HedgeRecommendation
                     → Risk checks → Execution
         → PnL attribution + monitoring
```

---

## Configuration (proposed additions)

### `config/strategy_params.yaml` (V2 fields)
Add a `gamma_scalping_v2:` section:

- **Campaign**
  - `enable_campaign_manager: true`
  - `min_dte: 1`
  - `max_dte: 7`
  - `holding_days_target: 1`
  - `exit_edge_floor: 0.0`
  - `exit_confirm_bars: 5`

- **Signals**
  - `rv_ewma_lambda: 0.97`
  - `iv_smoothing_span: 10`
  - `data_stale_seconds: 5`

- **Entry thresholds**
  - `edge_min_dollars_per_day: 5.0`
  - `var_edge_min: 0.00`  (optional, use edge_min instead)
  - `liq_min: 0.7`
  - `iv_percentile_max: 0.9` (optional guard)

- **Option selection**
  - `candidate_expiries: [0, 1, 5]`   # DTE buckets
  - `max_strike_distance_pct: 0.5`    # ATM band
  - `min_open_interest: 1000`
  - `max_option_spread_pct: 0.10`

- **Costs**
  - `stock_half_spread: 0.005`
  - `stock_slippage_bps: 0.2`
  - `commission_per_share: 0.0`
  - `option_half_spread: 0.02`         # per option, dollars
  - `contract_multiplier: 100`

- **Hedging cadence**
  - `delta_band_base: 10.0`
  - `k_band: 1.0`
  - `band_min: 5.0`
  - `band_max: 100.0`
  - `min_seconds_between_hedges: 30`
  - `max_hedges_per_hour: 30`
  - `min_price_move_to_hedge_bps: 2.0`

- **Risk**
  - `max_contracts: 20`
  - `max_premium_dollars: 5000`
  - `max_daily_loss: 200`
  - `max_intraday_drawdown: 300`
  - `kill_switch_on_data_stale: true`

---

## PnL Attribution (required to iterate intelligently)
Add a PnL decomposition reported per interval and per day:

- `PnL_total`
- `PnL_gamma_capture` (approx)
- `PnL_theta`
- `PnL_vega` (IV changes)
- `PnL_delta_residual` (hedge error)
- `PnL_costs` (stock + option)

Minimum viable approximation:
- Theta from Greeks (`PortfolioGreeks.theta`) aggregated over time
- Costs from fills (or modeled if paper)
- Delta residual: mark-to-market of net delta between hedges

**Suggested file**
- `src/metrics/pnl_attribution.py`

---

## Testing Strategy

### Unit tests
- Signals:
  - EWMA RV reproduces known variance on synthetic returns
  - IV snapshot handles stale/missing data
- Cost model:
  - sanity checks for increasing costs with spreads/gamma/sigma
- Alpha gate:
  - enter/exit decisions match expectations on constructed scenarios
- Hedge cadence:
  - enforces throttles and no-trade band

### Integration tests (paper)
- Simulated price path + synthetic IV:
  - verify that entry triggers only when edge positive
  - verify hedge frequency decreases when spreads widen
- Replay small historical window (later):
  - confirm PnL attribution totals reconcile with portfolio MTM within tolerance

---

## Monitoring & Observability

**Campaign health**
- current state: OFF/ARMED/ACTIVE/WINDDOWN
- edge estimate: `E[PnL/day]`
- sigma_fore vs sigma_imp
- liquidity score

**Hedging behavior**
- hedges/hour
- average trade size
- cost per hedge
- net delta distribution (time in band vs outside band)

**Performance**
- daily PnL
- max intraday drawdown
- PnL attribution breakdown
- “gamma capture ratio”: gamma PnL / (theta + costs)

---

## Implementation Plan (checklist)

### Phase 1 — Make edge measurable
- [ ] Implement PnL attribution scaffold
- [ ] Add cost model + log realized costs

### Phase 2 — Add campaign + gates
- [ ] Implement RV forecast (EWMA) + IV snapshot
- [ ] Implement AlphaGate entry/exit rules
- [ ] Add CampaignManager state machine

### Phase 3 — Option selection
- [ ] Implement OptionSelector with scoring
- [ ] Integrate with data provider (chain + IV quotes)

### Phase 4 — Cost-aware hedging
- [ ] Implement no-trade band + cadence controller
- [ ] Add MPC scheduling (dynamic thresholds/weights)

### Phase 5 — Validate
- [ ] Paper trade loop + logs
- [ ] Backtest harness (even if approximate) + cost realism
- [ ] Iterate on thresholds using out-of-sample windows

---

## Open Questions
- What DTE regime is intended (0DTE intraday vs 1–7DTE swing)? The hedging + cost dynamics differ materially.
- Should we hedge vega (e.g., via calendars) in V2, or keep pure long gamma and accept vega exposure?
- What is the intended execution style (market vs limit, pegged, etc.) and how does that change the cost model?
- Should “event mode” be a separate gate (CPI/FOMC) with custom thresholds?

---

## References
- Repo docs:
  - `docs/decisions/002-greeks-mpc-integration.md`
  - `docs/design/mpc-controller-design-v1-revised.md`
- Concepts:
  - Volatility risk premium (IV vs RV)
  - No-trade region under proportional transaction costs (delta hedging with costs)
