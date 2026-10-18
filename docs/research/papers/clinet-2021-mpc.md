# Paper note: Clinet, Perreton, Reydellet (2021), "Optimal trading: a model predictive control approach"

arXiv:2110.11008v2 (PDF in this folder). This replaces an earlier summary whose sections "Horizon Selection" and "κ high → hedge less" were not in the paper; they are removed.

## What the paper is
An **optimal-execution** paper: a parent order must be completed over a fixed horizon (a completion constraint), and the question is the participation schedule. The model is dimensionless — slippage relative to a benchmark price and the fraction of the order still to execute — with a linear-quadratic cost and a mean-reverting price signal. The signal's decay κδt is a **hand-tuned constant**, explicitly not estimated in the paper. The LQR solution can go negative (round trips), so the authors use **MPC**: at each step solve a certainty-equivalent QP over the *remaining* horizon with the positivity/completion constraints, apply the first action, re-solve next step. That receding-horizon-with-constraints pattern is the paper's contribution for this project.

## What this project took from it
- The **QP-per-step pattern**: a convex program over the remaining horizon, first action applied, re-solved next tick. `src/hedge.py::qp_half_width` is exactly that program's first action for the delta-hedging objective (risk λ·Σσ_k²S²dt·D_k² + asymmetric linear costs + overnight terminal weight, N = steps to the session close), which has a closed form because E[dS] = 0 makes the predicted exposure path flat. `tests/test_hedge.py::test_qp_closed_form_equals_program` keeps the full N-step program as a scipy oracle; no solver ships (ADR-004).
- The **unit discipline**: dimensionless cost ratios (band `k` = $/share ÷ S), explicit dt per step, risk and cost in the same currency.

## What this project did not take
- No κ, no reference price S̄, no mean-reversion signal: a delta hedge has no completion constraint and no directional view; E[dS] = 0 is the assumption.
- No "aggressive in the money" behaviour: that needs the signal.
- No MHE / online parameter estimation: the only estimated quantity is realized variance (`src/vol.py`), and the gate uses it, not the hedger.
- No horizon tuning: N is the number of ticks to the close, which leaves one free parameter (λ) per policy.

## What the hedger actually rests on
- **Whalley & Wilmott (1997)**, *An asymptotic analysis of an optimal hedging model for option pricing with transaction costs*, Mathematical Finance 7(3): the no-trade band H = ((3/2)·e^{−rT}·k·S·Γ²/γ)^{1/3} that `band_half_width` implements (H grows with Γ^{2/3}, shrinks with γ^{1/3}).
- **Zakamouline (2006)**, *European option pricing and hedging with both fixed and proportional transaction costs*, JEDC 30: why a per-order minimum (IBKR's $0.35) widens the band; implemented as one fixed-point step that folds the minimum into `k` at the expected trade size.
- **Derman & Kamal (1999)**, *When you cannot hedge continuously*, Risk 12(1): the discrete-hedging error std ≈ premium·√(π/4N), asserted by the Monte-Carlo oracle `derman_kamal` in `tests/test_backtest_sim.py`.
- **Muravyev & Ni (2020)**, *Why do option returns change sign from day to night?*, JFE: overnight gap variance ∝ Δ², hence the hedge-to-centre rule in the last `loop.close_window_s` of the session.
