"""Hedge policies: Whalley–Wilmott band (production), the receding-horizon QP's closed-form first
action (research, shadow-logged), and the naive -Δ baseline. `hedge_all` returns all three per
tick; the runner logs them and acts on `cfg.policy`. Pure functions, no state, no solver.

Units (canonical table in src/book.py): delta_net shares (signed, options + shares); gamma
shares/$ (signed); S $; T calendar years; seconds_to_close s; W_var years·σ² (vol.horizon_variance
= Σσ_k²·dt_y + overnight term); band k dimensionless ($/share ÷ S); risk aversions 1/$; trades are
signed shares (< 0 = sell). Maths: ADR-004 §2 — implement, do not re-derive.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.costs import per_share_rate

if TYPE_CHECKING:
    from src.config import Costs, Hedge


@dataclass(frozen=True)
class HedgeState:
    delta_net: float
    gamma: float
    S: float
    T: float
    seconds_to_close: float
    W_var: float | None


@dataclass(frozen=True)
class HedgeDecision:
    band: int
    qp: int
    naive: int
    chosen: int
    band_h: float
    qp_h: float
    reason: str


def band_half_width(
    S: float, gamma: float, T: float, r: float, k: float, risk_aversion: float
) -> float:
    """Whalley–Wilmott half-width, shares: ((3/2)·e^{-rT}·k·S·Γ²/γ)^{1/3}."""
    return (1.5 * math.exp(-r * T) * k * S * gamma**2 / risk_aversion) ** (1 / 3)


def band_k(S: float, side: int, c: Costs, h_prev: float | None, inner_fraction: float) -> float:
    """Dimensionless cost; given h_prev, min_commission is amortised over the expected trade (1-f)·h_prev."""
    rate = per_share_rate(S, side, c)
    if h_prev is not None:
        rate += c.min_commission / max((1 - inner_fraction) * h_prev, 1.0)
    return rate / S


def qp_half_width(c_side: float, S: float, risk_aversion: float, W_var: float) -> float:
    """Soft threshold of the receding-horizon QP (= its first action), shares: c/(2·λ·S²·W_var)."""
    return c_side / (2 * risk_aversion * S**2 * W_var)


def _trade(delta_net: float, h: float, f: float) -> int:
    """0 inside ±h, else the trade to the interior target sign(Δ)·f·h."""
    return 0 if abs(delta_net) <= h else int(round(math.copysign(f * h, delta_net) - delta_net))


def hedge_all(
    st: HedgeState, cfg: Hedge, c: Costs, r: float, close_window_s: float
) -> HedgeDecision:
    d, S = st.delta_net, st.S
    if not all(map(math.isfinite, (d, st.gamma, S, st.T, st.seconds_to_close))) or S <= 0:
        return HedgeDecision(0, 0, 0, 0, math.nan, math.nan, "nan")
    side = -1 if d > 0 else 1  # selling reduces a positive delta
    c_side = per_share_rate(S, side, c)
    f, g = cfg.inner_fraction, cfg.band_risk_aversion
    h0 = band_half_width(S, st.gamma, st.T, r, band_k(S, side, c, None, f), g)
    h1 = band_half_width(S, st.gamma, st.T, r, band_k(S, side, c, h0, f), g)  # one fixed-point step
    band_h = min(max(h1, cfg.band_min_shares), cfg.band_max_shares)  # gamma == 0 -> band_min
    has_vol = st.W_var is not None and 0 < st.W_var < math.inf
    qp_h = qp_half_width(c_side, S, cfg.qp_risk_aversion, st.W_var) if has_vol else math.nan
    band = _trade(d, band_h, f)
    qp = _trade(d, qp_h, 1.0) if has_vol else 0
    naive = -int(round(d)) if abs(d) >= 1 else 0
    reason = ""
    if st.seconds_to_close <= close_window_s:
        band = qp = naive = -int(round(d))
        reason = "close_hedge"
    m = cfg.max_trade_shares
    out = []
    for u in (band, qp, naive):
        u = max(-m, min(m, u))
        if abs(d + u) > abs(d) + 1e-9:  # bug guard: interior targets never flip sign
            u, reason = 0, "increases_exposure"
        out.append(u)
    band, qp, naive = out
    chosen, h_policy = {"band": (band, band_h), "qp": (qp, qp_h), "naive": (naive, 1.0)}[cfg.policy]
    if not reason:
        if cfg.policy == "qp" and not has_vol:
            reason = "no_vol"
        else:
            reason = "in_band" if chosen == 0 and abs(d) <= h_policy else "hedge"
    return HedgeDecision(band, qp, naive, chosen, band_h, qp_h, reason)
