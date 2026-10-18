"""Backtest report: fixed-width tables, three PNGs, CSVs — all read from `backtest.sim.Result`
objects (hedger, risk_aversion, decisions, fills, daily, summary). Daily P&L only: no Sharpe on
intraday bars. matplotlib is imported inside `plots`.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from backtest.sim import Result

ADR_RULE = (
    "ADR rule: qp replaces band only if its (hedge friction/d, hedge-error std) frontier lies "
    "below the band's in two disjoint real-bar windows, p<0.05 on paired daily differences"
)
# Fixed colour per entity (never cycled): hedgers and attribution components.
COLORS = {"band": "#2a78d6", "qp": "#eb6834", "naive": "#1baf7a"}
_ATTR = {"gamma": "#2a78d6", "theta": "#eb6834", "vega": "#1baf7a", "costs": "#eda100"}
_DAILY_SUM = (
    ("delta", ".0f"),
    ("gamma", ".0f"),
    ("theta", ".0f"),
    ("vega", ".0f"),
    ("hedge_spread", ".2f"),
    ("commissions", ".2f"),
    ("residual", ".0f"),
    ("pnl", ".0f"),
    ("trades", ".0f"),
    ("shares", ".0f"),
)
_DAILY_MEAN = (("mean_abs_delta", ".1f"), ("sigma_f", ".3f"), ("sigma_iv", ".3f"))


def _f(v, spec: str = ".2f") -> str:
    return "-" if v is None or pd.isna(v) else format(v, spec)


def _pct(v) -> str:
    return "-" if v is None or pd.isna(v) else f"{100 * float(v):.0f}%"


def _table(header: list[str], rows: list[list[str]]) -> str:
    w = [max(len(c) for c in col) for col in zip(header, *rows, strict=True)]

    def line(r):
        cells = zip(r, w, strict=True)
        return "  ".join(c.ljust(n) if i == 0 else c.rjust(n) for i, (c, n) in enumerate(cells))

    return "\n".join([line(header), "  ".join("-" * n for n in w), *map(line, rows)])


def summary_table(results: dict[str, Result], gate_turnover: float | None = None) -> str:
    """One row per hedger. `turnover` = realised/configured (gate.turnover_frac) so the gap shows."""
    header = ["hedger", "γ/λ", "P&L", "days", "std/day", "Sharpe", "maxDD", "trades/d", "cost/d"]
    header += ["mean|Δ|", "turnover", "gamma", "theta", "gate%"]
    rows = []
    for r in results.values():
        s = r.summary
        turn = _f(s.get("turnover_frac_realised"))
        if gate_turnover is not None:
            turn += f"/{gate_turnover:.2f}"
        rows.append(
            [
                r.hedger,
                _f(r.risk_aversion, ".1e"),
                _f(s.get("total_pnl"), ".0f"),
                _f(s.get("days"), ".0f"),
                _f(s.get("daily_std"), ".0f"),
                _f(s.get("sharpe")),
                _f(s.get("max_drawdown"), ".0f"),
                _f(s.get("trades_per_day"), ".1f"),
                _f(s.get("hedge_cost_per_day")),
                _f(s.get("mean_abs_delta"), ".1f"),
                turn,
                _f(s.get("gamma_pnl"), ".0f"),
                _f(s.get("theta_pnl"), ".0f"),
                _pct(s.get("gate_open_frac")),
            ]
        )
    return _table(header, rows)


def attribution_table(r: Result, last_n: int | None = None) -> str:
    """Per-session attribution + TOTAL (sums; means for mean|Δ|, σ_f, σ_iv; gate = % open)."""
    d = r.daily if last_n is None else r.daily.tail(last_n)
    header = ["date", *(k for k, _ in _DAILY_SUM), "mean|Δ|", "σ_f", "σ_iv", "gate"]
    rows = [
        [
            str(row["date"])[:10],
            *(_f(row[k], f) for k, f in _DAILY_SUM),
            *(_f(row[k], f) for k, f in _DAILY_MEAN),
            _pct(row["gate_open"]),
        ]
        for _, row in d.iterrows()
    ]
    total = [
        "TOTAL",
        *(_f(d[k].sum(), f) for k, f in _DAILY_SUM),
        *(_f(d[k].mean(), f) for k, f in _DAILY_MEAN),
        _pct(d["gate_open"].astype(float).mean()),
    ]
    return _table(header, [*rows, total])


def _gamma_theta(s: dict) -> float | None:
    g, t = s.get("gamma_pnl"), s.get("theta_pnl")
    return None if not t or g is None else g / abs(t)


def _residual_share(d: pd.DataFrame) -> float | None:
    """Σ|residual| over the gross absolute attribution (how much P&L the model does not explain)."""
    parts = d[["delta", "gamma", "theta", "vega", "hedge_spread", "commissions", "residual"]]
    gross = parts.abs().to_numpy().sum()
    return None if gross == 0 else d["residual"].abs().sum() / gross


def comparison_table(results: dict[str, Result]) -> str:
    """Hedgers side by side on the same bars; raises if the gate decisions differ between them."""
    gates = [r.daily["gate_open"].to_numpy() for r in results.values()]
    if any(not np.array_equal(g, gates[0]) for g in gates[1:]):
        raise ValueError(
            "gate_open differs between hedgers: the gate must not depend on the hedger"
        )
    metrics = [
        ("total P&L", lambda r: _f(r.summary.get("total_pnl"), ".0f")),
        ("std daily P&L", lambda r: _f(r.summary.get("daily_std"), ".0f")),
        ("hedge cost/day", lambda r: _f(r.summary.get("hedge_cost_per_day"))),
        ("hedge friction/day", lambda r: _f(r.summary.get("hedge_friction_per_day"))),
        ("std hedge-err P&L", lambda r: _f(r.summary.get("hedge_error_std"), ".0f")),
        ("trades/day", lambda r: _f(r.summary.get("trades_per_day"), ".1f")),
        ("mean |Δ|", lambda r: _f(r.summary.get("mean_abs_delta"), ".1f")),
        ("gamma/theta", lambda r: _f(_gamma_theta(r.summary))),
        ("residual share", lambda r: _pct(_residual_share(r.daily))),
    ]
    rows = [[name, *(fn(r) for r in results.values())] for name, fn in metrics]
    return _table(["metric", *results], rows)


def frontier_table(sweep: dict[str, list[Result]]) -> str:
    """(hedge friction/day, hedge-error std) per risk aversion for band and qp, sorted by
    risk_aversion; ends with the ADR rule."""
    rows = []
    for policy in ("band", "qp"):
        for r in sorted(sweep.get(policy, []), key=lambda r: r.risk_aversion):
            s = r.summary
            rows.append(
                [
                    policy,
                    _f(r.risk_aversion, ".1e"),
                    _f(s.get("hedge_friction_per_day")),
                    _f(s.get("hedge_error_std"), ".0f"),
                    _f(s.get("trades_per_day"), ".1f"),
                    _f(s.get("total_pnl"), ".0f"),
                ]
            )
    header = ["policy", "risk_aversion", "friction/d", "err std/d", "trades/d", "P&L"]
    return _table(header, rows) + "\n" + ADR_RULE


def plots(results: dict[str, Result], out_dir: Path) -> list[Path]:
    """equity.png (all hedgers), attribution.png and delta.png (first hedger)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    first = next(iter(results.values()))
    paths = []

    def save(fig, name):
        fig.autofmt_xdate()
        fig.tight_layout()
        p = out_dir / name
        fig.savefig(p, dpi=110)
        plt.close(fig)
        paths.append(p)

    fig, ax = plt.subplots(figsize=(10, 5))
    for name, r in results.items():
        x = pd.to_datetime(r.daily["date"])
        ax.plot(x, r.daily["pnl"].cumsum(), lw=2, color=COLORS.get(name), label=name)
    ax.axhline(0, color="#c3c2b7", lw=1)
    ax.set(title="Cumulative daily P&L", ylabel="$")
    ax.legend(frameon=False)
    save(fig, "equity.png")

    d = first.daily
    x = pd.to_datetime(d["date"])
    parts = {
        "gamma": d["gamma"],
        "theta": d["theta"],
        "vega": d["vega"],
        "costs": -(d["hedge_spread"] + d["commissions"]),
    }
    fig, ax = plt.subplots(figsize=(10, 5))
    pos, neg = np.zeros(len(d)), np.zeros(len(d))
    for k, v in parts.items():
        v = v.to_numpy(float)
        ax.bar(x, v, bottom=np.where(v >= 0, pos, neg), width=0.8, color=_ATTR[k], label=k)
        pos += np.clip(v, 0, None)
        neg += np.clip(v, None, 0)
    ax.axhline(0, color="#c3c2b7", lw=1)
    ax.set(title=f"Daily P&L attribution ({first.hedger})", ylabel="$")
    ax.legend(frameon=False, ncol=4)
    save(fig, "attribution.png")

    dec = first.decisions
    t = pd.to_datetime(dec["ts"], utc=True)
    same_session = t.diff() < pd.Timedelta(
        hours=1
    )  # NaN at each session's first tick: no overnight line
    h = pd.to_numeric(dec["band_h"], errors="coerce").where(same_session)
    dn = pd.to_numeric(dec["delta_net"], errors="coerce").abs().where(same_session)
    c = COLORS.get(first.hedger, "#2a78d6")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.fill_between(t, 0, h, color=c, alpha=0.15, lw=0, label="band H")
    ax.plot(t, dn, lw=1, color=c, label="|Δ net|")
    ax.set(title=f"|net delta| vs band half-width ({first.hedger})", ylabel="shares")
    ax.legend(frameon=False)
    save(fig, "delta.png")
    return paths


def write_csvs(results: dict[str, Result], out_dir: Path) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, r in results.items():
        for kind, df in (("daily", r.daily), ("decisions", r.decisions)):
            p = out_dir / f"{kind}_{name}.csv"
            df.to_csv(p, index=False)
            paths.append(p)
    return paths
