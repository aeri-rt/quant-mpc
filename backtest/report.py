"""
Backtest reporting -- plots and console summary.

Reads a ``BacktestResult`` and produces:
1. Equity curve (cumulative P&L)
2. P&L decomposition (gamma P&L, theta decay, costs)
3. Delta exposure time series
4. Spot price with hedge trade markers
5. Drawdown chart

Plots are saved as PNGs to the configured output directory.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# matplotlib is a dev dependency; import at function level to keep
# module importable even when matplotlib isn't installed.


from backtest.metrics import BacktestMetrics, BacktestResult
from backtest.pnl_engine import PnLEngine


class BacktestReport:
    """Generate plots and console summary from backtest results."""

    def __init__(self, result: BacktestResult):
        self._result = result
        self._metrics = BacktestMetrics()
        self._metrics._records = list(result.records)
        self._pnl: PnLEngine = result.pnl_engine  # type: ignore[assignment]
        self._output_dir = Path(result.config.output_dir)  # type: ignore[union-attr]
        self._output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Console summary
    # ------------------------------------------------------------------

    def print_summary(self) -> str:
        """Print and return the console summary string."""
        cfg = self._result.config
        m = self._metrics
        decomp = self._pnl.get_decomposition()

        start = self._result.bars.index[0] if len(self._result.bars) > 0 else "N/A"
        end = self._result.bars.index[-1] if len(self._result.bars) > 0 else "N/A"

        # Estimate trading days
        if len(self._result.bars) > 0:
            unique_days = self._result.bars.index.normalize().nunique()
        else:
            unique_days = 0

        initial_cap = getattr(cfg, "initial_capital", 100_000)

        lines = [
            "",
            "=" * 68,
            f"  Backtest Results: {start} to {end}",
            "=" * 68,
            f"  Duration: {unique_days} trading days | "
            f"Bars: {self._result.total_bars:,} "
            f"(warmup: {self._result.warmup_bars})",
            f"  Total P&L: ${m.total_pnl():,.2f} | "
            f"Sharpe: {m.sharpe_ratio():.2f} | "
            f"Max DD: ${m.max_drawdown():,.2f} "
            f"({m.max_drawdown_pct(initial_cap):.1%})",
            f"  Equity hedge P&L: ${decomp['equity_pnl']:,.2f} | "
            f"Option P&L: ${decomp['option_pnl']:,.2f} | "
            f"Costs: ${decomp['costs']:,.2f}",
            f"  Gamma P&L: ${decomp['gamma_pnl']:,.2f} | "
            f"Theta P&L: ${decomp['theta_pnl']:,.2f}",
            f"  Trades: {m.total_trades()} | "
            f"Avg size: {m.avg_trade_size():.0f} shares | "
            f"Rolls: {self._result.num_rolls}",
            f"  Avg |delta|: {m.avg_abs_delta():.1f} | "
            f"Max |delta|: {m.max_abs_delta():.1f} | "
            f"% time hedged (<50 delta): {m.pct_time_hedged():.1f}%",
            "=" * 68,
            "",
        ]
        summary = "\n".join(lines)
        print(summary)
        return summary

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------

    def generate_plots(self) -> None:
        """Generate and save all plots."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        df = self._metrics.to_dataframe()
        if df.empty:
            print("No bar records to plot.")
            return

        # Common date formatter
        date_fmt = mdates.DateFormatter("%m-%d %H:%M")

        # -- 1. Equity curve ------------------------------------------
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(df.index, df["total_pnl"], linewidth=1.0, color="#2196F3")
        ax.axhline(0, color="grey", linewidth=0.5, linestyle="--")
        ax.set_title("Equity Curve (Cumulative P&L)")
        ax.set_ylabel("P&L ($)")
        ax.xaxis.set_major_formatter(date_fmt)
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(self._output_dir / "equity_curve.png", dpi=150)
        plt.close(fig)

        # -- 2. P&L decomposition (stacked area) ----------------------
        decomp = self._pnl.get_decomposition()
        fig, ax = plt.subplots(figsize=(14, 5))
        # Build cumulative series from bar records
        equity_curve = self._pnl.get_equity_curve()
        if not equity_curve.empty:
            ax.fill_between(
                equity_curve.index, 0, equity_curve.values,
                alpha=0.3, label="Total P&L", color="#4CAF50",
            )
            ax.plot(equity_curve.index, equity_curve.values, linewidth=0.8, color="#4CAF50")
        ax.set_title("P&L Decomposition")
        ax.set_ylabel("P&L ($)")
        ax.legend()
        ax.xaxis.set_major_formatter(date_fmt)
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(self._output_dir / "pnl_decomposition.png", dpi=150)
        plt.close(fig)

        # -- 3. Delta exposure -----------------------------------------
        fig, ax = plt.subplots(figsize=(14, 4))
        ax.plot(df.index, df["net_delta"], linewidth=0.7, color="#FF9800")
        ax.axhline(0, color="grey", linewidth=0.5, linestyle="--")
        ax.set_title("Net Portfolio Delta Exposure")
        ax.set_ylabel("Delta (share equiv.)")
        ax.xaxis.set_major_formatter(date_fmt)
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(self._output_dir / "delta_exposure.png", dpi=150)
        plt.close(fig)

        # -- 4. Spot price with trade markers --------------------------
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(df.index, df["spot"], linewidth=0.6, color="#607D8B", label="SPY")

        buys = df[df["trade_direction"] == "BUY"]
        sells = df[df["trade_direction"] == "SELL"]
        if not buys.empty:
            ax.scatter(buys.index, buys["spot"], marker="^", color="#4CAF50",
                       s=12, alpha=0.6, label="Buy")
        if not sells.empty:
            ax.scatter(sells.index, sells["spot"], marker="v", color="#F44336",
                       s=12, alpha=0.6, label="Sell")

        ax.set_title("SPY Price with Hedge Trades")
        ax.set_ylabel("Price ($)")
        ax.legend(loc="upper left", fontsize=8)
        ax.xaxis.set_major_formatter(date_fmt)
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(self._output_dir / "spot_with_trades.png", dpi=150)
        plt.close(fig)

        # -- 5. Drawdown chart -----------------------------------------
        fig, ax = plt.subplots(figsize=(14, 4))
        pnls = df["total_pnl"].values
        running_max = np.maximum.accumulate(pnls)
        drawdown = running_max - pnls
        ax.fill_between(df.index, 0, drawdown, alpha=0.4, color="#F44336")
        ax.set_title("Drawdown")
        ax.set_ylabel("Drawdown ($)")
        ax.xaxis.set_major_formatter(date_fmt)
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(self._output_dir / "drawdown.png", dpi=150)
        plt.close(fig)

        print(f"Plots saved to {self._output_dir}/")
