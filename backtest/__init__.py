"""
Backtest package for the quant-mpc gamma scalping system.

Provides offline simulation of the GammaScalpingStrategy + DeltaHedgeMPC
over historical data with VIX-calibrated synthetic options and full
P&L accounting.
"""

from backtest.config import BacktestConfig
from backtest.engine import BacktestEngine
from backtest.metrics import BacktestResult

__all__ = [
    "BacktestConfig",
    "BacktestEngine",
    "BacktestResult",
]
