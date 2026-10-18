"""
Backtest configuration.

Defines all parameters for running a historical backtest of the
gamma scalping strategy.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class BacktestConfig:
    """Configuration for the backtesting engine."""

    # Time range
    start_date: str = "2025-12-01"
    end_date: str = "2026-01-31"
    bar_size_minutes: int = 5

    # Warmup (SimpleKappaEstimator needs ~60 bars)
    warmup_bars: int = 60

    # Capital and sizing
    initial_capital: float = 100_000.0
    position_size: int = 10       # Contracts per straddle leg
    target_dte: int = 30          # Target DTE for new straddles

    # Position management
    roll_dte: int = 7             # Roll when DTE falls below
    roll_to_dte: int = 30         # Roll into this DTE
    take_profit_pct: float = 0.50
    stop_loss_pct: float = -0.30

    # Volatility model
    vix_scale_factor: float = 1.0  # Multiplier on VIX -> BS sigma

    # Market parameters
    risk_free_rate: float = 0.045
    dividend_yield: float = 0.0106

    # Transaction costs
    equity_slippage_bps: float = 1.0
    equity_commission_per_share: float = 0.005
    option_spread_dollars: float = 0.05   # Half-spread for ATM SPY options
    option_commission_per_contract: float = 0.65

    # MPC config overrides
    mpc_horizon: int = 20
    mpc_q_max: float = 1000.0
    mpc_u_max: float = 200.0
    mpc_w_delta: float = 10.0

    # Data paths
    cache_dir: str = "data/cache"
    output_dir: str = "data/backtest_results"

    @classmethod
    def from_yaml(cls, path: str) -> "BacktestConfig":
        """Load config from a YAML file, merging with defaults."""
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        # Flatten nested sections into a flat dict of constructor args
        flat: dict = {}
        for key, value in raw.items():
            if isinstance(value, dict):
                flat.update(value)
            else:
                flat[key] = value
        # Only keep keys that are valid fields
        valid = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in flat.items() if k in valid}
        return cls(**filtered)

    def ensure_dirs(self) -> None:
        """Create cache and output directories if they don't exist."""
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
