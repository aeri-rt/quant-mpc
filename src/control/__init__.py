"""
Control module for MPC-based trading.

Contains:
- DeltaHedgeMPC: Main MPC controller for delta hedging
- MPCConfig: Configuration dataclass
- MPCState: State input dataclass
- MPCSolution: Solution output dataclass
- SimpleKappaEstimator: v1 parameter estimator
"""

from .mpc_controller import (
    DeltaHedgeMPC,
    MPCConfig,
    MPCState,
    MPCSolution,
    SimpleKappaEstimator,
)

__all__ = [
    'DeltaHedgeMPC',
    'MPCConfig',
    'MPCState',
    'MPCSolution',
    'SimpleKappaEstimator',
]