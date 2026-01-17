"""
Strategy module for Quant-MPC.

Contains strategy implementations that combine:
- Models (Greeks calculation)
- Control (MPC optimization)
- Risk management

Current Strategies:
- GammaScalpingStrategy: MPC-based delta hedging for long gamma positions
"""

from .gamma_scalping import (
    GammaScalpingStrategy,
    GammaScalpingConfig,
    HedgeRecommendation,
)

from .position_manager import (
    PositionManager,
    PositionManagerConfig,
    PositionAction,
    PositionSignal,
)

__all__ = [
    'GammaScalpingStrategy',
    'GammaScalpingConfig',
    'HedgeRecommendation',
    'PositionManager',
    'PositionManagerConfig',
    'PositionAction',
    'PositionSignal',
]