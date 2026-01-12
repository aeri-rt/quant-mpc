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

__all__ = [
    'GammaScalpingStrategy',
    'GammaScalpingConfig',
    'HedgeRecommendation',
]