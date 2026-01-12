"""
Utility modules for Quant-MPC.

Contains:
- Structured logging system
- Timer utilities
- Test helpers
"""

from .logging import (
    # Setup
    setup_logging,
    LogConfig,
    get_logger,
    
    # Structured logging
    log_trade,
    log_mpc_solve,
    log_greeks,
    log_metrics,
    
    # Log records
    TradeLog,
    MPCSolveLog,
    GreeksLog,
    MetricsLog,
    
    # Utilities
    log_context,
    Timer,
    TestLogger,
    read_logs,
)

__all__ = [
    # Setup
    'setup_logging',
    'LogConfig',
    'get_logger',
    
    # Structured logging
    'log_trade',
    'log_mpc_solve',
    'log_greeks',
    'log_metrics',
    
    # Log records
    'TradeLog',
    'MPCSolveLog',
    'GreeksLog',
    'MetricsLog',
    
    # Utilities
    'log_context',
    'Timer',
    'TestLogger',
    'read_logs',
]