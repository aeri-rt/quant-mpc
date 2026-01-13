"""
Utility modules for Quant-MPC.

Contains:
- Structured logging system
- Configuration loader
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

from .config import (
    # Main loader
    load_config,
    AppConfig,
    
    # Typed configs
    BrokerConfig,
    InstrumentConfig,
    MPCParams,
    GammaScalpingParams,
    RiskParams,
    
    # Convenience functions
    get_settings,
    get_mpc_params,
    get_strategy_params,
    get_broker_config,
    get_mpc_config,
    get_strategy_config,
    get_risk_config,
    get_instrument_config,
)

__all__ = [
    # Logging - Setup
    'setup_logging',
    'LogConfig',
    'get_logger',
    
    # Logging - Structured
    'log_trade',
    'log_mpc_solve',
    'log_greeks',
    'log_metrics',
    
    # Logging - Records
    'TradeLog',
    'MPCSolveLog',
    'GreeksLog',
    'MetricsLog',
    
    # Logging - Utilities
    'log_context',
    'Timer',
    'TestLogger',
    'read_logs',
    
    # Config - Main
    'load_config',
    'AppConfig',
    
    # Config - Types
    'BrokerConfig',
    'InstrumentConfig',
    'MPCParams',
    'GammaScalpingParams',
    'RiskParams',
    
    # Config - Convenience
    'get_settings',
    'get_mpc_params',
    'get_strategy_params',
    'get_broker_config',
    'get_mpc_config',
    'get_strategy_config',
    'get_risk_config',
    'get_instrument_config',
]

from .metrics import (
    MetricsTracker,
    TradeRecord,
    GreeksSnapshot,
    get_metrics_tracker,
    reset_metrics_tracker,
)

# Add to __all__
__all__.extend([
    'MetricsTracker',
    'TradeRecord',
    'GreeksSnapshot',
    'get_metrics_tracker',
    'reset_metrics_tracker',
])
