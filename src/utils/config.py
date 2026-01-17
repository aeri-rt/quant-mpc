"""
Configuration Loader for Quant-MPC.

Loads YAML configuration files and converts them to typed configuration objects
for use throughout the system.

Usage:
    from src.utils.config import load_config, get_settings, get_mpc_config
    
    # Load all configs
    config = load_config()
    
    # Or load specific configs
    settings = get_settings()
    mpc_config = get_mpc_config()
    strategy_config = get_strategy_config()
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional
import yaml

from src.utils.logging import get_logger

logger = get_logger("config")


# =============================================================================
# Configuration Models
# =============================================================================

@dataclass
class BrokerConfig:
    """Broker connection configuration."""
    name: str = "ibkr"
    host: str = "127.0.0.1"
    port: int = 7497
    client_id: int = 1
    timeout: int = 30
    readonly: bool = False


@dataclass
class InstrumentConfig:
    """Target instrument configuration."""
    symbol: str = "SPY"
    exchange: str = "SMART"
    currency: str = "USD"
    sec_type: str = "STK"
    dividend_yield: float = 0.0106


@dataclass
class MPCWeights:
    """MPC objective function weights."""
    execution_cost: float = 1.0
    delta_deviation: float = 10.0
    delta_rate: float = 0.1
    terminal_delta: float = 100.0


@dataclass
class MPCCosts:
    """MPC transaction cost model."""
    c_quad: float = 0.001  # Quadratic cost per share² (market impact)
    c_lin: float = 0.01    # Linear cost per share (spread + commission)


@dataclass
class MPCConstraints:
    """MPC constraints."""
    max_position: int = 1000
    min_position: int = -1000
    max_trade_rate: int = 200
    max_delta: int = 500
    min_delta: int = -500


@dataclass
class MPCParams:
    """MPC controller parameters."""
    dt_seconds: float = 60.0
    horizon: int = 20
    solver_name: str = "ipopt"
    solver_max_iter: int = 100
    solver_tol: float = 1e-6
    weights: MPCWeights = field(default_factory=MPCWeights)
    constraints: MPCConstraints = field(default_factory=MPCConstraints)
    costs: MPCCosts = field(default_factory=MPCCosts)
    
    # Mean reversion
    kappa_init: float = 0.1
    kappa_min: float = 0.0
    kappa_max: float = 1.0
    ema_span: int = 20
    
    @property
    def dt(self) -> float:
        """Time step in years."""
        # Convert seconds to years (assuming 252 trading days, 6.5 hours/day)
        return self.dt_seconds / (252 * 6.5 * 3600)


@dataclass
class InstrumentSelection:
    """Option instrument selection parameters."""
    structure: str = "straddle"
    strangle_width: float = 0.02
    strike_selection: str = "atm"
    target_dte: int = 30
    min_dte: int = 21
    max_dte: int = 45
    roll_dte: int = 7
    roll_to_dte: int = 30


@dataclass
class HedgingConfig:
    """Delta hedging configuration."""
    hedge_instrument: str = "equity"
    max_delta_exposure: int = 100
    min_hedge_size: int = 10


@dataclass
class GammaScalpingParams:
    """Gamma scalping strategy parameters."""
    enabled: bool = True
    instrument_selection: InstrumentSelection = field(default_factory=InstrumentSelection)
    
    # Position sizing
    sizing_method: str = "fixed"
    fixed_contracts: int = 10
    
    # Entry/exit
    min_vol_premium: float = 0.01
    take_profit: float = 0.20
    stop_loss: float = -0.30
    
    # Hedging
    hedging: HedgingConfig = field(default_factory=HedgingConfig)


@dataclass
class RiskParams:
    """Risk management parameters."""
    max_contracts: int = 50
    max_notional: float = 100000.0
    max_loss_daily: float = 2000.0
    max_delta: float = 500.0
    max_gamma: float = 100.0
    max_vega: float = 5000.0
    max_theta: float = -500.0
    margin_buffer: float = 0.30


@dataclass
class AppConfig:
    """Complete application configuration."""
    broker: BrokerConfig = field(default_factory=BrokerConfig)
    instrument: InstrumentConfig = field(default_factory=InstrumentConfig)
    mpc: MPCParams = field(default_factory=MPCParams)
    strategy: GammaScalpingParams = field(default_factory=GammaScalpingParams)
    risk: RiskParams = field(default_factory=RiskParams)
    mode: str = "paper"


# =============================================================================
# Config Loading
# =============================================================================

_config_cache: Optional[AppConfig] = None
_config_dir: Path = Path(__file__).parent.parent.parent / "config"


def _load_yaml(filename: str) -> Dict[str, Any]:
    """Load a YAML file from the config directory."""
    path = _config_dir / filename
    
    if not path.exists():
        logger.warning(f"Config file not found: {path}")
        return {}
    
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def load_config(config_dir: Optional[Path] = None, reload: bool = False) -> AppConfig:
    """
    Load all configuration files and return a unified AppConfig.
    
    Args:
        config_dir: Optional custom config directory
        reload: If True, reload even if cached
        
    Returns:
        AppConfig with all settings
    """
    global _config_cache, _config_dir
    
    if _config_cache is not None and not reload:
        return _config_cache
    
    if config_dir:
        _config_dir = config_dir
    
    # Load individual config files
    settings = _load_yaml("settings.yaml")
    mpc_params = _load_yaml("mpc_params.yaml")
    strategy_params = _load_yaml("strategy_params.yaml")
    
    # Build config objects
    broker_data = settings.get("broker", {})
    broker = BrokerConfig(
        name=broker_data.get("name", "ibkr"),
        host=broker_data.get("host", "127.0.0.1"),
        port=broker_data.get("port", 7497),
        client_id=broker_data.get("client_id", 1),
        timeout=broker_data.get("timeout", 30),
        readonly=broker_data.get("readonly", False),
    )
    
    instrument_data = settings.get("instrument", {})
    instrument = InstrumentConfig(
        symbol=instrument_data.get("symbol", "SPY"),
        exchange=instrument_data.get("exchange", "SMART"),
        currency=instrument_data.get("currency", "USD"),
        sec_type=instrument_data.get("sec_type", "STK"),
        dividend_yield=instrument_data.get("dividend_yield", 0.0106),
    )
    
    # MPC params
    mpc_section = mpc_params.get("mpc", {})
    weights_data = mpc_section.get("weights", {})
    constraints_data = mpc_section.get("constraints", {})
    costs_data = mpc_section.get("costs", {})
    mean_rev_data = mpc_params.get("mean_reversion", {})
    
    mpc = MPCParams(
        dt_seconds=mpc_params.get("time", {}).get("dt", 60.0),
        horizon=mpc_section.get("horizon", 20),
        solver_name=mpc_section.get("solver", {}).get("name", "ipopt"),
        solver_max_iter=mpc_section.get("solver", {}).get("max_iter", 100),
        solver_tol=mpc_section.get("solver", {}).get("tol", 1e-6),
        weights=MPCWeights(
            execution_cost=weights_data.get("execution_cost", 1.0),
            delta_deviation=weights_data.get("delta_deviation", 10.0),
            delta_rate=weights_data.get("delta_rate", 0.1),
            terminal_delta=weights_data.get("terminal_delta", 100.0),
        ),
        constraints=MPCConstraints(
            max_position=constraints_data.get("max_position", 1000),
            min_position=constraints_data.get("min_position", -1000),
            max_trade_rate=constraints_data.get("max_trade_rate", 200),
            max_delta=constraints_data.get("max_delta", 500),
            min_delta=constraints_data.get("min_delta", -500),
        ),
        costs=MPCCosts(
            c_quad=costs_data.get("c_quad", 0.001),
            c_lin=costs_data.get("c_lin", 0.01),
        ),
        kappa_init=mean_rev_data.get("kappa_init", 0.1),
        kappa_min=mean_rev_data.get("kappa_min", 0.0),
        kappa_max=mean_rev_data.get("kappa_max", 1.0),
        ema_span=mean_rev_data.get("ema_span", 20),
    )
    
    # Strategy params
    gs_data = strategy_params.get("gamma_scalping", {})
    inst_sel_data = gs_data.get("instrument_selection", {})
    hedging_data = gs_data.get("hedging", {})
    
    strategy = GammaScalpingParams(
        enabled=gs_data.get("enabled", True),
        instrument_selection=InstrumentSelection(
            structure=inst_sel_data.get("structure", "straddle"),
            strangle_width=inst_sel_data.get("strangle_width", 0.02),
            strike_selection=inst_sel_data.get("strike_selection", "atm"),
            target_dte=inst_sel_data.get("target_dte", 30),
            min_dte=inst_sel_data.get("min_dte", 21),
            max_dte=inst_sel_data.get("max_dte", 45),
            roll_dte=inst_sel_data.get("roll_dte", 7),
            roll_to_dte=inst_sel_data.get("roll_to_dte", 30),
        ),
        sizing_method=gs_data.get("position_sizing", {}).get("method", "fixed"),
        fixed_contracts=gs_data.get("position_sizing", {}).get("fixed_contracts", 10),
        min_vol_premium=gs_data.get("entry", {}).get("min_vol_premium", 0.01),
        take_profit=gs_data.get("exit", {}).get("take_profit", 0.20),
        stop_loss=gs_data.get("exit", {}).get("stop_loss", -0.30),
        hedging=HedgingConfig(
            hedge_instrument=hedging_data.get("hedge_instrument", "equity"),
            max_delta_exposure=hedging_data.get("max_delta_exposure", 100),
            min_hedge_size=hedging_data.get("min_hedge_size", 10),
        ),
    )
    
    # Risk params
    risk_data = strategy_params.get("risk", {})
    risk = RiskParams(
        max_contracts=risk_data.get("max_contracts", 50),
        max_notional=risk_data.get("max_notional", 100000.0),
        max_loss_daily=risk_data.get("max_loss_daily", 2000.0),
        max_delta=risk_data.get("max_delta", 500.0),
        max_gamma=risk_data.get("max_gamma", 100.0),
        max_vega=risk_data.get("max_vega", 5000.0),
        max_theta=risk_data.get("max_theta", -500.0),
        margin_buffer=risk_data.get("margin_buffer", 0.30),
    )
    
    # Build complete config
    app_config = AppConfig(
        broker=broker,
        instrument=instrument,
        mpc=mpc,
        strategy=strategy,
        risk=risk,
        mode=settings.get("app", {}).get("mode", "paper"),
    )
    
    _config_cache = app_config
    logger.info(
        "Configuration loaded",
        mode=app_config.mode,
        broker=app_config.broker.name,
        symbol=app_config.instrument.symbol,
    )
    
    return app_config


# =============================================================================
# Convenience Functions
# =============================================================================

def get_settings() -> Dict[str, Any]:
    """Get raw settings.yaml as dict."""
    return _load_yaml("settings.yaml")


def get_mpc_params() -> Dict[str, Any]:
    """Get raw mpc_params.yaml as dict."""
    return _load_yaml("mpc_params.yaml")


def get_strategy_params() -> Dict[str, Any]:
    """Get raw strategy_params.yaml as dict."""
    return _load_yaml("strategy_params.yaml")


def get_broker_config() -> BrokerConfig:
    """Get typed broker configuration."""
    config = load_config()
    return config.broker


def get_mpc_config() -> MPCParams:
    """Get typed MPC configuration."""
    config = load_config()
    return config.mpc


def get_strategy_config() -> GammaScalpingParams:
    """Get typed strategy configuration."""
    config = load_config()
    return config.strategy


def get_risk_config() -> RiskParams:
    """Get typed risk configuration."""
    config = load_config()
    return config.risk


def get_instrument_config() -> InstrumentConfig:
    """Get typed instrument configuration."""
    config = load_config()
    return config.instrument
