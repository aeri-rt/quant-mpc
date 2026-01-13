"""
Structured Logging System for Quant-MPC

Provides:
- Structured JSON logging for machine parsing
- Human-readable console output
- Specialized loggers for different components
- Performance metrics tracking
- Trade audit trail

Usage:
    from src.utils.logging import get_logger, log_trade, log_mpc_solve, log_metrics
    
    logger = get_logger("my_component")
    logger.info("Processing started", extra={"symbol": "SPY"})
"""

import sys
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union
from dataclasses import dataclass, asdict
from contextlib import contextmanager
import threading
import time

from loguru import logger

# Remove default loguru handler
logger.remove()


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class LogConfig:
    """Logging configuration."""
    
    # Paths
    log_dir: Path = Path("logs")
    
    # Levels
    console_level: str = "INFO"
    file_level: str = "DEBUG"
    
    # Rotation
    rotation: str = "100 MB"
    retention: str = "30 days"
    
    # Format
    console_format: str = (
        "<green>{time:HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{extra[component]}</cyan> | "
        "<level>{message}</level>"
    )
    
    # Serialization
    serialize_files: bool = True  # JSON format for file logs


_config = LogConfig()
_initialized = False
_lock = threading.Lock()


# =============================================================================
# Structured Log Records
# =============================================================================

@dataclass
class TradeLog:
    """Structured trade execution log."""
    timestamp: str
    component: str = "trade"
    event: str = "trade_executed"
    
    # Trade details
    symbol: str = ""
    side: str = ""  # BUY/SELL
    quantity: float = 0.0
    price: float = 0.0
    order_type: str = ""
    
    # Order tracking
    status: str = ""  # SUBMITTED, FILLED, CANCELLED, REJECTED
    order_id: str = ""
    broker_order_id: Optional[int] = None
    
    # Fill details
    filled_quantity: int = 0
    avg_fill_price: float = 0.0
    fill_price: float = 0.0
    commission: float = 0.0
    
    # Context
    delta_before: float = 0.0
    delta_after: float = 0.0
    mpc_recommendation: float = 0.0
    
    # Execution quality
    slippage: float = 0.0
    latency_ms: float = 0.0
    
    # Reason/Error
    reason: str = ""
    error: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MPCSolveLog:
    """Structured MPC solver log."""
    timestamp: str
    component: str = "mpc"
    event: str = "mpc_solve"
    
    # Input state
    S: float = 0.0
    delta_opt: float = 0.0
    q: float = 0.0
    delta_portfolio: float = 0.0
    gamma_opt: float = 0.0
    sigma: float = 0.0
    S_bar: float = 0.0
    
    # Solution
    u_optimal: float = 0.0
    solver_status: str = ""
    solve_time_ms: float = 0.0
    
    # Regime
    direction_regime: str = ""  # BUY_ONLY, SELL_ONLY, BOTH
    
    # Predicted trajectory summary
    pred_delta_min: float = 0.0
    pred_delta_max: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GreeksLog:
    """Structured Greeks calculation log."""
    timestamp: str
    component: str = "greeks"
    event: str = "greeks_calculated"
    
    # Portfolio Greeks
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0
    vega: float = 0.0
    
    # Position counts
    n_options: int = 0
    n_equity: int = 0
    
    # Underlying
    spot_price: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MetricsLog:
    """Structured metrics/performance log."""
    timestamp: str
    component: str = "metrics"
    event: str = "metrics_snapshot"
    
    # P&L
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    total_pnl: float = 0.0
    
    # Risk
    portfolio_delta: float = 0.0
    portfolio_gamma: float = 0.0
    max_drawdown: float = 0.0
    
    # Activity
    n_trades: int = 0
    total_volume: float = 0.0
    total_commission: float = 0.0
    
    # Performance
    sharpe_ratio: Optional[float] = None
    win_rate: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# =============================================================================
# Logger Setup
# =============================================================================

def setup_logging(config: Optional[LogConfig] = None) -> None:
    """
    Initialize the logging system.
    
    Call this once at application startup.
    """
    global _config, _initialized
    
    with _lock:
        if _initialized:
            return
        
        if config:
            _config = config
        
        # Create log directory
        _config.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Console handler (human-readable)
        logger.add(
            sys.stderr,
            level=_config.console_level,
            format=_config.console_format,
            colorize=True,
            filter=lambda record: record["extra"].get("component", "root") != "_metrics"
        )
        
        # Main log file (all events)
        logger.add(
            _config.log_dir / "quant-mpc_{time:YYYY-MM-DD}.log",
            level=_config.file_level,
            rotation=_config.rotation,
            retention=_config.retention,
            serialize=_config.serialize_files,
            enqueue=True,  # Thread-safe
        )
        
        # Trade-specific log file
        logger.add(
            _config.log_dir / "trades_{time:YYYY-MM-DD}.log",
            level="INFO",
            rotation=_config.rotation,
            retention=_config.retention,
            serialize=True,
            filter=lambda record: record["extra"].get("component") == "trade",
            enqueue=True,
        )
        
        # MPC solver log file
        logger.add(
            _config.log_dir / "mpc_{time:YYYY-MM-DD}.log",
            level="DEBUG",
            rotation=_config.rotation,
            retention=_config.retention,
            serialize=True,
            filter=lambda record: record["extra"].get("component") == "mpc",
            enqueue=True,
        )
        
        # Metrics log file (for analysis)
        logger.add(
            _config.log_dir / "metrics_{time:YYYY-MM-DD}.log",
            level="INFO",
            rotation=_config.rotation,
            retention=_config.retention,
            serialize=True,
            filter=lambda record: record["extra"].get("component") == "metrics",
            enqueue=True,
        )
        
        # Error-only log file
        logger.add(
            _config.log_dir / "errors_{time:YYYY-MM-DD}.log",
            level="ERROR",
            rotation=_config.rotation,
            retention=_config.retention,
            serialize=True,
            enqueue=True,
        )
        
        _initialized = True
        logger.bind(component="system").info("Logging system initialized", log_dir=str(_config.log_dir))


def get_logger(component: str = "root"):
    """
    Get a logger bound to a specific component.
    
    Args:
        component: Component name (e.g., "mpc", "greeks", "execution")
        
    Returns:
        Logger instance bound to the component
        
    Example:
        logger = get_logger("mpc")
        logger.info("Solving MPC problem")
    """
    if not _initialized:
        setup_logging()
    
    return logger.bind(component=component)


# =============================================================================
# Convenience Functions for Structured Logging
# =============================================================================

def log_trade(
    symbol: str,
    side: str,
    quantity: float,
    order_type: str = "MARKET",
    price: float = 0.0,
    status: str = "",
    order_id: str = "",
    broker_order_id: Optional[int] = None,
    filled_quantity: int = 0,
    avg_fill_price: float = 0.0,
    fill_price: float = 0.0,
    commission: float = 0.0,
    delta_before: float = 0.0,
    delta_after: float = 0.0,
    mpc_recommendation: float = 0.0,
    slippage: float = 0.0,
    latency_ms: float = 0.0,
    reason: str = "",
    error: str = "",
) -> None:
    """
    Log a trade/order event with full context.
    
    Example:
        # Log order submission
        log_trade(
            symbol="SPY",
            side="SELL",
            quantity=100,
            order_type="MARKET",
            status="SUBMITTED",
            order_id="abc123",
            reason="Delta hedge"
        )
        
        # Log fill
        log_trade(
            symbol="SPY",
            side="SELL",
            quantity=100,
            status="FILL",
            order_id="abc123",
            fill_price=500.25,
            commission=0.50,
        )
    """
    if not _initialized:
        setup_logging()
    
    trade_log = TradeLog(
        timestamp=datetime.now().isoformat(),
        symbol=symbol,
        side=side,
        quantity=quantity,
        price=price,
        order_type=order_type,
        status=status,
        order_id=order_id,
        broker_order_id=broker_order_id,
        filled_quantity=filled_quantity,
        avg_fill_price=avg_fill_price,
        fill_price=fill_price,
        commission=commission,
        delta_before=delta_before,
        delta_after=delta_after,
        mpc_recommendation=mpc_recommendation,
        slippage=slippage,
        latency_ms=latency_ms,
        reason=reason,
        error=error,
    )
    
    # Exclude 'component' from dict since we bind it separately
    log_data = {k: v for k, v in trade_log.to_dict().items() if k != 'component'}
    
    # Format message based on status
    if status == "FILL":
        msg = f"FILL: {quantity:.0f} {symbol} @ ${fill_price:.2f}"
    elif status:
        msg = f"ORDER [{status}]: {side} {quantity:.0f} {symbol}"
    else:
        msg = f"TRADE: {side} {quantity:.0f} {symbol} @ ${price:.2f}"
    
    logger.bind(component="trade", **log_data).info(msg)


def log_mpc_solve(
    state: "MPCState",  # Forward reference
    solution: "MPCSolution",  # Forward reference
    solve_time_ms: float,
    direction_regime: str = "",
) -> None:
    """
    Log an MPC solve with full state and solution.
    
    Example:
        log_mpc_solve(state, solution, solve_time_ms=5.2, direction_regime="SELL_ONLY")
    """
    if not _initialized:
        setup_logging()
    
    delta_portfolio = state.delta_opt + state.q
    
    # Get trajectory bounds
    pred_delta_min = float(solution.delta_portfolio_pred.min()) if len(solution.delta_portfolio_pred) > 0 else 0.0
    pred_delta_max = float(solution.delta_portfolio_pred.max()) if len(solution.delta_portfolio_pred) > 0 else 0.0
    
    mpc_log = MPCSolveLog(
        timestamp=datetime.now().isoformat(),
        S=state.S,
        delta_opt=state.delta_opt,
        q=state.q,
        delta_portfolio=delta_portfolio,
        gamma_opt=state.gamma_opt,
        sigma=state.sigma,
        S_bar=state.S_bar,
        u_optimal=solution.u_optimal,
        solver_status=solution.solver_status,
        solve_time_ms=solve_time_ms,
        direction_regime=direction_regime,
        pred_delta_min=pred_delta_min,
        pred_delta_max=pred_delta_max,
    )
    
    # Exclude 'component' from dict since we bind it separately
    log_data = {k: v for k, v in mpc_log.to_dict().items() if k != 'component'}
    logger.bind(component="mpc", **log_data).debug(
        f"MPC: u={solution.u_optimal:.1f}, Δ_port={delta_portfolio:.1f}, status={solution.solver_status}"
    )


def log_greeks(
    delta: float,
    gamma: float,
    theta: float,
    vega: float,
    n_options: int,
    n_equity: int,
    spot_price: float,
) -> None:
    """Log portfolio Greeks snapshot."""
    if not _initialized:
        setup_logging()
    
    greeks_log = GreeksLog(
        timestamp=datetime.now().isoformat(),
        delta=delta,
        gamma=gamma,
        theta=theta,
        vega=vega,
        n_options=n_options,
        n_equity=n_equity,
        spot_price=spot_price,
    )
    
    # Exclude 'component' from dict since we bind it separately
    log_data = {k: v for k, v in greeks_log.to_dict().items() if k != 'component'}
    logger.bind(component="greeks", **log_data).debug(
        f"Greeks: Δ={delta:.1f}, Γ={gamma:.2f}, Θ={theta:.2f}, V={vega:.2f}"
    )


def log_metrics(
    realized_pnl: float = 0.0,
    unrealized_pnl: float = 0.0,
    portfolio_delta: float = 0.0,
    portfolio_gamma: float = 0.0,
    n_trades: int = 0,
    total_volume: float = 0.0,
    total_commission: float = 0.0,
    max_drawdown: float = 0.0,
    sharpe_ratio: Optional[float] = None,
    win_rate: Optional[float] = None,
) -> None:
    """Log performance metrics snapshot."""
    if not _initialized:
        setup_logging()
    
    total_pnl = realized_pnl + unrealized_pnl
    
    metrics_log = MetricsLog(
        timestamp=datetime.now().isoformat(),
        realized_pnl=realized_pnl,
        unrealized_pnl=unrealized_pnl,
        total_pnl=total_pnl,
        portfolio_delta=portfolio_delta,
        portfolio_gamma=portfolio_gamma,
        max_drawdown=max_drawdown,
        n_trades=n_trades,
        total_volume=total_volume,
        total_commission=total_commission,
        sharpe_ratio=sharpe_ratio,
        win_rate=win_rate,
    )
    
    # Exclude 'component' from dict since we bind it separately
    log_data = {k: v for k, v in metrics_log.to_dict().items() if k != 'component'}
    logger.bind(component="metrics", **log_data).info(
        f"Metrics: P&L=${total_pnl:.2f}, Δ={portfolio_delta:.1f}, trades={n_trades}"
    )


# =============================================================================
# Context Managers and Utilities
# =============================================================================

@contextmanager
def log_context(**kwargs):
    """
    Context manager for adding context to all logs within the block.
    
    Example:
        with log_context(session_id="abc123", symbol="SPY"):
            logger.info("Processing...")  # Will include session_id and symbol
    """
    with logger.contextualize(**kwargs):
        yield


class Timer:
    """
    Timer for measuring and logging operation duration.
    
    Example:
        with Timer("mpc_solve", component="mpc") as t:
            result = mpc.solve(state)
        print(f"Took {t.elapsed_ms:.2f} ms")
    """
    
    def __init__(self, operation: str, component: str = "timing", log_level: str = "DEBUG"):
        self.operation = operation
        self.component = component
        self.log_level = log_level
        self.start_time: float = 0
        self.end_time: float = 0
        self.elapsed_ms: float = 0
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, *args):
        self.end_time = time.perf_counter()
        self.elapsed_ms = (self.end_time - self.start_time) * 1000
        
        log_func = getattr(get_logger(self.component), self.log_level.lower())
        log_func(
            f"{self.operation} completed",
            elapsed_ms=self.elapsed_ms,
            operation=self.operation,
        )


# =============================================================================
# Test-Specific Logging
# =============================================================================

class TestLogger:
    """
    Specialized logger for validation tests.
    
    Tracks test results and provides summary.
    """
    
    def __init__(self, test_name: str):
        self.test_name = test_name
        self.logger = get_logger("test")
        self.results: list = []
        self.start_time = datetime.now()
    
    def log_test_start(self, description: str = "") -> None:
        """Log test suite start."""
        self.logger.info(
            f"{'='*50}\nTEST SUITE: {self.test_name}\n{'='*50}",
            test_name=self.test_name,
            description=description,
            event="test_start",
        )
    
    def log_case(
        self,
        case_name: str,
        passed: bool,
        expected: Any = None,
        actual: Any = None,
        details: str = "",
    ) -> None:
        """Log individual test case result."""
        status = "PASS" if passed else "FAIL"
        self.results.append({
            "case": case_name,
            "passed": passed,
            "expected": expected,
            "actual": actual,
            "details": details,
        })
        
        level = "info" if passed else "warning"
        log_func = getattr(self.logger, level)
        
        log_func(
            f"  [{status}] {case_name}",
            case_name=case_name,
            passed=passed,
            expected=expected,
            actual=actual,
            details=details,
            event="test_case",
        )
    
    def log_metric(self, metric_name: str, value: float, unit: str = "") -> None:
        """Log a test metric."""
        self.logger.info(
            f"  METRIC: {metric_name} = {value:.4f} {unit}",
            metric_name=metric_name,
            value=value,
            unit=unit,
            event="test_metric",
        )
    
    def log_summary(self) -> Dict[str, Any]:
        """Log test summary and return results."""
        duration = (datetime.now() - self.start_time).total_seconds()
        n_passed = sum(1 for r in self.results if r["passed"])
        n_total = len(self.results)
        all_passed = n_passed == n_total
        
        summary = {
            "test_name": self.test_name,
            "passed": n_passed,
            "failed": n_total - n_passed,
            "total": n_total,
            "all_passed": all_passed,
            "duration_seconds": duration,
        }
        
        status = "ALL PASSED" if all_passed else "SOME FAILED"
        level = "info" if all_passed else "error"
        log_func = getattr(self.logger, level)
        
        log_func(
            f"\n{'='*50}\n"
            f"SUMMARY: {self.test_name}\n"
            f"  {status}: {n_passed}/{n_total} tests passed\n"
            f"  Duration: {duration:.2f}s\n"
            f"{'='*50}",
            event="test_summary",
            **summary,
        )
        
        return summary


# =============================================================================
# Log Reader (for analysis)
# =============================================================================

def read_logs(
    log_type: str = "main",
    date: Optional[str] = None,
    component: Optional[str] = None,
    event: Optional[str] = None,
    limit: int = 1000,
) -> list:
    """
    Read and parse log files.
    
    Args:
        log_type: One of "main", "trades", "mpc", "metrics", "errors"
        date: Date string YYYY-MM-DD (defaults to today)
        component: Filter by component
        event: Filter by event type
        limit: Maximum records to return
        
    Returns:
        List of log record dicts
    """
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")
    
    file_map = {
        "main": f"quant-mpc_{date}.log",
        "trades": f"trades_{date}.log",
        "mpc": f"mpc_{date}.log",
        "metrics": f"metrics_{date}.log",
        "errors": f"errors_{date}.log",
    }
    
    log_file = _config.log_dir / file_map.get(log_type, f"{log_type}_{date}.log")
    
    if not log_file.exists():
        return []
    
    records = []
    with open(log_file, "r") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                
                # Apply filters
                if component and record.get("record", {}).get("extra", {}).get("component") != component:
                    continue
                if event and record.get("record", {}).get("extra", {}).get("event") != event:
                    continue
                
                records.append(record)
                
                if len(records) >= limit:
                    break
            except json.JSONDecodeError:
                continue
    
    return records


# =============================================================================
# Module-level initialization
# =============================================================================

# Auto-initialize with defaults on import if not in test mode
def _maybe_init():
    """Initialize logging if not already done."""
    if not _initialized:
        setup_logging()

