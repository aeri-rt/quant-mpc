"""
Custom exceptions for the quant-mpc system.

Organized by module/subsystem for easier error handling and debugging.
"""


class QuantMPCError(Exception):
    """Base exception for all quant-mpc errors."""
    pass


# =============================================================================
# Data Errors
# =============================================================================

class DataError(QuantMPCError):
    """Base class for data-related errors."""
    pass


class DataProviderError(DataError):
    """Error from a data provider (IBKR, Polygon, etc.)."""
    pass


class DataNotFoundError(DataError):
    """Requested data not found."""
    pass


class DataValidationError(DataError):
    """Data failed validation checks."""
    pass


# =============================================================================
# Broker Errors
# =============================================================================

class BrokerError(QuantMPCError):
    """Base class for broker-related errors."""
    pass


class BrokerConnectionError(BrokerError):
    """Failed to connect to broker."""
    pass


class BrokerOrderError(BrokerError):
    """Error submitting or managing an order."""
    pass


class BrokerPositionError(BrokerError):
    """Error with position data."""
    pass


class InsufficientFundsError(BrokerError):
    """Insufficient funds/margin for trade."""
    pass


# =============================================================================
# Configuration Errors
# =============================================================================

class ConfigError(QuantMPCError):
    """Base class for configuration errors."""
    pass


class ConfigNotFoundError(ConfigError):
    """Configuration file not found."""
    pass


class ConfigValidationError(ConfigError):
    """Configuration failed validation."""
    pass


# =============================================================================
# Control Errors (MPC/MHE)
# =============================================================================

class ControlError(QuantMPCError):
    """Base class for control-related errors."""
    pass


class MPCSolverError(ControlError):
    """MPC solver failed to find solution."""
    pass


class MHESolverError(ControlError):
    """MHE solver failed to find solution."""
    pass


class InfeasibleProblemError(ControlError):
    """Optimization problem is infeasible."""
    pass


class NumericalError(ControlError):
    """Numerical issues in optimization."""
    pass


# =============================================================================
# Strategy Errors
# =============================================================================

class StrategyError(QuantMPCError):
    """Base class for strategy errors."""
    pass


class InvalidSignalError(StrategyError):
    """Strategy generated an invalid signal."""
    pass


# =============================================================================
# Risk Errors
# =============================================================================

class RiskError(QuantMPCError):
    """Base class for risk-related errors."""
    pass


class RiskLimitExceededError(RiskError):
    """A risk limit has been exceeded."""
    pass


class KillSwitchTriggeredError(RiskError):
    """Emergency kill switch has been triggered."""
    pass

