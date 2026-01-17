"""
Shared helper functions and constants.

This module consolidates commonly used utilities to avoid code duplication
and ensure consistency across the codebase.
"""

import math
from typing import Union

# =============================================================================
# Time Constants
# =============================================================================

# Trading days per year (NYSE)
TRADING_DAYS_PER_YEAR = 252

# Trading hours per day (NYSE regular trading hours)
TRADING_HOURS_PER_DAY = 6.5  # 9:30 AM - 4:00 PM

# Trading minutes per year
TRADING_MINUTES_PER_YEAR = TRADING_DAYS_PER_YEAR * TRADING_HOURS_PER_DAY * 60  # ~98,280

# Trading seconds per year
TRADING_SECONDS_PER_YEAR = TRADING_MINUTES_PER_YEAR * 60


# =============================================================================
# Safe Type Conversion
# =============================================================================

def safe_int(val: Union[int, float, None], default: int = 0) -> int:
    """
    Safely convert a value to int, handling None and NaN.
    
    This is particularly useful when working with IBKR ticker data,
    which can return NaN for unavailable fields.
    
    Args:
        val: Value to convert (int, float, or None)
        default: Default value if conversion fails
        
    Returns:
        Integer value, or default if input is None/NaN
        
    Examples:
        >>> safe_int(10.5)
        10
        >>> safe_int(float('nan'))
        0
        >>> safe_int(None, default=-1)
        -1
    """
    if val is None:
        return default
    if isinstance(val, float) and math.isnan(val):
        return default
    return int(val)


def safe_float(val: Union[int, float, None], default: float = 0.0) -> float:
    """
    Safely convert a value to float, handling None and NaN.
    
    Args:
        val: Value to convert (int, float, or None)
        default: Default value if conversion fails
        
    Returns:
        Float value, or default if input is None/NaN
    """
    if val is None:
        return default
    if isinstance(val, float) and math.isnan(val):
        return default
    return float(val)


def safe_price(
    val: Union[int, float, None],
    fallback: Union[int, float, None] = None,
    default: float = 0.0,
) -> float:
    """
    Get a safe price value, with fallback and default.
    
    Handles None, NaN, and non-positive prices.
    
    Args:
        val: Primary price value
        fallback: Fallback value if primary is invalid
        default: Final default if both are invalid
        
    Returns:
        Valid price, or default
    """
    # Check primary value
    if val is not None and isinstance(val, (int, float)):
        if not math.isnan(val) and val > 0:
            return float(val)
    
    # Check fallback
    if fallback is not None and isinstance(fallback, (int, float)):
        if not math.isnan(fallback) and fallback > 0:
            return float(fallback)
    
    return default


# =============================================================================
# Time Conversion
# =============================================================================

def seconds_to_years(seconds: float) -> float:
    """
    Convert seconds to years (trading time).
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Time in years (based on trading calendar)
    """
    return seconds / TRADING_SECONDS_PER_YEAR


def minutes_to_years(minutes: float) -> float:
    """
    Convert minutes to years (trading time).
    
    Args:
        minutes: Time in minutes
        
    Returns:
        Time in years (based on trading calendar)
    """
    return minutes / TRADING_MINUTES_PER_YEAR


def years_to_seconds(years: float) -> float:
    """
    Convert years to seconds (trading time).
    
    Args:
        years: Time in years
        
    Returns:
        Time in seconds (based on trading calendar)
    """
    return years * TRADING_SECONDS_PER_YEAR


def years_to_minutes(years: float) -> float:
    """
    Convert years to minutes (trading time).
    
    Args:
        years: Time in years
        
    Returns:
        Time in minutes (based on trading calendar)
    """
    return years * TRADING_MINUTES_PER_YEAR
