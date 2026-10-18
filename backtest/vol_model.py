"""
Volatility model for backtesting.

Maps daily VIX levels to implied volatility for synthetic options,
with simple term-structure scaling.

Limitations (documented):
- Intraday: IV is constant per day (uses daily VIX close).
- Term structure: simple sqrt(T) scaling, no skew.
"""

import math
from datetime import datetime
from typing import Optional

import pandas as pd


class VolModel:
    """
    VIX-calibrated implied volatility model.

    VIX represents ~30-day ATM SPX implied vol (annualised, in % points).
    We convert to a decimal IV and apply term-structure scaling.

    Args:
        vix_data: DataFrame with DatetimeIndex and a 'close' column (VIX daily).
        scale_factor: Multiplier applied after base conversion (default 1.0).
    """

    # VIX is quoted as annualised vol * 100, targeting 30-calendar-day variance
    _VIX_REFERENCE_DAYS = 30

    def __init__(self, vix_data: pd.DataFrame, scale_factor: float = 1.0):
        if vix_data.empty:
            raise ValueError("VIX data is empty")
        self._vix = vix_data["close"].sort_index()
        self._scale = scale_factor

    # ------------------------------------------------------------------
    def get_iv(
        self,
        sim_time: datetime,
        target_dte: int = 30,
        *,
        fallback_iv: float = 0.18,
    ) -> float:
        """
        Return annualised implied volatility for *sim_time* and *target_dte*.

        Steps:
        1. Look up the most recent VIX close on or before *sim_time*.
        2. Convert to decimal: base_iv = VIX / 100.
        3. Apply term-structure scaling:
               iv = base_iv * sqrt(VIX_REF / target_dte)
           (clamped so target_dte >= 1 to avoid division by zero).
        4. Multiply by the user-configurable scale_factor.

        Args:
            sim_time: Current simulation timestamp.
            target_dte: Days to expiry of the option being priced.
            fallback_iv: Value returned when no VIX data is available.

        Returns:
            Annualised implied volatility as a decimal (e.g. 0.18 for 18%).
        """
        date_key = pd.Timestamp(sim_time).normalize()

        # Find the most recent VIX close on or before this date
        mask = self._vix.index <= date_key
        if not mask.any():
            # Before any VIX data -- use fallback
            return fallback_iv * self._scale

        vix_level = float(self._vix.loc[mask].iloc[-1])

        base_iv = vix_level / 100.0

        # Term-structure scaling: sqrt(30 / target_dte)
        effective_dte = max(target_dte, 1)
        term_adj = math.sqrt(self._VIX_REFERENCE_DAYS / effective_dte)
        iv = base_iv * term_adj * self._scale

        # Clamp to reasonable range [0.01, 3.0]
        return max(0.01, min(iv, 3.0))
