"""ATM implied-vol proxy from the Cboe VIX term structure (PRIOR session's closes only).

VIX9D/VIX/VIX3M are variance-swap vols sigma_var at 9/30/91 calendar days. Total variance
w(T) = sigma_var(T)^2 * T is interpolated linearly in T between the tenors and extrapolated
flat in sigma outside them. The ATM vol follows the DDKZ relation (Demeterfi, Derman, Kamal,
Zou 1999, eq. 31) sigma_var^2 ~= sigma_atm^2 * (1 + 3 T b^2) for a linear skew slope b.

`skew_b` is a calibration knob so the 30-day VIX/ATM ratio lands in Cboe's observed 1.1-1.2
(b = 1 -> sqrt(1 + 3 * 30/365) ~= 1.1165); it is NOT a measured skew slope.
"""

import math
from datetime import date

import numpy as np
import pandas as pd

TENORS = ((9 / 365, "vix9d"), (30 / 365, "vix"), (91 / 365, "vix3m"))


def atm_iv(term: pd.DataFrame, d: date, T_years: float, skew_b: float = 1.0) -> float:
    """ATM vol for a T_years option on session `d`, from the last term-structure row BEFORE d."""
    prior = term.loc[: pd.Timestamp(d) - pd.Timedelta(days=1)]
    if prior.empty:
        raise ValueError("no prior-day term structure")
    row = prior.iloc[-1]
    ts = [t for t, _ in TENORS]
    sig = [float(row[c]) for _, c in TENORS]
    if T_years <= ts[0]:
        sigma_var = sig[0]
    elif T_years >= ts[-1]:
        sigma_var = sig[-1]
    else:
        w = np.interp(T_years, ts, [s * s * t for s, t in zip(sig, ts, strict=True)])
        sigma_var = math.sqrt(w / T_years)
    return sigma_var / math.sqrt(1 + 3 * T_years * skew_b**2)
