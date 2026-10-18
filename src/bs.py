"""Black-Scholes-Merton pricing, Greeks and implied vol — scalar, per-share.

Units (canonical table in src/book.py):
    S, K       $ per share
    T          calendar years to expiry (src.clock.T_years)
    r, q       continuous annual rates (decimal)
    sigma      annualized vol (decimal)
    right      "C" or "P" (the IBKR right string)
    delta      per share, in [-1, 1]
    gamma      1/$ per share
    vega       $ per 1 vol-point per share  (= dV/dsigma / 100)
    theta      $ per calendar day per share (= dV/dt_year / 365)
"""

import math
from dataclasses import dataclass

from scipy.optimize import brentq
from scipy.stats import norm

_SIGN = {"C": 1.0, "P": -1.0}  # KeyError on any other right


@dataclass(frozen=True)
class Greeks:
    delta: float
    gamma: float
    vega: float
    theta: float

    def __add__(self, o: "Greeks") -> "Greeks":
        return Greeks(
            self.delta + o.delta, self.gamma + o.gamma, self.vega + o.vega, self.theta + o.theta
        )

    def scale(self, k: float) -> "Greeks":
        return Greeks(self.delta * k, self.gamma * k, self.vega * k, self.theta * k)

    def finite(self) -> bool:
        return all(map(math.isfinite, (self.delta, self.gamma, self.vega, self.theta)))


ZERO = Greeks(0.0, 0.0, 0.0, 0.0)


def d1(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    return (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))


def d2(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    return d1(S, K, T, r, q, sigma) - sigma * math.sqrt(T)


def price(S: float, K: float, T: float, r: float, q: float, sigma: float, right: str) -> float:
    s = _SIGN[right]
    if T <= 1e-12 or sigma <= 0:
        return max(s * (S - K), 0.0)
    a, b = d1(S, K, T, r, q, sigma), d2(S, K, T, r, q, sigma)
    return float(
        s * (S * math.exp(-q * T) * norm.cdf(s * a) - K * math.exp(-r * T) * norm.cdf(s * b))
    )


def greeks(S: float, K: float, T: float, r: float, q: float, sigma: float, right: str) -> Greeks:
    s = _SIGN[right]
    if T <= 1e-9 or sigma <= 0:
        return Greeks(s if s * (S - K) > 0 else 0.0, 0.0, 0.0, 0.0)
    a, b = d1(S, K, T, r, q, sigma), d2(S, K, T, r, q, sigma)
    sq_t = math.sqrt(T)
    fwd, disc = S * math.exp(-q * T), K * math.exp(-r * T)
    phi, n_a, n_b = float(norm.pdf(a)), float(norm.cdf(s * a)), float(norm.cdf(s * b))
    theta_year = -fwd * phi * sigma / (2 * sq_t) - s * r * disc * n_b + s * q * fwd * n_a
    return Greeks(
        delta=s * math.exp(-q * T) * n_a,
        gamma=math.exp(-q * T) * phi / (S * sigma * sq_t),
        vega=fwd * phi * sq_t / 100,
        theta=theta_year / 365,
    )


def implied_vol(
    mid: float, S: float, K: float, T: float, r: float, q: float, right: str
) -> float | None:
    """Vol in [1e-4, 5] reproducing `mid`; None if T <= 0 or mid violates no-arbitrage bounds."""
    if T <= 0 or not math.isfinite(mid):
        return None
    fwd, disc = S * math.exp(-q * T), K * math.exp(-r * T)
    lo, hi = (max(fwd - disc, 0.0), fwd) if right == "C" else (max(disc - fwd, 0.0), disc)
    if mid <= lo + 1e-10 or mid >= hi:
        return None
    try:
        return float(
            brentq(
                lambda v: price(S, K, T, r, q, v, right) - mid, 1e-4, 5.0, xtol=1e-10, maxiter=200
            )
        )
    except ValueError:  # no sign change on the bracket
        return None


def straddle_premium(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    return price(S, K, T, r, q, sigma, "C") + price(S, K, T, r, q, sigma, "P")
