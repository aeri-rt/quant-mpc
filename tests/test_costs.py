"""src/costs.py: the one cost model. Defaults from src.config.Costs, no literal dollar amounts
where the config value is the source of truth."""

import pytest

from src.config import Costs
from src.costs import (
    commission,
    option_cost,
    option_half_spread,
    per_share_rate,
    reg_fees,
)

C = Costs()
C_TAF = Costs(finra_taf_per_share=0.000195)


@pytest.mark.parametrize(
    ("n", "expected"),
    [(4, 0.35), (200, 0.70), (-200, 0.70), (0, 0.0)],
)
def test_commission(n, expected):
    assert commission(n, C) == pytest.approx(expected)


@pytest.mark.parametrize("c", [C, C_TAF], ids=["default", "taf"])
def test_reg_fees(c):
    assert reg_fees(100, 600, c) == 0.0
    assert reg_fees(-100, 600, c) == pytest.approx(
        c.sec_fee_rate * 600 * 100 + c.finra_taf_per_share * 100
    )


def test_reg_fees_taf_cap_binds():
    n = -100_000
    assert C_TAF.finra_taf_per_share * abs(n) > C_TAF.finra_taf_cap
    assert reg_fees(n, 600, C_TAF) == pytest.approx(
        C_TAF.sec_fee_rate * 600 * abs(n) + C_TAF.finra_taf_cap
    )


@pytest.mark.parametrize("c", [C, C_TAF], ids=["default", "taf"])
def test_per_share_rate_sell_premium(c):
    assert per_share_rate(600, -1, c) - per_share_rate(600, +1, c) == pytest.approx(
        c.sec_fee_rate * 600 + c.finra_taf_per_share
    )


@pytest.mark.parametrize(("dte", "hs"), [(3, 0.015), (7, 0.04), (30, 0.06), (200, 0.07)])
def test_option_half_spread(dte, hs):
    assert option_half_spread(dte, C) == hs


def test_option_cost_straddle():
    assert option_cost(5, 30, C) == pytest.approx(max(1.0, 0.65 * 10) + 0.05 * 10 + 100 * 0.06 * 10)
    assert option_cost(5, 30, C) == pytest.approx(67.0)


def test_option_cost_single_leg_minimum_binds():
    assert C.option_commission * 1 < C.option_min_commission
    assert option_cost(1, 30, C, legs=1) == pytest.approx(
        C.option_min_commission + C.option_fees_per_contract + 100 * 0.06
    )
