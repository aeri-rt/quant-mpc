"""
Tests for the data validation module.

Tests validate_quote, validate_option_quote, and validate_spot_price
with various edge cases: NaN, zero, negative, bid > ask, etc.
"""

import math
import pytest
from datetime import datetime

from src.core.types import Quote, OptionQuote, OptionType
from src.utils.validation import (
    validate_quote,
    validate_option_quote,
    validate_spot_price,
    ValidationError,
)


# =============================================================================
# validate_spot_price
# =============================================================================

class TestValidateSpotPrice:
    """Tests for validate_spot_price."""
    
    def test_valid_price(self):
        assert validate_spot_price(500.0) == 500.0
    
    def test_nan_raises(self):
        with pytest.raises(ValidationError, match="invalid"):
            validate_spot_price(float('nan'))
    
    def test_inf_raises(self):
        with pytest.raises(ValidationError, match="invalid"):
            validate_spot_price(float('inf'))
    
    def test_zero_raises(self):
        with pytest.raises(ValidationError, match="non-positive"):
            validate_spot_price(0.0)
    
    def test_negative_raises(self):
        with pytest.raises(ValidationError, match="non-positive"):
            validate_spot_price(-10.0)


# =============================================================================
# validate_quote
# =============================================================================

class TestValidateQuote:
    """Tests for validate_quote."""
    
    def _make_quote(self, bid=500.0, ask=501.0, last=500.5, bid_size=100, ask_size=200, last_size=50):
        return Quote(
            bid=bid, ask=ask, bid_size=bid_size, ask_size=ask_size,
            last=last, last_size=last_size, timestamp=datetime.now(),
        )
    
    def test_valid_quote_passes_through(self):
        q = self._make_quote()
        result = validate_quote(q, "SPY")
        assert result.bid == 500.0
        assert result.ask == 501.0
    
    def test_nan_bid_clamped(self):
        q = self._make_quote(bid=float('nan'))
        result = validate_quote(q, "SPY")
        assert result.bid > 0  # Should be filled from ask or last
        assert not math.isnan(result.bid)
    
    def test_nan_ask_clamped(self):
        q = self._make_quote(ask=float('nan'))
        result = validate_quote(q, "SPY")
        assert result.ask > 0
        assert not math.isnan(result.ask)
    
    def test_nan_last_clamped(self):
        q = self._make_quote(last=float('nan'))
        result = validate_quote(q, "SPY")
        assert result.last > 0
        assert not math.isnan(result.last)
    
    def test_all_zero_raises(self):
        with pytest.raises(ValidationError, match="No usable price"):
            validate_quote(self._make_quote(bid=0, ask=0, last=0), "SPY")
    
    def test_inverted_spread_swapped(self):
        q = self._make_quote(bid=501.0, ask=500.0)
        result = validate_quote(q, "SPY")
        assert result.bid <= result.ask
    
    def test_zero_bid_filled_from_ask(self):
        q = self._make_quote(bid=0, ask=501.0, last=500.5)
        result = validate_quote(q, "SPY")
        assert result.bid == 501.0
    
    def test_zero_ask_filled_from_bid(self):
        q = self._make_quote(bid=500.0, ask=0, last=500.5)
        result = validate_quote(q, "SPY")
        assert result.ask == 500.0
    
    def test_negative_bid_clamped(self):
        q = self._make_quote(bid=-1.0)
        result = validate_quote(q, "SPY")
        assert result.bid >= 0


# =============================================================================
# validate_option_quote
# =============================================================================

class TestValidateOptionQuote:
    """Tests for validate_option_quote."""
    
    def _make_oq(self, delta=0.5, gamma=0.02, vega=0.1, theta=-0.05, iv=0.20,
                  bid=5.0, ask=5.5, last=5.25):
        return OptionQuote(
            symbol="SPY20260315C00500000",
            underlying="SPY",
            option_type=OptionType.CALL,
            strike=500.0,
            expiry=datetime(2026, 3, 15),
            bid=bid, ask=ask, last=last,
            volume=1000, open_interest=5000,
            timestamp=datetime.now(),
            delta=delta, gamma=gamma, vega=vega, theta=theta, iv=iv,
        )
    
    def test_valid_option_quote_passes_through(self):
        oq = self._make_oq()
        result = validate_option_quote(oq)
        assert result.delta == 0.5
        assert result.gamma == 0.02
        assert result.iv == 0.20
    
    def test_nan_delta_replaced(self):
        oq = self._make_oq(delta=float('nan'))
        result = validate_option_quote(oq)
        assert result.delta == 0.0
    
    def test_nan_gamma_replaced(self):
        oq = self._make_oq(gamma=float('nan'))
        result = validate_option_quote(oq)
        assert result.gamma == 0.0
    
    def test_nan_iv_replaced(self):
        oq = self._make_oq(iv=float('nan'))
        result = validate_option_quote(oq)
        assert result.iv == 0.0
    
    def test_delta_clamped_above_1(self):
        oq = self._make_oq(delta=1.5)
        result = validate_option_quote(oq)
        assert result.delta == 1.0
    
    def test_delta_clamped_below_neg1(self):
        oq = self._make_oq(delta=-1.5)
        result = validate_option_quote(oq)
        assert result.delta == -1.0
    
    def test_negative_gamma_clamped(self):
        oq = self._make_oq(gamma=-0.5)
        result = validate_option_quote(oq)
        assert result.gamma == 0.0
    
    def test_iv_clamped_above_max(self):
        oq = self._make_oq(iv=6.0)
        result = validate_option_quote(oq)
        assert result.iv == 5.0
    
    def test_negative_bid_clamped(self):
        oq = self._make_oq(bid=-1.0)
        result = validate_option_quote(oq)
        assert result.bid == 0.0
    
    def test_inf_vega_replaced(self):
        oq = self._make_oq(vega=float('inf'))
        result = validate_option_quote(oq)
        assert result.vega == 0.0
