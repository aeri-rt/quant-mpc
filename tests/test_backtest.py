"""
Tests for the backtest system.

Covers:
1. PnL engine -- equity trade cost basis and realised P&L
2. PnL engine -- option mark-to-market matches BS repricing
3. Fill simulator -- equity slippage and option spreads
4. Synthetic options -- ATM straddle delta neutrality
5. Vol model -- VIX -> IV conversion
6. Engine integration -- run 100 bars of synthetic data without crashes
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.black_scholes import BlackScholesNumeric
from src.models.greeks import OptionPosition, PositionType

from backtest.config import BacktestConfig
from backtest.fill_simulator import FillSimulator
from backtest.pnl_engine import PnLEngine
from backtest.synthetic_options import (
    compute_T,
    compute_dte,
    create_straddle,
    reprice_positions,
)
from backtest.vol_model import VolModel


# =========================================================================
# 1. PnL engine -- equity trades
# =========================================================================

class TestPnLEngineEquity:
    """Test equity hedge P&L accounting."""

    def test_buy_sell_realized_pnl(self):
        """Buy 100 shares at $690, sell at $692 -> realised P&L = $200 minus costs."""
        pnl = PnLEngine(initial_capital=100_000)
        t0 = datetime(2025, 12, 1, 10, 0)
        t1 = datetime(2025, 12, 1, 10, 5)

        # Buy 100 shares
        pnl.record_equity_trade(
            quantity=100, fill_price=690.0,
            commission=0.50, slippage=0.07,
            timestamp=t0,
        )
        assert pnl.equity_qty == 100

        # Sell 100 shares
        pnl.record_equity_trade(
            quantity=-100, fill_price=692.0,
            commission=0.50, slippage=0.07,
            timestamp=t1,
        )
        assert pnl.equity_qty == 0

        snap = pnl.snapshot(t1)
        assert abs(snap.equity_realized_pnl - 200.0) < 1e-6
        assert snap.total_commissions == pytest.approx(1.0, abs=1e-6)
        assert snap.total_slippage == pytest.approx(0.14, abs=1e-6)

    def test_partial_close(self):
        """Buy 200, sell 100 -> partial close, cost basis unchanged."""
        pnl = PnLEngine()
        t0 = datetime(2025, 12, 1, 10, 0)

        pnl.record_equity_trade(200, 500.0, 0.0, 0.0, t0)
        assert pnl.equity_qty == 200

        pnl.record_equity_trade(-100, 505.0, 0.0, 0.0, t0)
        assert pnl.equity_qty == 100
        snap = pnl.snapshot(t0)
        assert abs(snap.equity_realized_pnl - 500.0) < 1e-6  # (505-500)*100

    def test_add_to_position(self):
        """Buy 100 at $690, then 100 at $694 -> avg cost = $692."""
        pnl = PnLEngine()
        t0 = datetime(2025, 12, 1, 10, 0)

        pnl.record_equity_trade(100, 690.0, 0.0, 0.0, t0)
        pnl.record_equity_trade(100, 694.0, 0.0, 0.0, t0)

        snap = pnl.snapshot(t0)
        assert pnl.equity_qty == 200
        assert abs(snap.equity_cost_basis - 692.0) < 1e-6


# =========================================================================
# 2. PnL engine -- options mark-to-market
# =========================================================================

class TestPnLEngineOptions:
    """Test option P&L tracking."""

    def test_option_entry_exit(self):
        """Open straddle for $5000, close for $5500 -> realised $500."""
        pnl = PnLEngine()
        t0 = datetime(2025, 12, 1, 10, 0)
        t1 = datetime(2025, 12, 1, 14, 0)

        # Entry
        pnl.record_option_entry([], total_cost=5000.0, commission=13.0, timestamp=t0)
        snap = pnl.snapshot(t0)
        assert abs(snap.option_entry_value - 5000.0) < 1e-6
        assert abs(snap.option_unrealized_pnl) < 1e-6  # at entry, mtm == cost

        # Exit (proceeds = 5500)
        pnl.record_option_exit([], total_proceeds=5500.0, commission=13.0, timestamp=t1)
        snap = pnl.snapshot(t1)
        assert abs(snap.option_realized_pnl - 500.0) < 1e-6
        assert snap.total_commissions == pytest.approx(26.0)

    def test_mark_to_market_unrealized(self):
        """After opening at $5000, MtM at $5200 -> unrealized = $200."""
        pnl = PnLEngine()
        t0 = datetime(2025, 12, 1, 10, 0)
        t1 = datetime(2025, 12, 1, 10, 5)

        pnl.record_option_entry([], total_cost=5000.0, commission=0.0, timestamp=t0)
        pnl.mark_to_market(
            option_value=5200.0, equity_value=0.0, spot=590.0,
            timestamp=t1, portfolio_gamma=0.5,
        )

        snap = pnl.snapshot(t1)
        assert abs(snap.option_unrealized_pnl - 200.0) < 1e-6


# =========================================================================
# 3. Fill simulator
# =========================================================================

class TestFillSimulator:
    """Test fill simulation for equity and options."""

    def test_equity_buy_slippage(self):
        """Buying should fill above mid."""
        fs = FillSimulator(equity_slippage_bps=1.0)
        result = fs.equity_fill(mid_price=600.0, quantity=100)
        assert result.fill_price > 600.0
        expected = 600.0 * (1 + 1.0 / 10_000)
        assert abs(result.fill_price - expected) < 1e-6
        assert result.commission == pytest.approx(0.50)  # 100 * 0.005

    def test_equity_sell_slippage(self):
        """Selling should fill below mid."""
        fs = FillSimulator(equity_slippage_bps=1.0)
        result = fs.equity_fill(mid_price=600.0, quantity=-100)
        assert result.fill_price < 600.0
        expected = 600.0 * (1 - 1.0 / 10_000)
        assert abs(result.fill_price - expected) < 1e-6

    def test_option_fill_entry(self):
        """Option entry should cost more than theoretical (pay half-spread)."""
        fs = FillSimulator(option_spread_dollars=0.05)

        pos = OptionPosition(
            symbol="SPY_C",
            position_type=PositionType.CALL,
            strike=590.0,
            expiry=datetime(2026, 1, 15),
            quantity=10,
            multiplier=100,
            current_price=5.00,
            underlying_price=590.0,
            iv=0.18,
        )
        cost = fs.option_fill([pos], is_entry=True)
        # Entry: (5.00 + 0.05) * 10 * 100 = $5050
        assert abs(cost - 5050.0) < 1e-6

    def test_option_fill_exit(self):
        """Option exit should receive less than theoretical (lose half-spread)."""
        fs = FillSimulator(option_spread_dollars=0.05)

        pos = OptionPosition(
            symbol="SPY_C",
            position_type=PositionType.CALL,
            strike=590.0,
            expiry=datetime(2026, 1, 15),
            quantity=10,
            multiplier=100,
            current_price=5.00,
            underlying_price=590.0,
            iv=0.18,
        )
        proceeds = fs.option_fill([pos], is_entry=False)
        # Exit: -(5.00 - 0.05) * 10 * 100 = -$4950
        assert abs(proceeds - (-4950.0)) < 1e-6


# =========================================================================
# 4. Synthetic options
# =========================================================================

class TestSyntheticOptions:
    """Test synthetic straddle creation and repricing."""

    def test_atm_straddle_delta_near_zero(self):
        """ATM call + put deltas should approximately cancel."""
        sim_time = datetime(2025, 12, 1, 10, 0)
        positions = create_straddle(
            spot=590.0, iv=0.18, sim_time=sim_time,
            target_dte=30, quantity=10,
        )
        assert len(positions) == 2
        call_pos = [p for p in positions if p.is_call][0]
        put_pos = [p for p in positions if p.is_put][0]

        # For ATM: call delta ~0.5, put delta ~-0.5
        call_delta = call_pos.greeks.delta * call_pos.quantity * call_pos.multiplier
        put_delta = put_pos.greeks.delta * put_pos.quantity * put_pos.multiplier
        net_delta = call_delta + put_delta

        # Net delta should be close to zero (within ±100 share-equiv for 10 contracts)
        assert abs(net_delta) < 100, f"Net delta too large: {net_delta:.1f}"

    def test_straddle_has_positive_gamma(self):
        """Long straddle should have positive portfolio gamma."""
        sim_time = datetime(2025, 12, 1, 10, 0)
        positions = create_straddle(
            spot=590.0, iv=0.18, sim_time=sim_time,
            target_dte=30, quantity=10,
        )
        total_gamma = sum(
            p.greeks.gamma * p.quantity * p.multiplier
            for p in positions
        )
        assert total_gamma > 0

    def test_reprice_updates_values(self):
        """After repricing with a new spot, current_price should change."""
        sim_time = datetime(2025, 12, 1, 10, 0)
        positions = create_straddle(
            spot=590.0, iv=0.18, sim_time=sim_time,
            target_dte=30, quantity=10,
        )
        old_call_price = positions[0].current_price

        # Move spot up by $5
        new_time = sim_time + timedelta(minutes=5)
        reprice_positions(positions, spot=595.0, iv=0.18, sim_time=new_time)

        # Call price should have increased
        assert positions[0].current_price != old_call_price

    def test_compute_dte_and_T(self):
        """Verify DTE and T calculations bypass datetime.now()."""
        sim_time = datetime(2025, 12, 1, 10, 0)
        expiry = datetime(2025, 12, 31, 16, 0)

        dte = compute_dte(expiry, sim_time)
        T = compute_T(expiry, sim_time)

        assert dte == 30  # ~30 days
        assert 0.08 < T < 0.09  # ~30/365 ≈ 0.082


# =========================================================================
# 5. Vol model
# =========================================================================

class TestVolModel:
    """Test VIX-to-IV conversion."""

    def test_vix_20_gives_iv_near_020(self):
        """VIX=20 with target_dte=30 should give IV close to 0.20."""
        dates = pd.bdate_range("2025-12-01", "2025-12-31")
        vix_df = pd.DataFrame({"close": [20.0] * len(dates)}, index=dates)
        model = VolModel(vix_df, scale_factor=1.0)

        iv = model.get_iv(datetime(2025, 12, 15, 10, 0), target_dte=30)
        # VIX=20 -> base_iv=0.20, term_adj=sqrt(30/30)=1.0 -> iv=0.20
        assert abs(iv - 0.20) < 0.01

    def test_term_structure_scaling(self):
        """Shorter DTE -> higher IV (sqrt scaling)."""
        dates = pd.bdate_range("2025-12-01", "2025-12-31")
        vix_df = pd.DataFrame({"close": [20.0] * len(dates)}, index=dates)
        model = VolModel(vix_df)

        iv_30d = model.get_iv(datetime(2025, 12, 15, 10, 0), target_dte=30)
        iv_7d = model.get_iv(datetime(2025, 12, 15, 10, 0), target_dte=7)
        # 7-day should have higher IV than 30-day
        assert iv_7d > iv_30d

    def test_scale_factor(self):
        """Scale factor should multiply the output IV."""
        dates = pd.bdate_range("2025-12-01", "2025-12-31")
        vix_df = pd.DataFrame({"close": [20.0] * len(dates)}, index=dates)

        m1 = VolModel(vix_df, scale_factor=1.0)
        m2 = VolModel(vix_df, scale_factor=1.5)

        iv1 = m1.get_iv(datetime(2025, 12, 15, 10, 0), target_dte=30)
        iv2 = m2.get_iv(datetime(2025, 12, 15, 10, 0), target_dte=30)
        assert abs(iv2 / iv1 - 1.5) < 0.01

    def test_fallback_before_data(self):
        """Before any VIX data, should return fallback IV."""
        dates = pd.bdate_range("2025-12-15", "2025-12-31")
        vix_df = pd.DataFrame({"close": [20.0] * len(dates)}, index=dates)
        model = VolModel(vix_df)

        iv = model.get_iv(datetime(2025, 12, 1, 10, 0), target_dte=30, fallback_iv=0.15)
        assert abs(iv - 0.15) < 1e-6


# =========================================================================
# 6. Engine integration -- run synthetic data without crashes
# =========================================================================

class TestEngineIntegration:
    """Integration test: run a short synthetic backtest."""

    def test_100_bars_no_crash(self):
        """Run 100 bars of synthetic data and verify P&L is computed."""
        # Create synthetic bars
        np.random.seed(42)
        n_bars = 120  # 60 warmup + 60 actual
        base_price = 590.0
        timestamps = pd.date_range("2025-12-15 09:30", periods=n_bars, freq="5min")
        prices = base_price + np.cumsum(np.random.randn(n_bars) * 0.3)
        bars = pd.DataFrame(
            {
                "open": prices,
                "high": prices + 0.2,
                "low": prices - 0.2,
                "close": prices,
                "volume": np.random.randint(100_000, 500_000, n_bars),
            },
            index=timestamps,
        )

        # Create VIX data
        vix_dates = pd.bdate_range("2025-12-01", "2025-12-31")
        vix = pd.DataFrame({"close": [18.0] * len(vix_dates)}, index=vix_dates)

        # Config with small position
        cfg = BacktestConfig(
            warmup_bars=60,
            position_size=2,
            target_dte=30,
            roll_dte=5,
            initial_capital=50_000,
        )

        from backtest.engine import BacktestEngine
        engine = BacktestEngine(cfg)
        result = engine.run(bars, vix)

        # Should have processed all bars
        assert result.total_bars == n_bars
        assert result.warmup_bars == 60
        # Should have some bar records (bars after warmup and straddle open)
        assert len(result.records) > 0
        # P&L engine should have an equity curve
        from backtest.pnl_engine import PnLEngine
        pnl: PnLEngine = result.pnl_engine  # type: ignore[assignment]
        curve = pnl.get_equity_curve()
        assert len(curve) > 0

    def test_config_from_yaml(self, tmp_path):
        """BacktestConfig.from_yaml round-trips correctly."""
        yaml_content = """
time:
  start_date: "2025-11-01"
  end_date: "2025-12-31"
  bar_size_minutes: 1

capital:
  initial_capital: 200000.0
  position_size: 5
"""
        yaml_file = tmp_path / "test_bt.yaml"
        yaml_file.write_text(yaml_content)

        cfg = BacktestConfig.from_yaml(str(yaml_file))
        assert cfg.start_date == "2025-11-01"
        assert cfg.bar_size_minutes == 1
        assert cfg.initial_capital == 200_000.0
        assert cfg.position_size == 5
        # Defaults should remain
        assert cfg.warmup_bars == 60
