"""
Integration tests for the Quant-MPC system.

These tests validate the full pipeline from data through strategy to execution.
Run with: pytest tests/test_integration.py -v

Note: Some tests require IBKR TWS/Gateway to be running.
"""

import pytest
import asyncio
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Ensure src is in path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.black_scholes import BlackScholesNumeric
from src.models.greeks import (
    PortfolioGreeksCalculator,
    OptionPosition,
    EquityPosition,
    PositionType,
)
from src.control.mpc_controller import DeltaHedgeMPC, MPCConfig, MPCState
from src.strategy.gamma_scalping import GammaScalpingStrategy, GammaScalpingConfig
from src.risk.risk_manager import RiskManager, RiskConfig, RiskCheckResult
from src.execution.order_manager import OrderRequest, OrderType
from src.utils.config import load_config, AppConfig
from src.utils.logging import setup_logging, LogConfig


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(scope="module")
def setup_logging_fixture():
    """Set up logging for tests."""
    log_config = LogConfig(
        log_dir=Path("./tests/logs"),
        console_level="WARNING",
        file_level="DEBUG",
    )
    setup_logging(log_config)


@pytest.fixture
def bs_numeric():
    """Black-Scholes numeric calculator."""
    return BlackScholesNumeric()


@pytest.fixture
def greeks_calculator():
    """Portfolio Greeks calculator."""
    return PortfolioGreeksCalculator(
        risk_free_rate=0.05,
        dividend_yield=0.0106,
    )


@pytest.fixture
def mpc_controller():
    """MPC controller with default config."""
    config = MPCConfig(
        n_horizon=10,
        q_max=500,
        u_max=100,
        kappa=0.2,
    )
    return DeltaHedgeMPC(config)


@pytest.fixture
def risk_manager():
    """Risk manager with test config."""
    config = RiskConfig(
        max_equity_position=500,
        max_single_trade=100,
        max_daily_loss=1000,
        max_portfolio_delta=300,
    )
    return RiskManager(config)


@pytest.fixture
def sample_option_positions():
    """Sample option positions for testing (ATM straddle)."""
    spot = 500.0
    strike = 500.0
    expiry = datetime.now() + timedelta(days=30)
    
    call_pos = OptionPosition(
        symbol="SPY240215C00500000",
        position_type=PositionType.CALL,
        quantity=10,
        strike=strike,
        expiry=expiry,
        iv=0.15,
        underlying_price=spot,
    )
    
    put_pos = OptionPosition(
        symbol="SPY240215P00500000",
        position_type=PositionType.PUT,
        quantity=10,
        strike=strike,
        expiry=expiry,
        iv=0.15,
        underlying_price=spot,
    )
    
    return [call_pos, put_pos]


# =============================================================================
# Greeks Pipeline Tests
# =============================================================================

class TestGreeksPipeline:
    """Test Greeks calculation pipeline."""
    
    def test_call_delta_range(self, bs_numeric):
        """Call delta should be between 0 and 1."""
        S, K, T, r, sigma, q = 500, 500, 0.1, 0.05, 0.2, 0.01
        delta = bs_numeric.call_delta(S, K, T, r, sigma, q)
        assert 0 <= delta <= 1, f"Call delta {delta} not in [0, 1]"
    
    def test_put_delta_range(self, bs_numeric):
        """Put delta should be between -1 and 0."""
        S, K, T, r, sigma, q = 500, 500, 0.1, 0.05, 0.2, 0.01
        delta = bs_numeric.put_delta(S, K, T, r, sigma, q)
        assert -1 <= delta <= 0, f"Put delta {delta} not in [-1, 0]"
    
    def test_atm_straddle_delta_near_zero(self, greeks_calculator, sample_option_positions):
        """ATM straddle should have near-zero delta."""
        for pos in sample_option_positions:
            greeks_calculator.add_option_position(pos)
        
        greeks = greeks_calculator.calculate(underlying_price=500.0)
        
        # ATM straddle delta should be small (call delta + put delta ≈ 0)
        assert abs(greeks.delta) < 100, f"ATM straddle delta {greeks.delta} too large"
    
    def test_gamma_positive(self, greeks_calculator, sample_option_positions):
        """Long options should have positive gamma."""
        for pos in sample_option_positions:
            greeks_calculator.add_option_position(pos)
        
        greeks = greeks_calculator.calculate(underlying_price=500.0)
        assert greeks.gamma > 0, f"Gamma {greeks.gamma} should be positive for long options"
    
    def test_theta_negative(self, greeks_calculator, sample_option_positions):
        """Long options should have negative theta (time decay)."""
        for pos in sample_option_positions:
            greeks_calculator.add_option_position(pos)
        
        greeks = greeks_calculator.calculate(underlying_price=500.0)
        assert greeks.theta < 0, f"Theta {greeks.theta} should be negative for long options"


# =============================================================================
# MPC Integration Tests
# =============================================================================

class TestMPCIntegration:
    """Test MPC controller integration."""
    
    def test_mpc_solve_succeeds(self, mpc_controller):
        """MPC should successfully solve given valid state."""
        state = MPCState(
            S=500.0,
            delta_opt=50.0,
            q=0.0,
            gamma_opt=10.0,
            sigma=0.15,
            S_bar=500.0,
        )
        
        solution = mpc_controller.solve(state)
        
        assert solution.solver_status == "optimal", f"Solver failed: {solution.solver_status}"
    
    def test_mpc_reduces_delta(self, mpc_controller):
        """MPC should recommend trades that reduce delta exposure."""
        # Start with positive delta (long exposure)
        state = MPCState(
            S=500.0,
            delta_opt=100.0,  # Long delta
            q=0.0,
            gamma_opt=10.0,
            sigma=0.15,
            S_bar=500.0,
        )
        
        solution = mpc_controller.solve(state)
        
        # Should recommend selling to reduce delta
        assert solution.u_optimal < 0, f"Expected sell (u<0), got {solution.u_optimal}"
    
    def test_mpc_respects_position_limit(self, mpc_controller):
        """MPC should not exceed position limits."""
        # Already at max position
        state = MPCState(
            S=500.0,
            delta_opt=-100.0,  # Short delta, wants to buy
            q=500.0,           # Already at max
            gamma_opt=10.0,
            sigma=0.15,
            S_bar=500.0,
        )
        
        solution = mpc_controller.solve(state)
        
        # New position should not exceed limit
        new_q = state.q + solution.u_optimal
        assert new_q <= mpc_controller.config.q_max, f"Position {new_q} exceeds limit"
    
    def test_mpc_direction_regime(self, mpc_controller):
        """MPC should respect direction regime constraints."""
        # Large positive delta -> should only sell
        state = MPCState(
            S=500.0,
            delta_opt=200.0,
            q=0.0,
            gamma_opt=10.0,
            sigma=0.15,
            S_bar=500.0,
        )
        
        solution = mpc_controller.solve(state)
        
        # Should only sell (u <= 0)
        assert solution.u_optimal <= 0, f"Expected sell only, got {solution.u_optimal}"


# =============================================================================
# Strategy Integration Tests
# =============================================================================

class TestStrategyIntegration:
    """Test gamma scalping strategy integration."""
    
    def test_strategy_initialization(self):
        """Strategy should initialize correctly."""
        config = GammaScalpingConfig()
        strategy = GammaScalpingStrategy(config)
        
        assert strategy is not None
        assert strategy.config == config
    
    def test_strategy_update_returns_recommendation(self, sample_option_positions):
        """Strategy update should return valid recommendation."""
        config = GammaScalpingConfig()
        strategy = GammaScalpingStrategy(config)
        
        # Set positions
        strategy.set_option_positions(sample_option_positions)
        strategy.set_equity_position("SPY", 0, 500.0)
        
        # Get recommendation
        rec = strategy.update(spot_price=500.0)
        
        assert rec is not None
        assert hasattr(rec, 'trade_quantity')
        assert hasattr(rec, 'trade_direction')
        assert rec.mpc_status == "optimal"
    
    def test_strategy_hedge_execution(self, sample_option_positions):
        """Strategy should track hedge execution."""
        config = GammaScalpingConfig()
        strategy = GammaScalpingStrategy(config)
        
        strategy.set_option_positions(sample_option_positions)
        strategy.set_equity_position("SPY", 0, 500.0)
        
        # Execute a hedge
        strategy.execute_hedge(50)
        
        assert strategy.equity_position == 50


# =============================================================================
# Risk Manager Tests
# =============================================================================

class TestRiskManager:
    """Test risk management."""
    
    def test_trade_within_limits_passes(self, risk_manager):
        """Trade within limits should pass."""
        response = risk_manager.validate_trade(
            symbol="SPY",
            quantity=50,
            current_position=0,
            current_delta=0,
        )
        
        assert response.passed
        assert response.result == RiskCheckResult.PASS
    
    def test_trade_exceeding_position_limit_fails(self, risk_manager):
        """Trade exceeding position limit should fail or be adjusted."""
        response = risk_manager.validate_trade(
            symbol="SPY",
            quantity=600,  # Exceeds 500 limit
            current_position=0,
            current_delta=0,
        )
        
        # Should either fail or adjust quantity
        if response.passed:
            assert response.adjusted_quantity <= 500
        else:
            assert response.result == RiskCheckResult.FAIL_POSITION_LIMIT
    
    def test_trade_exceeding_single_trade_limit_adjusted(self, risk_manager):
        """Trade exceeding single trade limit should be adjusted."""
        response = risk_manager.validate_trade(
            symbol="SPY",
            quantity=150,  # Exceeds 100 limit
            current_position=0,
            current_delta=0,
        )
        
        assert response.passed
        assert response.adjusted_quantity == 100  # Capped at limit
    
    def test_kill_switch_blocks_trades(self, risk_manager):
        """Kill switch should block all trades."""
        risk_manager.activate_kill_switch("Test")
        
        response = risk_manager.validate_trade(
            symbol="SPY",
            quantity=10,
            current_position=0,
            current_delta=0,
        )
        
        assert not response.passed
        assert response.result == RiskCheckResult.FAIL_KILL_SWITCH
        
        # Clean up
        risk_manager.deactivate_kill_switch("CONFIRM")
    
    def test_risk_reducing_trade_allowed(self, risk_manager):
        """Risk-reducing trades should be allowed even near limits."""
        # Start at position limit
        response = risk_manager.validate_trade(
            symbol="SPY",
            quantity=-50,  # Selling reduces position
            current_position=500,  # At max
            current_delta=500,
        )
        
        assert response.passed


# =============================================================================
# Config Loading Tests
# =============================================================================

class TestConfigLoading:
    """Test configuration loading."""
    
    def test_load_config_returns_app_config(self):
        """load_config should return AppConfig."""
        config = load_config(reload=True)
        assert isinstance(config, AppConfig)
    
    def test_config_has_broker_settings(self):
        """Config should have broker settings."""
        config = load_config()
        assert config.broker.host == "127.0.0.1"
        assert config.broker.port in [7496, 7497, 4001, 4002]
    
    def test_config_has_instrument_settings(self):
        """Config should have instrument settings."""
        config = load_config()
        assert config.instrument.symbol == "SPY"
        assert config.instrument.dividend_yield > 0
    
    def test_config_has_mpc_params(self):
        """Config should have MPC parameters."""
        config = load_config()
        assert config.mpc.horizon > 0
        assert config.mpc.weights.delta_deviation > 0


# =============================================================================
# Order Request Tests
# =============================================================================

class TestOrderRequest:
    """Test order request creation."""
    
    def test_order_request_side_from_quantity(self):
        """Order side should be derived from quantity."""
        buy_order = OrderRequest(symbol="SPY", quantity=100)
        sell_order = OrderRequest(symbol="SPY", quantity=-100)
        
        assert buy_order.side.value == "BUY"
        assert sell_order.side.value == "SELL"
    
    def test_order_request_abs_quantity(self):
        """abs_quantity should return absolute value."""
        order = OrderRequest(symbol="SPY", quantity=-50)
        assert order.abs_quantity == 50


# =============================================================================
# End-to-End Pipeline Test
# =============================================================================

class TestEndToEndPipeline:
    """Test the complete pipeline without IBKR connection."""
    
    def test_full_pipeline_simulation(self, sample_option_positions):
        """Test full pipeline: Greeks -> MPC -> Risk -> Order."""
        # 1. Initialize components
        strategy = GammaScalpingStrategy(GammaScalpingConfig(
            mpc_horizon=10,
            mpc_q_max=500,
            mpc_u_max=100,
            min_trade_size=5,
        ))
        
        risk_mgr = RiskManager(RiskConfig(
            max_equity_position=500,
            max_single_trade=100,
        ))
        
        # 2. Set up positions
        strategy.set_option_positions(sample_option_positions)
        strategy.set_equity_position("SPY", 0, 500.0)
        
        # 3. Simulate price movement and get recommendations
        prices = [500.0, 502.0, 498.0, 501.0, 499.0]
        trades_executed = 0
        
        for price in prices:
            # Update option prices
            for pos in sample_option_positions:
                pos.underlying_price = price
            
            # Get strategy recommendation
            rec = strategy.update(spot_price=price)
            
            if rec.trade_quantity != 0:
                # Validate through risk manager
                risk_check = risk_mgr.validate_trade(
                    symbol="SPY",
                    quantity=rec.trade_quantity,
                    current_position=strategy.equity_position,
                    current_delta=rec.net_delta,
                )
                
                if risk_check.passed:
                    # Execute (simulate)
                    strategy.execute_hedge(risk_check.adjusted_quantity)
                    trades_executed += 1
        
        # Verify pipeline worked
        assert trades_executed >= 0  # May or may not trade
        assert strategy._tick_count == len(prices)


# =============================================================================
# Run tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
