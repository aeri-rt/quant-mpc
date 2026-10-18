"""
Backtest engine -- main simulation loop.

Replays historical intraday bars through the existing
``GammaScalpingStrategy`` + ``DeltaHedgeMPC``, with VIX-calibrated
synthetic options, fill simulation, and full P&L accounting.
"""

from datetime import datetime
from typing import List, Optional

import pandas as pd
from src.utils.logging import get_logger

logger = get_logger("backtest")

from src.models.greeks import OptionPosition
from src.strategy.gamma_scalping import GammaScalpingConfig, GammaScalpingStrategy
from src.strategy.position_manager import PositionManager, PositionManagerConfig
from src.control.mpc_controller import MPCConfig

from backtest.config import BacktestConfig
from backtest.fill_simulator import FillSimulator
from backtest.metrics import BacktestMetrics, BacktestResult
from backtest.pnl_engine import PnLEngine
from backtest.synthetic_options import (
    compute_T,
    compute_dte,
    create_straddle,
    reprice_positions,
)
from backtest.vol_model import VolModel


class BacktestEngine:
    """
    Offline backtesting engine for the gamma scalping strategy.

    Usage::

        engine = BacktestEngine(config)
        result = engine.run(bars_df, vix_df)
        # result contains metrics, P&L engine, per-bar records
    """

    def __init__(self, config: BacktestConfig):
        self.config = config

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self, bars: pd.DataFrame, vix: pd.DataFrame) -> BacktestResult:
        """
        Run the backtest over historical data.

        Args:
            bars: DataFrame with DatetimeIndex and at minimum a ``close``
                  column (OHLCV preferred).  Index must be timezone-naive
                  or UTC-normalised.
            vix:  DataFrame with DatetimeIndex and a ``close`` column
                  (daily VIX levels).

        Returns:
            BacktestResult with per-bar records and P&L details.
        """
        cfg = self.config
        cfg.ensure_dirs()

        # -- Wire up components ----------------------------------------
        vol_model = VolModel(vix, scale_factor=cfg.vix_scale_factor)

        gamma_cfg = GammaScalpingConfig(
            risk_free_rate=cfg.risk_free_rate,
            dividend_yield=cfg.dividend_yield,
            mpc_horizon=cfg.mpc_horizon,
            mpc_q_max=cfg.mpc_q_max,
            mpc_u_max=cfg.mpc_u_max,
            mpc_w_delta=cfg.mpc_w_delta,
        )
        mpc_cfg = MPCConfig(
            n_horizon=cfg.mpc_horizon,
            q_max=cfg.mpc_q_max,
            u_max=cfg.mpc_u_max,
            w_delta=cfg.mpc_w_delta,
        )
        strategy = GammaScalpingStrategy(gamma_cfg, mpc_cfg)

        pos_cfg = PositionManagerConfig(
            roll_dte=cfg.roll_dte,
            roll_to_dte=cfg.roll_to_dte,
            take_profit_pct=cfg.take_profit_pct,
            stop_loss_pct=cfg.stop_loss_pct,
        )
        position_mgr = PositionManager(pos_cfg)

        fill_sim = FillSimulator(
            equity_slippage_bps=cfg.equity_slippage_bps,
            equity_commission_per_share=cfg.equity_commission_per_share,
            option_spread_dollars=cfg.option_spread_dollars,
            option_commission_per_contract=cfg.option_commission_per_contract,
        )

        pnl = PnLEngine(initial_capital=cfg.initial_capital)
        metrics = BacktestMetrics()

        # -- State -----------------------------------------------------
        option_positions: List[OptionPosition] = []
        equity_qty: int = 0
        num_rolls: int = 0

        total_bars = len(bars)
        logger.info(
            f"Starting backtest: {total_bars} bars, "
            f"warmup={cfg.warmup_bars}, "
            f"capital=${cfg.initial_capital:,.0f}"
        )

        # -- Main loop -------------------------------------------------
        for i, (timestamp, bar) in enumerate(bars.iterrows()):
            sim_time: datetime = pd.Timestamp(timestamp).to_pydatetime()
            spot: float = float(bar["close"])

            # Current IV from VIX model
            iv = vol_model.get_iv(sim_time, cfg.target_dte)

            # ── WARMUP: feed price into strategy (kappa estimator) ──
            if i < cfg.warmup_bars:
                strategy.update(spot_price=spot)
                continue

            # ── OPEN INITIAL STRADDLE if nothing held ──
            if not option_positions:
                option_positions = create_straddle(
                    spot=spot,
                    iv=iv,
                    sim_time=sim_time,
                    target_dte=cfg.target_dte,
                    quantity=cfg.position_size,
                    risk_free_rate=cfg.risk_free_rate,
                    dividend_yield=cfg.dividend_yield,
                )
                entry_cost = fill_sim.option_fill(option_positions, is_entry=True)
                entry_comm = fill_sim.option_commission(option_positions)
                pnl.record_option_entry(option_positions, entry_cost, entry_comm, sim_time)

                strategy.set_option_positions(option_positions)
                strategy.set_equity_position("SPY", equity_qty, spot)

                for pos in option_positions:
                    position_mgr.set_entry_price(pos.symbol, pos.current_price)

                logger.info(
                    f"[bar {i}] Opened straddle: "
                    f"K={option_positions[0].strike}, "
                    f"DTE={cfg.target_dte}, IV={iv:.4f}"
                )
                continue

            # ── REPRICE existing option positions ──
            reprice_positions(
                option_positions,
                spot=spot,
                iv=iv,
                sim_time=sim_time,
                risk_free_rate=cfg.risk_free_rate,
                dividend_yield=cfg.dividend_yield,
            )

            # ── CHECK ROLL / EXIT ──
            dte = compute_dte(option_positions[0].expiry, sim_time)
            if dte <= cfg.roll_dte:
                # Close current straddle
                close_proceeds = fill_sim.option_fill(option_positions, is_entry=False)
                close_comm = fill_sim.option_commission(option_positions)
                pnl.record_option_exit(option_positions, close_proceeds, close_comm, sim_time)

                # Open new straddle
                option_positions = create_straddle(
                    spot=spot,
                    iv=iv,
                    sim_time=sim_time,
                    target_dte=cfg.roll_to_dte,
                    quantity=cfg.position_size,
                    risk_free_rate=cfg.risk_free_rate,
                    dividend_yield=cfg.dividend_yield,
                )
                entry_cost = fill_sim.option_fill(option_positions, is_entry=True)
                entry_comm = fill_sim.option_commission(option_positions)
                pnl.record_option_entry(option_positions, entry_cost, entry_comm, sim_time)

                position_mgr.clear_all_positions()
                for pos in option_positions:
                    position_mgr.set_entry_price(pos.symbol, pos.current_price)

                num_rolls += 1
                logger.info(
                    f"[bar {i}] Rolled straddle: "
                    f"K={option_positions[0].strike}, "
                    f"DTE={cfg.roll_to_dte}, IV={iv:.4f}"
                )

            # ── HEDGING via strategy + MPC ──
            strategy.set_option_positions(option_positions)
            strategy.set_equity_position("SPY", equity_qty, spot)
            rec = strategy.update(spot_price=spot)

            if rec.trade_quantity != 0:
                qty = int(rec.trade_quantity)
                fill_result = fill_sim.equity_fill(spot, qty)
                pnl.record_equity_trade(
                    qty,
                    fill_result.fill_price,
                    fill_result.commission,
                    fill_result.slippage,
                    sim_time,
                )
                equity_qty += qty
                strategy.execute_hedge(rec.trade_quantity)

            # ── MARK TO MARKET ──
            option_mtm = sum(
                pos.current_price * abs(pos.quantity) * pos.multiplier
                for pos in option_positions
            )
            equity_mtm = equity_qty * spot
            portfolio_gamma = getattr(rec, "portfolio_gamma", 0.0)
            pnl.mark_to_market(option_mtm, equity_mtm, spot, sim_time, portfolio_gamma)

            # ── RECORD METRICS ──
            snap = pnl.snapshot(sim_time)
            metrics.record(sim_time, spot, rec, snap, option_positions)

            # Progress logging every 500 bars
            if i > 0 and i % 500 == 0:
                logger.info(
                    f"[bar {i}/{total_bars}] "
                    f"spot={spot:.2f}, P&L=${snap.total_pnl:,.2f}, "
                    f"delta={rec.net_delta:.1f}"
                )

        # -- Build result object ---------------------------------------
        logger.info(
            f"Backtest complete: {total_bars} bars processed, "
            f"{num_rolls} rolls"
        )
        return BacktestResult(
            config=self.config,
            bars=bars,
            records=metrics.records,
            pnl_engine=pnl,
            total_bars=total_bars,
            warmup_bars=cfg.warmup_bars,
            num_rolls=num_rolls,
        )
