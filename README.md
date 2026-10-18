# Quant-MPC

A quantitative trading system for SPY options and equities using Model Predictive Control (MPC) and Moving Horizon Estimation (MHE).

## Overview

This system applies control theory to options trading, specifically:
- **Gamma Scalping Strategy**: Buy options for gamma exposure, profit from realized vol > implied vol
- **MPC-Based Delta Hedging**: Optimize when and how much to hedge using predictive control
- **MHE Parameter Estimation**: Real-time estimation of volatility and mean-reversion parameters

### Why Control Theory?

Most quant systems use a "predict → trade" pipeline. This project treats trading as an **optimal control problem**, enabling:
- Unified optimization of position management and execution
- Natural constraint handling (position limits, Greeks bounds)
- "Aggressive in the money" behavior (accelerate trades when prices are favorable)
- Robust adaptation to changing market conditions

Based on: Clinet, Perreton, Reydellet (2021) - "Optimal trading: a model predictive control approach"

## Quick Start

### Prerequisites
- Python 3.11+
- Interactive Brokers account with TWS or IB Gateway
- (Optional) Polygon.io account for historical data

### Installation

**macOS / Linux:**
```bash
git clone https://github.com/aeri-rt/quant-mpc.git
cd quant-mpc
python3 -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
```

**Windows (PowerShell):**
```powershell
git clone https://github.com/aeri-rt/quant-mpc.git
cd quant-mpc
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

See [docs/guides/setup.md](docs/guides/setup.md) for detailed setup instructions and troubleshooting.

### Configuration

1. Edit `config/settings.yaml` for general settings
2. Edit `config/mpc_params.yaml` for MPC tuning
3. Edit `config/strategy_params.yaml` for strategy parameters

### Verify Setup

```bash
# Test that all dependencies are working
python scripts/verify_setup.py

# Test IBKR connection (TWS/Gateway must be running)
python scripts/test_ibkr_connection.py --paper
```

### Running

```bash
# Paper trading (dry run - no execution)
python scripts/run_paper.py --dry-run

# Paper trading with execution
python scripts/run_paper.py --execute

# Custom settings
python scripts/run_paper.py --interval 30 --dte 45 --contracts 5
```

## Project Structure

```
quant-mpc/
├── config/              # Configuration files (YAML)
│   ├── settings.yaml    # General settings (broker, data source)
│   ├── mpc_params.yaml  # MPC controller tuning
│   ├── strategy_params.yaml  # Strategy parameters
│   └── logging.yaml     # Logging configuration
├── src/                 # Main source code
│   ├── core/            # Core types, state machine, exceptions
│   │   ├── types.py         # Quote, Position, MarketState, etc.
│   │   ├── state_machine.py # Trading lifecycle state machine
│   │   └── exceptions.py    # Custom exception hierarchy
│   ├── data/            # Data providers
│   │   ├── provider.py      # Abstract DataProvider interface
│   │   ├── ibkr_provider.py # IBKR implementation
│   │   ├── options_chain.py # Options chain manager
│   │   └── trade_store.py   # SQLite trade persistence
│   ├── models/          # Financial models
│   │   ├── black_scholes.py # BS pricing (numeric + CasADi symbolic)
│   │   └── greeks.py        # Portfolio Greeks aggregation
│   ├── control/         # MPC implementation
│   │   └── mpc_controller.py # DeltaHedgeMPC with do-mpc
│   ├── strategy/        # Trading strategies
│   │   ├── gamma_scalping.py # GammaScalpingStrategy
│   │   └── position_manager.py # Roll/exit logic
│   ├── execution/       # Order execution
│   │   └── order_manager.py # Order lifecycle, straddle orders
│   ├── risk/            # Risk management
│   │   └── risk_manager.py  # Position/delta/gamma limits, kill switch
│   └── utils/           # Utilities
│       ├── logging.py       # Structured logging
│       ├── config.py        # YAML config loader
│       ├── metrics.py       # Session metrics tracker
│       ├── validation.py    # Data validation (NaN/Inf guards)
│       └── helpers.py       # Safe type conversions, constants
├── scripts/             # Runnable scripts
│   ├── test_ibkr_connection.py  # Test IBKR connectivity
│   └── run_paper.py     # Paper trading runner
├── notebooks/           # Jupyter notebooks
│   ├── 01_greeks_engine_demo.ipynb  # Greeks engine demo
│   └── 02_mpc_validation.ipynb      # MPC validation tests
├── docs/                # Documentation
│   ├── decisions/       # Architecture Decision Records
│   └── design/          # Design documents
├── logs/                # Log files (gitignored)
└── data/                # Local data storage (gitignored)
```

## Current Status (V1)

| Component | Status | Notes |
|-----------|--------|-------|
| Black-Scholes Greeks | ✅ Complete | Numeric + CasADi symbolic |
| Portfolio Greeks | ✅ Complete | Aggregation with dividend yield |
| MPC Controller | ✅ Complete | Validated with 4 test scenarios |
| IBKR Data Provider | ✅ Complete | Quotes, options, positions, streaming, reconnect |
| Options Chain Manager | ✅ Complete | Caching, queries, ATM straddle selection |
| Gamma Scalping Strategy | ✅ Complete | Full pipeline integration |
| Paper Trading Script | ✅ Complete | Dry-run + execution, state machine lifecycle |
| Execution Layer | ✅ Complete | Order management, straddle orders, fill tracking |
| Risk Management | ✅ Complete | Position/delta/gamma limits, loss limits, kill switch |
| Data Validation | ✅ Complete | NaN/Inf/zero guards on quotes, Greeks, prices |
| Trade Persistence | ✅ Complete | SQLite session/order/fill storage |
| State Machine | ✅ Complete | Formal lifecycle (init -> run -> stop) |
| Position Recovery | ✅ Complete | Recover existing IBKR positions on restart |
| Auto Roll/Exit | ✅ Complete | Auto-execute roll and exit signals |
| Custom Exceptions | ✅ Complete | Typed exception hierarchy throughout |
| Backtest Framework | 🚧 V2 | Historical simulation |
| MHE Estimator | 🚧 V2 | Real-time volatility/kappa estimation |

## Documentation

See [docs/README.md](docs/README.md) for full documentation, including:
- [Architecture Decision Records](docs/decisions/)
- [Design Documents](docs/design/)
- [Guides](docs/guides/)

## Key Technologies

| Component | Technology |
|-----------|------------|
| MPC/MHE | [do-mpc](https://www.do-mpc.com/) |
| Symbolic Math | [CasADi](https://web.casadi.org/) |
| Broker | [ib_insync](https://ib-insync.readthedocs.io/) (IBKR) |
| Data | IBKR (live), Polygon.io (historical) |

## Development

```bash
# Run tests
pytest

# Run linter
ruff check src/

# Format code
ruff format src/
```

## Disclaimer

This software is for educational and research purposes only. Trading involves substantial risk of loss. Past performance does not guarantee future results. Use at your own risk.

## License

MIT License - see LICENSE file

