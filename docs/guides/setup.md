# Development Setup Guide

This guide walks through setting up the development environment for quant-mpc.

## Prerequisites

- **Python 3.11+** (3.11 or 3.12 recommended)
- **pip** (comes with Python)
- **Git** (for version control)
- **Interactive Brokers TWS or IB Gateway** (for live/paper trading)

## Quick Setup

```bash
# Navigate to project
cd /path/to/quant-mpc

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # macOS/Linux
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -e ".[dev]"

# Verify installation
python -c "import casadi; print(f'CasADi: {casadi.__version__}')"
python -c "import do_mpc; print(f'do-mpc: {do_mpc.__version__}')"
```

## Detailed Steps

### 1. Python Version

Check your Python version:
```bash
python3 --version
```

Should be 3.11.x or 3.12.x. If not, install via:
- **macOS:** `brew install python@3.11`
- **Ubuntu:** `sudo apt install python3.11`
- **Windows:** Download from python.org

### 2. Virtual Environment

Create an isolated environment:
```bash
python3 -m venv venv
source venv/bin/activate
```

Your prompt should now show `(venv)`.

### 3. Install Dependencies

Install the package in editable mode with dev dependencies:
```bash
pip install -e ".[dev]"
```

This installs:
- **Core:** do-mpc, casadi, numpy, pandas, scipy
- **Broker:** ib_insync
- **Dev:** pytest, jupyter, matplotlib, ruff

### 4. Verify Installation

Run the verification script:
```bash
python scripts/verify_setup.py
```

Or manually check each component:
```bash
# CasADi (symbolic math)
python -c "import casadi; print('CasADi OK:', casadi.__version__)"

# do-mpc (MPC framework)
python -c "import do_mpc; print('do-mpc OK:', do_mpc.__version__)"

# ib_insync (IBKR)
python -c "import ib_insync; print('ib_insync OK:', ib_insync.__version__)"

# NumPy
python -c "import numpy; print('NumPy OK:', numpy.__version__)"
```

### 5. IPOPT Solver

do-mpc uses IPOPT for nonlinear optimization. It should be installed automatically with CasADi. Verify:

```bash
python -c "
import casadi as ca
x = ca.MX.sym('x')
nlp = {'x': x, 'f': x**2}
solver = ca.nlpsol('solver', 'ipopt', nlp)
print('IPOPT solver OK')
"
```

## Common Issues

### Issue: CasADi installation fails

**Solution:** Try installing with conda instead:
```bash
conda install -c conda-forge casadi
```

### Issue: IPOPT not found

**Solution:** IPOPT is bundled with CasADi. If missing:
```bash
pip uninstall casadi
pip install casadi --no-cache-dir
```

### Issue: ib_insync import error

**Solution:** Ensure you have the latest version:
```bash
pip install --upgrade ib_insync
```

## IDE Setup

### VS Code / Cursor

1. Install Python extension
2. Select interpreter: `Cmd+Shift+P` → "Python: Select Interpreter" → choose `venv`
3. Install recommended extensions:
   - Python
   - Pylance
   - Jupyter

### PyCharm

1. Open project folder
2. Settings → Project → Python Interpreter → Add → Existing environment → select `venv/bin/python`

## Next Steps

After setup is complete:
1. Run tests: `pytest`
2. Start Jupyter: `jupyter lab`
3. Explore notebooks in `notebooks/`

