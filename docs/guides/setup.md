# Development Setup Guide

This guide walks through setting up the development environment for quant-mpc.

## Prerequisites

- **Python 3.11+** (3.11 or 3.12 recommended)
- **pip** (comes with Python)
- **Git** (for version control)
- **Interactive Brokers TWS or IB Gateway** (for live/paper trading)

## Quick Setup

### macOS / Linux

```bash
# Clone repository
git clone https://github.com/aeri-rt/quant-mpc.git
cd quant-mpc

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -e ".[dev]"

# Verify installation
python scripts/verify_setup.py
```

### Windows (PowerShell)

```powershell
# Clone repository
git clone https://github.com/aeri-rt/quant-mpc.git
cd quant-mpc

# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -e ".[dev]"

# Verify installation
python scripts/verify_setup.py
```

### Windows (Command Prompt)

```cmd
# Clone repository
git clone https://github.com/aeri-rt/quant-mpc.git
cd quant-mpc

# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate.bat

# Install dependencies
pip install -e ".[dev]"

# Verify installation
python scripts/verify_setup.py
```

## Detailed Steps

### 1. Python Version

Check your Python version:
```bash
python3 --version   # macOS/Linux
python --version    # Windows
```

Should be 3.11.x or 3.12.x. If not, install via:
- **macOS:** `brew install python@3.11`
- **Ubuntu:** `sudo apt install python3.11`
- **Windows:** Download from [python.org](https://www.python.org/downloads/) - **Check "Add Python to PATH"** during installation

### 2. Virtual Environment

Create an isolated environment:

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Windows (Command Prompt):**
```cmd
python -m venv venv
venv\Scripts\activate.bat
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

## Windows-Specific Issues

### Issue: PowerShell execution policy blocks activation

**Error:** `cannot be loaded because running scripts is disabled on this system`

**Solution:** Open PowerShell as Administrator and run:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Issue: Microsoft Visual C++ Build Tools required

Some packages may require C++ compilation. If you see errors about `Microsoft Visual C++`:

**Solution:** Install [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/):
1. Download the installer
2. Select "Desktop development with C++"
3. Restart terminal after installation

### Issue: Long path names on Windows

**Error:** `FileNotFoundError` or path-related errors with long file paths

**Solution:** Enable long paths in Windows:
1. Open Registry Editor (regedit)
2. Navigate to: `HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\FileSystem`
3. Set `LongPathsEnabled` to `1`
4. Restart your computer

Or via PowerShell (as Administrator):
```powershell
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
```

### Issue: Python not found on Windows

**Solution:** Ensure Python is in your PATH:
1. Reinstall Python from python.org
2. Check "Add Python to PATH" during installation
3. Or manually add `C:\Users\<username>\AppData\Local\Programs\Python\Python311\` to PATH

## IDE Setup

### VS Code / Cursor

1. Install Python extension
2. Select interpreter: 
   - macOS: `Cmd+Shift+P` → "Python: Select Interpreter" → choose `venv`
   - Windows: `Ctrl+Shift+P` → "Python: Select Interpreter" → choose `venv`
3. Install recommended extensions:
   - Python
   - Pylance
   - Jupyter

### PyCharm

1. Open project folder
2. Settings → Project → Python Interpreter → Add → Existing environment → select:
   - macOS/Linux: `venv/bin/python`
   - Windows: `venv\Scripts\python.exe`

## Next Steps

After setup is complete:
1. Run tests: `pytest`
2. Start Jupyter: `jupyter lab`
3. Explore notebooks in `notebooks/`

