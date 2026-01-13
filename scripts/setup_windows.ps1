#
# Setup script for Windows (PowerShell)
# Creates virtual environment and installs all dependencies
#
# Usage:
#   1. Open PowerShell as Administrator (if needed for execution policy)
#   2. Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
#   3. .\scripts\setup_windows.ps1
#

$ErrorActionPreference = "Stop"

Write-Host "============================================================"
Write-Host "Quant-MPC Setup Script (Windows)"
Write-Host "============================================================"
Write-Host ""

# Get project root
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir

Set-Location $ProjectRoot
Write-Host "Project root: $ProjectRoot"
Write-Host ""

# Check Python version
Write-Host "Checking Python version..."

$PythonCmd = $null
if (Get-Command python -ErrorAction SilentlyContinue) {
    $PythonCmd = "python"
} elseif (Get-Command python3 -ErrorAction SilentlyContinue) {
    $PythonCmd = "python3"
} else {
    Write-Host "X Python not found. Please install Python 3.11 or later from python.org" -ForegroundColor Red
    Write-Host "  Make sure to check 'Add Python to PATH' during installation"
    exit 1
}

$PythonVersion = & $PythonCmd --version 2>&1
$VersionMatch = [regex]::Match($PythonVersion, "(\d+)\.(\d+)")
$Major = [int]$VersionMatch.Groups[1].Value
$Minor = [int]$VersionMatch.Groups[2].Value

Write-Host "  Found $PythonVersion"

if ($Major -lt 3 -or ($Major -eq 3 -and $Minor -lt 11)) {
    Write-Host "X Python 3.11 or later required. Found $PythonVersion" -ForegroundColor Red
    Write-Host "  Download from: https://www.python.org/downloads/"
    exit 1
}
Write-Host "OK Python version OK" -ForegroundColor Green
Write-Host ""

# Create virtual environment
Write-Host "Creating virtual environment..."
if (Test-Path "venv") {
    Write-Host "  Existing venv found. Removing..."
    Remove-Item -Recurse -Force "venv"
}

& $PythonCmd -m venv venv
Write-Host "OK Virtual environment created" -ForegroundColor Green
Write-Host ""

# Activate virtual environment
Write-Host "Activating virtual environment..."
& ".\venv\Scripts\Activate.ps1"
Write-Host "OK Activated: $(Get-Command python | Select-Object -ExpandProperty Source)" -ForegroundColor Green
Write-Host ""

# Upgrade pip
Write-Host "Upgrading pip..."
& python -m pip install --upgrade pip --quiet
Write-Host "OK pip upgraded" -ForegroundColor Green
Write-Host ""

# Install dependencies
Write-Host "Installing dependencies..."
Write-Host "  This may take a few minutes..."
& pip install -e ".[dev]" --quiet
Write-Host "OK Dependencies installed" -ForegroundColor Green
Write-Host ""

# Verify installation
Write-Host "Verifying installation..."
Write-Host ""

$Failed = 0

# Core packages
$Packages = @("numpy", "scipy", "pandas", "casadi", "do_mpc", "ib_insync", "pydantic", "loguru")

foreach ($pkg in $Packages) {
    $result = & python -c "import $pkg; print(getattr($pkg, '__version__', 'OK'))" 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  OK $pkg : $result" -ForegroundColor Green
    } else {
        Write-Host "  X $pkg : FAILED" -ForegroundColor Red
        $Failed++
    }
}

# Test IPOPT solver
$ipoptTest = @"
import casadi as ca
x = ca.MX.sym('x')
nlp = {'x': x, 'f': x**2}
solver = ca.nlpsol('solver', 'ipopt', nlp, {'ipopt.print_level': 0, 'print_time': 0})
print('OK')
"@

$result = & python -c $ipoptTest 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "  OK IPOPT solver: OK" -ForegroundColor Green
} else {
    Write-Host "  X IPOPT solver: FAILED" -ForegroundColor Red
    $Failed++
}

# Test do-mpc model
$dompcTest = @"
import do_mpc
model = do_mpc.model.Model('discrete')
x = model.set_variable('_x', 'x')
model.set_rhs('x', x)
model.setup()
print('OK')
"@

$result = & python -c $dompcTest 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "  OK do-mpc model: OK" -ForegroundColor Green
} else {
    Write-Host "  X do-mpc model: FAILED" -ForegroundColor Red
    $Failed++
}

Write-Host ""

if ($Failed -eq 0) {
    Write-Host "============================================================"
    Write-Host "OK Setup complete! All packages installed successfully." -ForegroundColor Green
    Write-Host "============================================================"
    Write-Host ""
    Write-Host "To use the environment:"
    Write-Host "  .\venv\Scripts\Activate.ps1"
    Write-Host ""
    Write-Host "To run paper trading:"
    Write-Host "  python scripts/run_paper.py --dry-run"
    Write-Host ""
    Write-Host "To verify setup anytime:"
    Write-Host "  python scripts/verify_setup.py"
    Write-Host "============================================================"
} else {
    Write-Host "============================================================"
    Write-Host "X Setup completed with $Failed failures." -ForegroundColor Red
    Write-Host "  Please check the errors above."
    Write-Host "============================================================"
    exit 1
}
