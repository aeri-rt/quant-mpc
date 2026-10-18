#!/bin/bash
#
# Setup script for macOS/Linux
# Creates virtual environment and installs all dependencies
#
# Usage:
#   chmod +x scripts/setup_macos.sh
#   ./scripts/setup_macos.sh
#

set -e  # Exit on error

echo "============================================================"
echo "Quant-MPC Setup Script (macOS/Linux)"
echo "============================================================"
echo ""

# Get the script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"
echo "Project root: $PROJECT_ROOT"
echo ""

# Check Python version
echo "Checking Python version..."
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo "✗ Python not found. Please install Python 3.12 or later."
    exit 1
fi

PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | cut -d' ' -f2)
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)

echo "  Found Python $PYTHON_VERSION"

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 12 ]); then
    echo "✗ Python 3.12 or later required. Found $PYTHON_VERSION"
    echo "  Install with: brew install python@3.12"
    exit 1
fi
echo "✓ Python version OK"
echo ""

# Create virtual environment
echo "Creating virtual environment..."
if [ -d "venv" ]; then
    echo "  Existing venv found. Removing..."
    rm -rf venv
fi

$PYTHON_CMD -m venv venv
echo "✓ Virtual environment created"
echo ""

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate
echo "✓ Activated: $(which python)"
echo ""

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip --quiet
echo "✓ pip upgraded"
echo ""

# Install dependencies (runtime + dev, read straight from pyproject.toml)
echo "Installing dependencies..."
echo "  This may take a few minutes..."
pip install --quiet $(python -c "import tomllib;d=tomllib.load(open('pyproject.toml','rb'))['project'];print(' '.join(d['dependencies']+d['optional-dependencies']['dev']))")
echo "✓ Dependencies installed"
echo ""

# Verify installation
echo "Verifying installation..."
echo ""

FAILED=0

# Core packages
for pkg in numpy scipy pandas ib_async pydantic loguru exchange_calendars; do
    if python -c "import $pkg" 2>/dev/null; then
        VERSION=$(python -c "import $pkg; print(getattr($pkg, '__version__', 'OK'))")
        echo "  ✓ $pkg: $VERSION"
    else
        echo "  ✗ $pkg: FAILED"
        FAILED=$((FAILED + 1))
    fi
done

echo ""

if [ $FAILED -eq 0 ]; then
    echo "============================================================"
    echo "✓ Setup complete! All packages installed successfully."
    echo "============================================================"
    echo ""
    echo "To use the environment:"
    echo "  source venv/bin/activate"
    echo ""
    echo "To run paper trading:"
    echo "  python scripts/run_paper.py"
    echo ""
    echo "To verify setup anytime:"
    echo "  pytest -q"
    echo "============================================================"
else
    echo "============================================================"
    echo "✗ Setup completed with $FAILED failures."
    echo "  Please check the errors above."
    echo "============================================================"
    exit 1
fi
