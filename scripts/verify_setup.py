#!/usr/bin/env python3
"""
Verify that the development environment is set up correctly.

Run this after installing dependencies to ensure everything works.

Ensure VENV is activated. Activate it with: 

cd /Users/rtrinh/Quant/quant-mpc
source venv/bin/activate
python scripts/verify_setup.py

Usage:
    python scripts/verify_setup.py
"""

import sys
from typing import Tuple

def check_python_version() -> Tuple[bool, str]:
    """Check Python version is 3.11+."""
    version = sys.version_info
    if version.major == 3 and version.minor >= 11:
        return True, f"Python {version.major}.{version.minor}.{version.micro}"
    return False, f"Python {version.major}.{version.minor} (need 3.11+)"


def check_import(module: str, version_attr: str = "__version__") -> Tuple[bool, str]:
    """Try to import a module and get its version."""
    try:
        mod = __import__(module)
        version = getattr(mod, version_attr, "unknown")
        return True, f"{module} {version}"
    except ImportError as e:
        return False, f"{module} FAILED: {e}"


def check_casadi_solver() -> Tuple[bool, str]:
    """Check that CasADi can create an IPOPT solver."""
    try:
        import casadi as ca
        x = ca.MX.sym('x')
        nlp = {'x': x, 'f': x**2}
        solver = ca.nlpsol('solver', 'ipopt', nlp, {'print_time': False, 'ipopt': {'print_level': 0}})
        result = solver(x0=1.0)
        if float(result['x']) < 0.01:  # Should be ~0
            return True, "IPOPT solver working"
        return False, "IPOPT solver gave unexpected result"
    except Exception as e:
        return False, f"IPOPT solver FAILED: {e}"


def check_do_mpc_basic() -> Tuple[bool, str]:
    """Check that do-mpc can create a basic model."""
    try:
        import do_mpc
        import numpy as np
        
        # Create a simple model
        model_type = 'continuous'
        model = do_mpc.model.Model(model_type)
        
        # Define a state
        x = model.set_variable(var_type='_x', var_name='x', shape=(1, 1))
        
        # Define input
        u = model.set_variable(var_type='_u', var_name='u', shape=(1, 1))
        
        # Simple dynamics: dx/dt = u
        model.set_rhs('x', u)
        
        # Setup model
        model.setup()
        
        return True, "do-mpc model creation working"
    except Exception as e:
        return False, f"do-mpc FAILED: {e}"


def main():
    print("=" * 60)
    print("Quant-MPC Environment Verification")
    print("=" * 60)
    print()
    
    checks = [
        ("Python Version", check_python_version),
        ("NumPy", lambda: check_import("numpy")),
        ("SciPy", lambda: check_import("scipy")),
        ("Pandas", lambda: check_import("pandas")),
        ("CasADi", lambda: check_import("casadi")),
        ("do-mpc", lambda: check_import("do_mpc")),
        ("ib_insync", lambda: check_import("ib_insync")),
        ("Pydantic", lambda: check_import("pydantic", "VERSION")),
        ("Loguru", lambda: check_import("loguru")),
        ("IPOPT Solver", check_casadi_solver),
        ("do-mpc Model", check_do_mpc_basic),
    ]
    
    results = []
    for name, check_fn in checks:
        success, message = check_fn()
        results.append((name, success, message))
        status = "✓" if success else "✗"
        print(f"  {status} {name}: {message}")
    
    print()
    print("=" * 60)
    
    # Summary
    passed = sum(1 for _, s, _ in results if s)
    total = len(results)
    
    if passed == total:
        print(f"✓ All {total} checks passed! Environment is ready.")
        print("=" * 60)
        return 0
    else:
        print(f"✗ {passed}/{total} checks passed. Please fix the issues above.")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())

