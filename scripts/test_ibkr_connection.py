#!/usr/bin/env python3
"""
IBKR Connection Test Script

Tests connection to Interactive Brokers TWS/Gateway and fetches:
- SPY quote
- SPY options chain (nearest expiry)
- Account positions

Prerequisites:
1. TWS or IB Gateway must be running
2. API connections must be enabled (Configure -> API -> Settings)
3. Socket port should match (default: 7497 for paper, 7496 for live)

Usage:
    python scripts/test_ibkr_connection.py [--paper] [--live]
"""

import asyncio
import argparse
from datetime import datetime, timedelta

# Ensure we can import from src
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import IBKRDataProvider, DataProviderConfig, OptionsChainManager
from src.core.types import OptionType


async def test_connection(port: int = 7497, client_id: int = 1):
    """Test IBKR connection and data retrieval."""
    
    print("="*60)
    print("IBKR CONNECTION TEST")
    print("="*60)
    print()
    
    # Configure provider
    config = DataProviderConfig(
        host="127.0.0.1",
        port=port,
        client_id=client_id,
        connect_timeout=15.0,
        use_delayed_data=True,  # Set False if you have real-time data
    )
    
    provider = IBKRDataProvider(config)
    
    try:
        # Connect
        print(f"Connecting to IBKR at 127.0.0.1:{port}...")
        await provider.connect()
        print("✓ Connected!")
        print()
        
        # Get SPY quote
        print("Fetching SPY quote...")
        spy_quote = await provider.get_quote("SPY")
        print(f"✓ SPY Quote:")
        print(f"    Bid: ${spy_quote.bid:.2f}")
        print(f"    Ask: ${spy_quote.ask:.2f}")
        print(f"    Mid: ${spy_quote.mid:.2f}")
        print(f"    Last: ${spy_quote.last:.2f}")
        print()
        
        # Get account info
        print("Fetching account info...")
        account = await provider.get_account_info()
        print("✓ Account Info:")
        for key, value in account.items():
            print(f"    {key}: ${value:,.2f}")
        print()
        
        # Get positions
        print("Fetching positions...")
        positions = await provider.get_positions()
        if positions:
            print(f"✓ Positions ({len(positions)} total):")
            for symbol, pos in list(positions.items())[:5]:  # Show first 5
                print(f"    {symbol}: {pos.quantity} @ ${pos.avg_cost:.2f}")
        else:
            print("✓ No positions")
        print()
        
        # Get options chain
        print("Fetching SPY options chain (nearest expiry, ATM ±$10)...")
        target_expiry = datetime.now() + timedelta(days=30)
        
        options = await provider.get_options_chain(
            underlying="SPY",
            expiry=None,  # Let it find nearest
            strike_range=(spy_quote.mid - 10, spy_quote.mid + 10),
        )
        
        if options:
            print(f"✓ Options Chain ({len(options)} contracts):")
            
            # Use OptionsChainManager to organize
            chain_mgr = OptionsChainManager("SPY")
            chain_mgr.update_chain(options)
            chain_mgr.update_spot(spy_quote.mid)
            
            # Get ATM straddle
            atm_call, atm_put = chain_mgr.get_atm_straddle(target_dte=30)
            
            if atm_call and atm_put:
                print()
                print(f"  ATM Straddle (Strike: ${atm_call.strike:.0f}):")
                print(f"    Call: ${atm_call.mid:.2f} (Δ={atm_call.delta:.3f}, IV={atm_call.iv:.1%})")
                print(f"    Put:  ${atm_put.mid:.2f} (Δ={atm_put.delta:.3f}, IV={atm_put.iv:.1%})")
                print(f"    Total Premium: ${(atm_call.mid + atm_put.mid):.2f}")
            
            # Show chain stats
            stats = chain_mgr.get_stats()
            print()
            print(f"  Chain Stats:")
            print(f"    Expiries: {stats['n_expiries']}")
            print(f"    Calls: {stats['n_calls']}")
            print(f"    Puts: {stats['n_puts']}")
        else:
            print("✗ No options data (may need real-time data subscription)")
        print()
        
        print("="*60)
        print("✓ ALL TESTS PASSED")
        print("="*60)
        
    except ConnectionError as e:
        print(f"✗ Connection failed: {e}")
        print()
        print("Troubleshooting:")
        print("1. Make sure TWS or IB Gateway is running")
        print("2. Check that API connections are enabled:")
        print("   TWS: File -> Global Configuration -> API -> Settings")
        print("   - Enable ActiveX and Socket Clients")
        print(f"   - Socket port should be {port}")
        print("3. Check firewall settings")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        await provider.disconnect()
        print()
        print("Disconnected.")


def main():
    parser = argparse.ArgumentParser(description="Test IBKR connection")
    parser.add_argument("--paper", action="store_true", help="Use paper trading port (7497)")
    parser.add_argument("--live", action="store_true", help="Use live trading port (7496)")
    parser.add_argument("--port", type=int, help="Custom port")
    parser.add_argument("--client-id", type=int, default=1, help="Client ID")
    
    args = parser.parse_args()
    
    if args.port:
        port = args.port
    elif args.live:
        port = 7496
    else:
        port = 7497  # Default to paper
    
    asyncio.run(test_connection(port=port, client_id=args.client_id))


if __name__ == "__main__":
    main()

