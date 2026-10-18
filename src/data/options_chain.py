"""
Options Chain Manager

Organizes and manages options chain data for easy access by strategies.
Provides utilities for finding ATM options, filtering by criteria,
and converting between OptionQuote and OptionPosition.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

from src.core.types import OptionQuote, OptionType
from src.models.greeks import OptionPosition, PositionType
from src.utils.logging import get_logger
from src.utils.validation import validate_option_quote

logger = get_logger("options_chain")


@dataclass
class OptionChainEntry:
    """Entry in the options chain cache."""
    quote: OptionQuote
    last_updated: datetime = field(default_factory=datetime.now)
    
    @property
    def age_seconds(self) -> float:
        """How old this entry is in seconds."""
        return (datetime.now() - self.last_updated).total_seconds()


@dataclass
class ExpiryChain:
    """All options for a single expiry date."""
    expiry: datetime
    calls: Dict[float, OptionChainEntry] = field(default_factory=dict)  # strike -> entry
    puts: Dict[float, OptionChainEntry] = field(default_factory=dict)


class OptionsChainManager:
    """
    Manages options chain data for an underlying.
    
    Features:
    - Caches option quotes by expiry and strike
    - Finds ATM options efficiently
    - Filters by DTE, moneyness, liquidity
    - Converts quotes to positions
    - Tracks staleness of data
    
    Usage:
        manager = OptionsChainManager("SPY")
        
        # Update with data from provider
        manager.update_chain(option_quotes)
        
        # Get ATM straddle
        call, put = manager.get_atm_straddle(spot_price=500, target_dte=30)
        
        # Convert to positions
        positions = manager.to_option_positions([(call, 10), (put, 10)])
    """
    
    def __init__(self, underlying: str, max_age_seconds: float = 60.0):
        """
        Initialize the chain manager.
        
        Args:
            underlying: Underlying symbol (e.g., "SPY")
            max_age_seconds: Maximum age for quotes before considered stale
        """
        self.underlying = underlying
        self.max_age_seconds = max_age_seconds
        
        # Chain organized by expiry
        self._chains: Dict[datetime, ExpiryChain] = {}
        
        # Quick lookup by symbol
        self._by_symbol: Dict[str, OptionChainEntry] = {}
        
        # Current spot price (updated externally)
        self._spot_price: float = 0.0
        
        logger.info(f"OptionsChainManager initialized for {underlying}")
    
    # -------------------------------------------------------------------------
    # Data Updates
    # -------------------------------------------------------------------------
    
    def update_chain(self, quotes: List[OptionQuote]) -> None:
        """
        Update the chain with new option quotes.
        
        Args:
            quotes: List of OptionQuote objects
        """
        updated_count = 0
        
        for quote in quotes:
            if quote.underlying != self.underlying:
                logger.warning(f"Ignoring quote for {quote.underlying}, expected {self.underlying}")
                continue
            
            # Validate option quote before caching
            quote = validate_option_quote(quote)
            
            entry = OptionChainEntry(quote=quote, last_updated=datetime.now())
            
            # Get or create expiry chain
            expiry_key = quote.expiry.replace(hour=0, minute=0, second=0, microsecond=0)
            if expiry_key not in self._chains:
                self._chains[expiry_key] = ExpiryChain(expiry=expiry_key)
            
            chain = self._chains[expiry_key]
            
            # Add to appropriate dict
            if quote.option_type == OptionType.CALL:
                chain.calls[quote.strike] = entry
            else:
                chain.puts[quote.strike] = entry
            
            # Add to symbol lookup
            self._by_symbol[quote.symbol] = entry
            updated_count += 1
        
        logger.debug(f"Updated {updated_count} option quotes")
    
    def update_spot(self, spot_price: float) -> None:
        """Update the current spot price."""
        self._spot_price = spot_price
    
    def clear_stale(self) -> int:
        """
        Remove stale entries from the cache.
        
        Returns:
            Number of entries removed
        """
        removed = 0
        
        for expiry_key, chain in list(self._chains.items()):
            # Remove stale calls
            stale_strikes = [
                strike for strike, entry in chain.calls.items()
                if entry.age_seconds > self.max_age_seconds
            ]
            for strike in stale_strikes:
                symbol = chain.calls[strike].quote.symbol
                del chain.calls[strike]
                self._by_symbol.pop(symbol, None)
                removed += 1
            
            # Remove stale puts
            stale_strikes = [
                strike for strike, entry in chain.puts.items()
                if entry.age_seconds > self.max_age_seconds
            ]
            for strike in stale_strikes:
                symbol = chain.puts[strike].quote.symbol
                del chain.puts[strike]
                self._by_symbol.pop(symbol, None)
                removed += 1
            
            # Remove empty expiry chains
            if not chain.calls and not chain.puts:
                del self._chains[expiry_key]
        
        if removed:
            logger.debug(f"Cleared {removed} stale entries")
        
        return removed
    
    # -------------------------------------------------------------------------
    # Queries
    # -------------------------------------------------------------------------
    
    def get_quote(self, symbol: str) -> Optional[OptionQuote]:
        """Get option quote by symbol."""
        entry = self._by_symbol.get(symbol)
        return entry.quote if entry else None
    
    def get_by_strike(
        self,
        expiry: datetime,
        strike: float,
        option_type: OptionType,
    ) -> Optional[OptionQuote]:
        """Get option quote by expiry, strike, and type."""
        expiry_key = expiry.replace(hour=0, minute=0, second=0, microsecond=0)
        chain = self._chains.get(expiry_key)
        
        if not chain:
            return None
        
        entries = chain.calls if option_type == OptionType.CALL else chain.puts
        entry = entries.get(strike)
        
        return entry.quote if entry else None
    
    def get_available_expiries(self) -> List[datetime]:
        """Get list of available expiry dates, sorted ascending."""
        return sorted(self._chains.keys())
    
    def get_available_strikes(self, expiry: datetime) -> List[float]:
        """Get list of available strikes for an expiry."""
        expiry_key = expiry.replace(hour=0, minute=0, second=0, microsecond=0)
        chain = self._chains.get(expiry_key)
        
        if not chain:
            return []
        
        # Union of call and put strikes
        strikes = set(chain.calls.keys()) | set(chain.puts.keys())
        return sorted(strikes)
    
    def get_expiry_chain(self, expiry: datetime) -> List[OptionQuote]:
        """Get all options for a specific expiry."""
        expiry_key = expiry.replace(hour=0, minute=0, second=0, microsecond=0)
        chain = self._chains.get(expiry_key)
        
        if not chain:
            return []
        
        quotes = []
        for entry in chain.calls.values():
            quotes.append(entry.quote)
        for entry in chain.puts.values():
            quotes.append(entry.quote)
        
        return quotes
    
    # -------------------------------------------------------------------------
    # Option Selection
    # -------------------------------------------------------------------------
    
    def get_nearest_expiry(self, target_dte: int = 30) -> Optional[datetime]:
        """
        Get expiry nearest to target DTE (days to expiration).
        
        Args:
            target_dte: Target days to expiration
            
        Returns:
            Nearest expiry datetime, or None if no expiries
        """
        if not self._chains:
            return None
        
        target_date = datetime.now() + timedelta(days=target_dte)
        
        return min(
            self._chains.keys(),
            key=lambda exp: abs((exp - target_date).days),
        )
    
    def get_atm_strike(
        self,
        expiry: datetime,
        spot_price: Optional[float] = None,
    ) -> Optional[float]:
        """
        Get the at-the-money strike for an expiry.
        
        Args:
            expiry: Expiration date
            spot_price: Spot price (uses cached if None)
            
        Returns:
            Nearest strike to spot, or None
        """
        spot = spot_price or self._spot_price
        if spot <= 0:
            return None
        
        strikes = self.get_available_strikes(expiry)
        if not strikes:
            return None
        
        return min(strikes, key=lambda s: abs(s - spot))
    
    def get_atm_straddle(
        self,
        spot_price: Optional[float] = None,
        target_dte: int = 30,
    ) -> Tuple[Optional[OptionQuote], Optional[OptionQuote]]:
        """
        Get ATM call and put for a straddle.
        
        Args:
            spot_price: Spot price (uses cached if None)
            target_dte: Target days to expiration
            
        Returns:
            Tuple of (call, put) quotes, or (None, None)
        """
        spot = spot_price or self._spot_price
        
        # Try multiple expiries if needed
        for dte_offset in [0, 7, -7, 14, -14]:
            adjusted_dte = max(1, target_dte + dte_offset)
            expiry = self.get_nearest_expiry(adjusted_dte)
            if not expiry:
                continue
            
            strikes = self.get_available_strikes(expiry)
            if not strikes:
                continue
            
            # Sort strikes by distance from spot
            sorted_strikes = sorted(strikes, key=lambda s: abs(s - spot))
            
            # Try nearest strikes until we find one with both call and put
            for strike in sorted_strikes[:10]:  # Try up to 10 nearest strikes
                call = self.get_by_strike(expiry, strike, OptionType.CALL)
                put = self.get_by_strike(expiry, strike, OptionType.PUT)
                
                if call and put:
                    logger.info(f"Found straddle at strike ${strike:.0f}, expiry {expiry.strftime('%Y-%m-%d')}")
                    return call, put
            
            logger.debug(f"No complete straddle found for expiry {expiry.strftime('%Y-%m-%d')}")
        
        logger.warning(f"Could not find straddle in any expiry (spot=${spot:.2f}, target_dte={target_dte})")
        return None, None
    
    def filter_by_criteria(
        self,
        expiry: Optional[datetime] = None,
        min_dte: int = 0,
        max_dte: int = 365,
        min_delta: float = 0.0,
        max_delta: float = 1.0,
        min_iv: float = 0.0,
        max_iv: float = 10.0,
        min_bid: float = 0.01,
        option_type: Optional[OptionType] = None,
    ) -> List[OptionQuote]:
        """
        Filter options by various criteria.
        
        Args:
            expiry: Specific expiry (if None, use all)
            min_dte, max_dte: DTE range
            min_delta, max_delta: Absolute delta range
            min_iv, max_iv: IV range
            min_bid: Minimum bid price (liquidity filter)
            option_type: CALL, PUT, or None for both
            
        Returns:
            List of matching OptionQuote objects
        """
        results = []
        
        for expiry_key, chain in self._chains.items():
            # Check expiry
            if expiry and expiry_key != expiry.replace(hour=0, minute=0, second=0, microsecond=0):
                continue
            
            dte = (expiry_key - datetime.now()).days
            if dte < min_dte or dte > max_dte:
                continue
            
            # Check calls
            if option_type is None or option_type == OptionType.CALL:
                for entry in chain.calls.values():
                    q = entry.quote
                    if self._matches_criteria(q, min_delta, max_delta, min_iv, max_iv, min_bid):
                        results.append(q)
            
            # Check puts
            if option_type is None or option_type == OptionType.PUT:
                for entry in chain.puts.values():
                    q = entry.quote
                    if self._matches_criteria(q, min_delta, max_delta, min_iv, max_iv, min_bid):
                        results.append(q)
        
        return results
    
    def _matches_criteria(
        self,
        quote: OptionQuote,
        min_delta: float,
        max_delta: float,
        min_iv: float,
        max_iv: float,
        min_bid: float,
    ) -> bool:
        """Check if quote matches filter criteria."""
        import math
        
        # Guard against NaN in comparisons
        abs_delta = abs(quote.delta) if not math.isnan(quote.delta) else 0.0
        iv = quote.iv if not math.isnan(quote.iv) else 0.0
        bid = quote.bid if not math.isnan(quote.bid) else 0.0
        
        if abs_delta < min_delta or abs_delta > max_delta:
            return False
        
        if iv < min_iv or iv > max_iv:
            return False
        
        if bid < min_bid:
            return False
        
        return True
    
    # -------------------------------------------------------------------------
    # Conversion
    # -------------------------------------------------------------------------
    
    def to_option_position(
        self,
        quote: OptionQuote,
        quantity: int,
        spot_price: Optional[float] = None,
    ) -> OptionPosition:
        """
        Convert an OptionQuote to an OptionPosition for the Greeks calculator.
        
        Args:
            quote: Option quote
            quantity: Number of contracts (+ long, - short)
            spot_price: Underlying price (uses cached if None)
            
        Returns:
            OptionPosition ready for Greeks calculation
        """
        spot = spot_price or self._spot_price
        
        return OptionPosition(
            symbol=quote.symbol,
            position_type=PositionType.CALL if quote.option_type == OptionType.CALL else PositionType.PUT,
            strike=quote.strike,
            expiry=quote.expiry,
            quantity=quantity,
            multiplier=100,
            current_price=quote.mid,
            underlying_price=spot,
            iv=quote.iv if quote.iv > 0 else 0.20,  # Default IV if missing
        )
    
    def to_option_positions(
        self,
        quotes_and_quantities: List[Tuple[OptionQuote, int]],
        spot_price: Optional[float] = None,
    ) -> List[OptionPosition]:
        """
        Convert multiple quotes to positions.
        
        Args:
            quotes_and_quantities: List of (quote, quantity) tuples
            spot_price: Underlying price
            
        Returns:
            List of OptionPosition objects
        """
        return [
            self.to_option_position(quote, qty, spot_price)
            for quote, qty in quotes_and_quantities
        ]
    
    # -------------------------------------------------------------------------
    # Stats
    # -------------------------------------------------------------------------
    
    def get_stats(self) -> dict:
        """Get statistics about the current chain cache."""
        total_calls = sum(len(c.calls) for c in self._chains.values())
        total_puts = sum(len(c.puts) for c in self._chains.values())
        
        # Find oldest entry
        oldest_age = 0.0
        for chain in self._chains.values():
            for entry in list(chain.calls.values()) + list(chain.puts.values()):
                oldest_age = max(oldest_age, entry.age_seconds)
        
        return {
            "underlying": self.underlying,
            "n_expiries": len(self._chains),
            "n_calls": total_calls,
            "n_puts": total_puts,
            "total_contracts": total_calls + total_puts,
            "oldest_entry_seconds": oldest_age,
            "spot_price": self._spot_price,
        }

