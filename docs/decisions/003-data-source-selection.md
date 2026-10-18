# ADR 003: Data Source Selection

## Status
Accepted; IBKR remains the data and execution source. Superseded in part by
[ADR-004](004-v2-thesis-and-hedger.md) (2026-10-17): the client library is now `ib_async`, not
`ib_insync`. The body is kept as written.

## Date
2026-01-04

## Context

The trading system requires real-time and historical market data for:
- SPY spot price (bid, ask, last, volume)
- SPY options chain (all strikes, expiries, Greeks, IV)
- Trade execution and position management

Options considered:

### Option A: Interactive Brokers (IBKR) via `ib_insync`

**Pros:**
- Single integration for data AND execution
- No additional data costs (included with account)
- Real-time streaming via WebSocket
- Options chain with Greeks directly from exchange
- Position and account info in same connection
- `ib_insync` is mature, async-friendly, well-documented

**Cons:**
- Requires running TWS or IB Gateway
- API rate limits (50 messages/sec)
- Greeks may be delayed during market hours
- Connection can be flaky, needs reconnection handling

### Option B: Polygon.io

**Pros:**
- Pure REST/WebSocket, no TWS dependency
- Excellent historical data coverage
- Fast and reliable
- Good documentation

**Cons:**
- Costs money ($99+/month for options data)
- Separate integration from broker
- IV/Greeks not included, must calculate ourselves
- Still need broker connection for execution

### Option C: OpenBB Platform

**Pros:**
- Open source, flexible
- Multiple data sources abstracted
- Good for research/exploration

**Cons:**
- Less suited for real-time production
- Still needs underlying data subscriptions
- Additional abstraction layer

## Decision

**Use IBKR via `ib_insync` as the primary data source**, with the following rationale:

1. **Unified interface**: Same connection handles data, execution, and positions
2. **Cost effective**: No additional data subscriptions needed
3. **Production ready**: `ib_insync` handles async patterns well
4. **Greeks included**: Option Greeks from exchange, saves calculation overhead
5. **We already have IBKR account**: Lower friction to get started

### Implementation Approach

1. **Data Provider abstraction**: Create `DataProvider` interface for future flexibility
2. **IBKR implementation**: `IBKRDataProvider` with connection management
3. **Options chain manager**: Cache and organize options data
4. **Fallback strategy**: If Greeks are stale, calculate using our own Black-Scholes

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     DataProvider (ABC)                      │
├─────────────────────────────────────────────────────────────┤
│  + connect() -> None                                        │
│  + disconnect() -> None                                     │
│  + get_quote(symbol) -> Quote                               │
│  + get_options_chain(symbol, expiry?) -> List[OptionQuote]  │
│  + subscribe_quotes(symbols, callback) -> None              │
│  + get_positions() -> Dict[str, Position]                   │
└─────────────────────────────────────────────────────────────┘
                              △
                              │
         ┌────────────────────┴────────────────────┐
         │                                         │
┌────────┴────────┐                     ┌─────────┴─────────┐
│ IBKRDataProvider │                     │ PolygonDataProvider│
│                  │                     │   (Future)         │
│ - ib_insync      │                     │                    │
│ - TWS/Gateway    │                     │                    │
└──────────────────┘                     └────────────────────┘
```

## Consequences

### Positive
- Faster time to market (single integration)
- Lower operational cost (no data fees)
- Simpler architecture (one source of truth)

### Negative
- Vendor lock-in to IBKR for now
- Dependent on TWS/Gateway running
- May need Polygon later for historical backtesting

### Risks
- IBKR API can be rate-limited under high activity
- Connection drops require careful handling
- Greeks from IBKR may occasionally be stale

## Implementation Notes

- Use `ib_insync.IB` with async event loop
- Implement connection watchdog (reconnect on drop)
- Cache options chain, update on change callbacks
- Log all data events for debugging
- Run TWS/Gateway in a separate process or Docker container

