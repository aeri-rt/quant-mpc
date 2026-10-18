"""Shared value types every module uses: frozen dataclasses, except the mutable `Order`
(the brokers update its state/fills in place).

Every datetime is tz-aware and comes from the broker or the clock, never from the wall clock
here. Shares and contracts are int and signed where stated; prices are $; multiplier 100.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


@dataclass(frozen=True)
class Quote:
    bid: float
    ask: float
    last: float
    ts: datetime  # broker timestamp (aware)
    market_data_type: int

    @property
    def mid(self) -> float:
        return 0.5 * (self.bid + self.ask)


@dataclass(frozen=True)
class OptionQuote:
    con_id: int
    right: str  # "C" | "P"
    strike: float
    expiry: datetime
    bid: float
    ask: float
    iv: float | None
    delta: float | None
    ts: datetime
    market_data_type: int

    @property
    def mid(self) -> float:
        return 0.5 * (self.bid + self.ask)


@dataclass(frozen=True)
class IbkrGreeks:
    delta: float
    gamma: float
    vega: float | None
    theta: float | None
    iv: float
    S_ref: float
    ts: datetime


class OrderState(str, Enum):
    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    PARTIAL = "PARTIAL"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    INACTIVE = "INACTIVE"
    UNKNOWN = "UNKNOWN"


@dataclass
class Order:
    id: str  # uuid4().hex[:12]
    con_id: int  # stock conId; 0 for a BAG
    sec_type: str  # "STK" | "BAG"
    action: str  # "BUY" | "SELL"
    qty: int  # positive shares or combo units
    limit: float | None
    reason: str  # hedge | close_hedge | entry | hard_exit | roll | roll_refused | exit_confirmed
    #              | max_loss | flatten | rebound (Broker: re-adopted open order on reconnect)
    state: OrderState = OrderState.PENDING
    filled: int = 0
    avg_price: float = 0.0
    leg_fills: dict[int, int] = field(default_factory=dict)  # BAG: con_id -> signed contracts
    error: str = ""
    broker_order_id: int = 0
    perm_id: int = 0
    created: datetime | None = None
    updated: datetime | None = None

    @property
    def is_open(self) -> bool:
        return self.state in (
            OrderState.PENDING,
            OrderState.SUBMITTED,
            OrderState.PARTIAL,
            OrderState.UNKNOWN,
        )


@dataclass(frozen=True)
class Fill:
    exec_id: str
    order_id: str
    con_id: int
    qty: int  # signed: +bought, -sold (shares or contracts)
    price: float
    ts: datetime
    commission: float | None = None
    source: str = "sim"  # "sim" | "broker"


@dataclass(frozen=True)
class Position:
    con_id: int
    symbol: str
    sec_type: str
    qty: int
    avg_cost: float  # STK: $/share; OPT: $/contract / mult (per share of underlying)
    right: str | None = None
    strike: float | None = None
    expiry: datetime | None = None
    mult: int = 1


@dataclass(frozen=True)
class Account:
    nlv: float
    excess_liquidity: float
    buying_power: float
    daily_pnl: float | None
    day_trades_remaining: int | None
    ts: datetime
