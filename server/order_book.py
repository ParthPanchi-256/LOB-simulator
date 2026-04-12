# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Limit Order Book with price-time priority matching engine.

Provides a realistic simulation of exchange micro-structure including:
- Price-time priority matching (FIFO within each price level)
- Partial fills
- Limit and market order support
- Order cancellation
- Top-N book snapshot queries
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

from sortedcontainers import SortedDict


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


class Side(str, Enum):
    BUY = "buy"
    SELL = "sell"


@dataclass
class Order:
    """A single order resting on the book."""

    id: str
    side: Side
    price: float
    quantity: int
    timestamp: float
    owner: str  # "agent" or background-trader id

    # Mutable — tracks remaining quantity after partial fills
    remaining: int = field(init=False)

    def __post_init__(self) -> None:
        self.remaining = self.quantity


@dataclass(frozen=True)
class Trade:
    """Record of a matched trade."""

    price: float
    quantity: int
    buyer_order_id: str
    seller_order_id: str
    aggressor_side: Side
    timestamp: float


# ---------------------------------------------------------------------------
# Order Book
# ---------------------------------------------------------------------------


class OrderBook:
    """Continuous double-auction limit order book.

    Bids are stored highest-price-first, asks lowest-price-first.
    Within each price level, orders are matched FIFO (price-time priority).
    """

    def __init__(self, tick_size: float = 0.01) -> None:
        self.tick_size = tick_size

        # price → list[Order]  (sorted by insertion order = time priority)
        # Bids: highest price first  →  use negated keys so SortedDict gives
        #        ascending order on *negative* prices = descending real price.
        # Asks: lowest price first   →  natural ascending order.
        self._bids: SortedDict = SortedDict()  # key = -price
        self._asks: SortedDict = SortedDict()  # key = +price

        # Fast lookup: order_id → Order
        self._orders: Dict[str, Order] = {}

        # Monotonic counter for unique order IDs
        self._id_counter = itertools.count(1)

        # Trade log (all trades in the current step — cleared each step)
        self.last_trades: List[Trade] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_limit_order(
        self,
        side: Side,
        price: float,
        quantity: int,
        owner: str = "agent",
    ) -> Tuple[str, List[Trade]]:
        """Submit a limit order.  Returns ``(order_id, trades)``."""
        price = self._round_price(price)
        order = Order(
            id=self._next_id(),
            side=side,
            price=price,
            quantity=quantity,
            timestamp=time.monotonic(),
            owner=owner,
        )

        trades = self._match(order)

        # If the order still has remaining quantity, rest it on the book
        if order.remaining > 0:
            self._insert(order)

        return order.id, trades

    def add_market_order(
        self,
        side: Side,
        quantity: int,
        owner: str = "agent",
    ) -> List[Trade]:
        """Submit a market order.  Executes immediately against resting liquidity."""
        order = Order(
            id=self._next_id(),
            side=side,
            # Use extreme price to guarantee matching
            price=float("inf") if side == Side.BUY else 0.0,
            quantity=quantity,
            timestamp=time.monotonic(),
            owner=owner,
        )
        trades = self._match(order)
        # Market orders never rest — any unfilled portion is discarded.
        return trades

    def cancel_order(self, order_id: str) -> bool:
        """Cancel a resting order.  Returns True if found and removed."""
        order = self._orders.pop(order_id, None)
        if order is None:
            return False

        book = self._bids if order.side == Side.BUY else self._asks
        key = -order.price if order.side == Side.BUY else order.price

        if key in book:
            queue: List[Order] = book[key]
            try:
                queue.remove(order)
            except ValueError:
                pass
            if not queue:
                del book[key]
        return True

    def get_top_n(
        self, n: int = 10
    ) -> Tuple[List[float], List[int], List[float], List[int]]:
        """Return the top *n* levels of the book.

        Returns:
            (bid_prices, bid_volumes, ask_prices, ask_volumes)
            Each list has at most *n* entries.  Bids are sorted best (highest)
            first; asks are sorted best (lowest) first.
        """
        bid_prices: List[float] = []
        bid_volumes: List[int] = []
        for neg_price in self._bids.islice(stop=n):
            queue: List[Order] = self._bids[neg_price]
            vol = sum(o.remaining for o in queue)
            if vol > 0:
                bid_prices.append(-neg_price)
                bid_volumes.append(vol)

        ask_prices: List[float] = []
        ask_volumes: List[int] = []
        for price in self._asks.islice(stop=n):
            queue = self._asks[price]
            vol = sum(o.remaining for o in queue)
            if vol > 0:
                ask_prices.append(price)
                ask_volumes.append(vol)

        return bid_prices, bid_volumes, ask_prices, ask_volumes

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def best_bid(self) -> Optional[float]:
        if not self._bids:
            return None
        return -self._bids.keys()[0]

    @property
    def best_ask(self) -> Optional[float]:
        if not self._asks:
            return None
        return self._asks.keys()[0]

    @property
    def mid_price(self) -> Optional[float]:
        bb, ba = self.best_bid, self.best_ask
        if bb is not None and ba is not None:
            return round((bb + ba) / 2, 8)
        return bb or ba

    @property
    def spread(self) -> float:
        bb, ba = self.best_bid, self.best_ask
        if bb is not None and ba is not None:
            return round(ba - bb, 8)
        return 0.0

    @property
    def total_bid_volume(self) -> int:
        return sum(
            o.remaining
            for queue in self._bids.values()
            for o in queue
        )

    @property
    def total_ask_volume(self) -> int:
        return sum(
            o.remaining
            for queue in self._asks.values()
            for o in queue
        )

    def get_orders_by_owner(self, owner: str) -> List[Order]:
        """Return all resting orders belonging to *owner*."""
        return [o for o in self._orders.values() if o.owner == owner]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _next_id(self) -> str:
        return f"ORD-{next(self._id_counter):08d}"

    def _round_price(self, price: float) -> float:
        return round(round(price / self.tick_size) * self.tick_size, 8)

    def _insert(self, order: Order) -> None:
        """Insert a (partially filled) order onto the book."""
        if order.side == Side.BUY:
            key = -order.price
            book = self._bids
        else:
            key = order.price
            book = self._asks

        if key not in book:
            book[key] = []
        book[key].append(order)
        self._orders[order.id] = order

    def _match(self, incoming: Order) -> List[Trade]:
        """Match *incoming* against resting orders on the opposite side."""
        trades: List[Trade] = []

        if incoming.side == Side.BUY:
            opposite = self._asks
            # Buy matches against asks at or below the limit price
            matchable = lambda ask_price: ask_price <= incoming.price  # noqa: E731
        else:
            opposite = self._bids
            # Sell matches against bids at or above the limit price
            matchable = lambda neg_bid: (-neg_bid) >= incoming.price  # noqa: E731

        keys_to_remove: List = []

        for key in opposite:
            if incoming.remaining <= 0:
                break
            if not matchable(key):
                break

            queue: List[Order] = opposite[key]
            filled_indices: List[int] = []

            for idx, resting in enumerate(queue):
                if incoming.remaining <= 0:
                    break

                fill_qty = min(incoming.remaining, resting.remaining)
                trade_price = resting.price  # price-time priority: resting order price

                if incoming.side == Side.BUY:
                    buyer_id, seller_id = incoming.id, resting.id
                else:
                    buyer_id, seller_id = resting.id, incoming.id

                trade = Trade(
                    price=trade_price,
                    quantity=fill_qty,
                    buyer_order_id=buyer_id,
                    seller_order_id=seller_id,
                    aggressor_side=incoming.side,
                    timestamp=time.monotonic(),
                )
                trades.append(trade)

                incoming.remaining -= fill_qty
                resting.remaining -= fill_qty

                if resting.remaining == 0:
                    filled_indices.append(idx)
                    self._orders.pop(resting.id, None)

            # Remove fully filled orders from queue (reverse to keep indices valid)
            for idx in reversed(filled_indices):
                queue.pop(idx)

            if not queue:
                keys_to_remove.append(key)

        for key in keys_to_remove:
            del opposite[key]

        return trades

    def clear(self) -> None:
        """Remove all orders from the book."""
        self._bids.clear()
        self._asks.clear()
        self._orders.clear()
        self.last_trades.clear()
