# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Background traders that generate realistic order flow in the LOB simulator.

Trader archetypes
-----------------
* **NoiseTrader** — random limit/market orders around mid-price; provides
  base liquidity and ensures the book is never empty.
* **MomentumTrader** — detects short-term trends and submits directional
  market orders.
* **MeanReversionTrader** — fades large price moves with contrarian limit
  orders.
* **AdversarialTrader** — adapts to the agent's inventory and open orders;
  front-runs, penny-jumps, and spoofs to exploit predictable behaviour.
"""

from __future__ import annotations

import math
import random
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List, Optional

from .order_book import Side, Trade

if TYPE_CHECKING:
    from .order_book import OrderBook


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class BackgroundTrader(ABC):
    """Abstract base for all simulated market participants."""

    def __init__(self, trader_id: str, intensity: float = 1.0) -> None:
        self.trader_id = trader_id
        self.intensity = intensity  # scales number/size of orders

    @abstractmethod
    def act(
        self,
        book: "OrderBook",
        mid_price: float,
        step: int,
        price_history: List[float],
        agent_inventory: int = 0,
        agent_orders: Optional[list] = None,
    ) -> List[Trade]:
        """Submit orders to *book* and return any resulting trades."""
        ...


# ---------------------------------------------------------------------------
# Noise Trader
# ---------------------------------------------------------------------------


class NoiseTrader(BackgroundTrader):
    """Random limit and market orders around mid-price.

    Provides base liquidity so the book is never empty.
    Each step, submits 2-6 limit orders on each side and occasionally
    a market order.
    """

    def __init__(
        self,
        trader_id: str = "noise",
        intensity: float = 1.0,
        spread_range: float = 0.5,
    ) -> None:
        super().__init__(trader_id, intensity)
        self.spread_range = spread_range  # max offset from mid in price units

    def act(
        self,
        book: "OrderBook",
        mid_price: float,
        step: int,
        price_history: List[float],
        agent_inventory: int = 0,
        agent_orders: Optional[list] = None,
    ) -> List[Trade]:
        trades: List[Trade] = []
        n_orders = max(2, int(random.gauss(4, 1) * self.intensity))

        for _ in range(n_orders):
            side = random.choice([Side.BUY, Side.SELL])
            offset = random.uniform(0.01, self.spread_range)
            qty = max(1, int(random.gauss(10, 5) * self.intensity))

            if side == Side.BUY:
                price = mid_price - offset
            else:
                price = mid_price + offset

            _, new_trades = book.add_limit_order(
                side=side, price=price, quantity=qty, owner=self.trader_id
            )
            trades.extend(new_trades)

        # Occasional market order (≈20% chance)
        if random.random() < 0.20 * self.intensity:
            side = random.choice([Side.BUY, Side.SELL])
            qty = max(1, int(random.gauss(5, 2)))
            new_trades = book.add_market_order(
                side=side, quantity=qty, owner=self.trader_id
            )
            trades.extend(new_trades)

        # Occasionally cancel own stale orders (≈30% chance)
        if random.random() < 0.30:
            own_orders = book.get_orders_by_owner(self.trader_id)
            n_cancel = min(len(own_orders), random.randint(1, 3))
            for order in random.sample(own_orders, n_cancel) if own_orders else []:
                book.cancel_order(order.id)

        return trades


# ---------------------------------------------------------------------------
# Momentum Trader
# ---------------------------------------------------------------------------


class MomentumTrader(BackgroundTrader):
    """Detects short-term trends and trades in that direction.

    Computes a simple moving-average crossover signal and submits
    market orders when momentum is strong enough.
    """

    def __init__(
        self,
        trader_id: str = "momentum",
        intensity: float = 1.0,
        lookback_short: int = 5,
        lookback_long: int = 20,
        threshold: float = 0.05,
    ) -> None:
        super().__init__(trader_id, intensity)
        self.lookback_short = lookback_short
        self.lookback_long = lookback_long
        self.threshold = threshold

    def act(
        self,
        book: "OrderBook",
        mid_price: float,
        step: int,
        price_history: List[float],
        agent_inventory: int = 0,
        agent_orders: Optional[list] = None,
    ) -> List[Trade]:
        trades: List[Trade] = []

        if len(price_history) < self.lookback_long:
            return trades  # not enough data

        short_ma = sum(price_history[-self.lookback_short :]) / self.lookback_short
        long_ma = sum(price_history[-self.lookback_long :]) / self.lookback_long

        signal = (short_ma - long_ma) / long_ma  # fractional crossover

        if abs(signal) > self.threshold:
            side = Side.BUY if signal > 0 else Side.SELL
            qty = max(1, int(abs(signal) * 50 * self.intensity))
            qty = min(qty, 20)  # cap

            new_trades = book.add_market_order(
                side=side, quantity=qty, owner=self.trader_id
            )
            trades.extend(new_trades)

        return trades


# ---------------------------------------------------------------------------
# Mean-Reversion Trader
# ---------------------------------------------------------------------------


class MeanReversionTrader(BackgroundTrader):
    """Fades large price moves with contrarian limit orders.

    When the price has moved significantly from its rolling mean,
    places limit orders in the opposite direction expecting a revert.
    """

    def __init__(
        self,
        trader_id: str = "meanrev",
        intensity: float = 1.0,
        lookback: int = 50,
        threshold_std: float = 1.5,
    ) -> None:
        super().__init__(trader_id, intensity)
        self.lookback = lookback
        self.threshold_std = threshold_std

    def act(
        self,
        book: "OrderBook",
        mid_price: float,
        step: int,
        price_history: List[float],
        agent_inventory: int = 0,
        agent_orders: Optional[list] = None,
    ) -> List[Trade]:
        trades: List[Trade] = []

        if len(price_history) < self.lookback:
            return trades

        window = price_history[-self.lookback :]
        mean = sum(window) / len(window)
        variance = sum((p - mean) ** 2 for p in window) / len(window)
        std = math.sqrt(variance) if variance > 0 else 0.01

        z_score = (mid_price - mean) / std

        if abs(z_score) > self.threshold_std:
            # Price is extended — fade the move
            if z_score > 0:
                # Price above mean → sell (expect reversion down)
                side = Side.SELL
                price = mid_price - 0.01  # aggressive limit
            else:
                # Price below mean → buy (expect reversion up)
                side = Side.BUY
                price = mid_price + 0.01

            qty = max(1, int(abs(z_score) * 8 * self.intensity))
            qty = min(qty, 30)

            _, new_trades = book.add_limit_order(
                side=side, price=price, quantity=qty, owner=self.trader_id
            )
            trades.extend(new_trades)

        return trades


# ---------------------------------------------------------------------------
# Adversarial Trader
# ---------------------------------------------------------------------------


class AdversarialTrader(BackgroundTrader):
    """Exploits the agent's predictable behaviour.

    Strategies:
    * **Front-running** — when the agent has large directional inventory,
      trades ahead of the expected unwind.
    * **Penny-jumping** — places orders one tick ahead of the agent's
      resting limit orders.
    * **Spoofing** — places and quickly cancels large orders to move
      the mid-price before the agent acts.
    """

    def __init__(
        self,
        trader_id: str = "adversary",
        intensity: float = 1.0,
    ) -> None:
        super().__init__(trader_id, intensity)
        self._spoof_ids: List[str] = []

    def act(
        self,
        book: "OrderBook",
        mid_price: float,
        step: int,
        price_history: List[float],
        agent_inventory: int = 0,
        agent_orders: Optional[list] = None,
    ) -> List[Trade]:
        trades: List[Trade] = []

        # --- Cancel previous spoof orders ---
        for oid in self._spoof_ids:
            book.cancel_order(oid)
        self._spoof_ids.clear()

        # --- Front-running ---
        # If agent is long, they'll likely sell → adversary sells first
        # If agent is short, they'll likely buy → adversary buys first
        if abs(agent_inventory) > 10:
            direction = Side.SELL if agent_inventory > 0 else Side.BUY
            qty = max(1, int(min(abs(agent_inventory) * 0.3, 15) * self.intensity))

            if random.random() < 0.5 * self.intensity:
                new_trades = book.add_market_order(
                    side=direction, quantity=qty, owner=self.trader_id
                )
                trades.extend(new_trades)

        # --- Penny-jumping ---
        if agent_orders:
            for ao in agent_orders[:3]:  # top 3 agent orders
                ao_side = ao.side if hasattr(ao, "side") else Side(ao.get("side", "buy"))
                ao_price = ao.price if hasattr(ao, "price") else ao.get("price", 0)

                if random.random() < 0.4 * self.intensity:
                    if ao_side == Side.BUY:
                        # Jump one tick above agent's bid
                        jump_price = ao_price + book.tick_size
                    else:
                        # Jump one tick below agent's ask
                        jump_price = ao_price - book.tick_size

                    qty = max(1, int(random.gauss(5, 2)))
                    oid, new_trades = book.add_limit_order(
                        side=ao_side,
                        price=jump_price,
                        quantity=qty,
                        owner=self.trader_id,
                    )
                    trades.extend(new_trades)

        # --- Spoofing ---
        if random.random() < 0.25 * self.intensity:
            spoof_side = random.choice([Side.BUY, Side.SELL])
            offset = random.uniform(0.05, 0.15)
            if spoof_side == Side.BUY:
                spoof_price = mid_price - offset
            else:
                spoof_price = mid_price + offset

            spoof_qty = int(random.uniform(50, 150) * self.intensity)
            oid, _ = book.add_limit_order(
                side=spoof_side,
                price=spoof_price,
                quantity=spoof_qty,
                owner=self.trader_id,
            )
            self._spoof_ids.append(oid)  # cancel next step

        return trades


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_default_traders(
    intensity: float = 1.0,
    enable_adversary: bool = True,
) -> List[BackgroundTrader]:
    """Create the default set of background traders."""
    traders: List[BackgroundTrader] = [
        NoiseTrader(trader_id="noise_1", intensity=intensity),
        NoiseTrader(trader_id="noise_2", intensity=intensity * 0.7),
        MomentumTrader(trader_id="momentum_1", intensity=intensity),
        MeanReversionTrader(trader_id="meanrev_1", intensity=intensity),
    ]
    if enable_adversary:
        traders.append(AdversarialTrader(trader_id="adversary_1", intensity=intensity))
    return traders
