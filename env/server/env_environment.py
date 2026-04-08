# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
High-Frequency Limit Order Book (LOB) Environment.

Simulates the micro-structure of a financial exchange where an agent acts
as a market maker or algorithmic execution bot.  Background traders
generate realistic order flow while the agent must manage PnL and
inventory risk.

Reward:  R_t = ΔPnL_t − λ · |Inventory_t|
"""

from __future__ import annotations

import math
import random
from collections import deque
from typing import Any, Deque, Dict, List, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

from models import LOBAction, LOBObservation

from .background_traders import BackgroundTrader, create_default_traders
from .order_book import OrderBook, Side, Trade


# ---------------------------------------------------------------------------
# Configuration defaults
# ---------------------------------------------------------------------------

_DEFAULT_CONFIG: Dict[str, Any] = {
    "initial_price": 100.0,
    "tick_size": 0.01,
    "book_depth": 10,
    "episode_length": 1000,
    "inventory_penalty_lambda": 0.01,
    "max_inventory": 100,
    "initial_cash": 100_000.0,
    "trader_intensity": 1.0,
    "enable_adversary": True,
    "volatility_window": 50,
    "ofi_window": 20,
    "vwap_window": 50,
}


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class LOBEnvironment(Environment):
    """High-Frequency Limit Order Book Simulator.

    Observation Space
    -----------------
    Top-N levels of the order book (bid/ask prices and volumes), historical
    order-flow imbalance, the agent's current inventory, PnL, and market
    micro-structure metrics.

    Action Space
    ------------
    * ``limit_buy`` / ``limit_sell`` — place a resting limit order.
    * ``market_buy`` / ``market_sell`` — execute immediately.
    * ``cancel`` — cancel an existing agent order.
    * ``hold`` — do nothing.

    Reward
    ------
    ``R_t = ΔPnL_t − λ · |Inventory_t|``

    The Challenge
    -------------
    Learn to execute large block trades while minimising slippage and
    avoiding exploitation by adversarial background traders.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self, **config: Any) -> None:
        cfg = {**_DEFAULT_CONFIG, **config}

        self._initial_price: float = cfg["initial_price"]
        self._tick_size: float = cfg["tick_size"]
        self._book_depth: int = cfg["book_depth"]
        self._episode_length: int = cfg["episode_length"]
        self._lambda: float = cfg["inventory_penalty_lambda"]
        self._max_inventory: int = cfg["max_inventory"]
        self._initial_cash: float = cfg["initial_cash"]
        self._trader_intensity: float = cfg["trader_intensity"]
        self._enable_adversary: bool = cfg["enable_adversary"]
        self._volatility_window: int = cfg["volatility_window"]
        self._ofi_window: int = cfg["ofi_window"]
        self._vwap_window: int = cfg["vwap_window"]

        # Initialised in reset()
        self._book: Optional[OrderBook] = None
        self._traders: List[BackgroundTrader] = []
        self._state = State(episode_id=str(uuid4()), step_count=0)

        # Agent portfolio
        self._cash: float = 0.0
        self._inventory: int = 0
        self._realized_pnl: float = 0.0
        self._avg_entry_price: float = 0.0

        # History tracking
        self._price_history: Deque[float] = deque(maxlen=200)
        self._trade_history: Deque[Trade] = deque(maxlen=500)
        self._ofi_history: Deque[float] = deque(maxlen=self._ofi_window)

        # Agent order tracking
        self._agent_order_ids: set[str] = set()

        # Previous PnL for reward diff
        self._prev_total_pnl: float = 0.0

    # ------------------------------------------------------------------
    # Environment interface
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> LOBObservation:
        """Reset the environment and seed the order book with initial liquidity."""
        if seed is not None:
            random.seed(seed)

        self._state = State(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
        )

        # Fresh order book
        self._book = OrderBook(tick_size=self._tick_size)

        # Background traders
        self._traders = create_default_traders(
            intensity=self._trader_intensity,
            enable_adversary=self._enable_adversary,
        )

        # Reset agent portfolio
        self._cash = self._initial_cash
        self._inventory = 0
        self._realized_pnl = 0.0
        self._avg_entry_price = 0.0
        self._prev_total_pnl = 0.0
        self._agent_order_ids.clear()

        # Reset history
        self._price_history.clear()
        self._trade_history.clear()
        self._ofi_history.clear()

        # Seed the book with initial liquidity
        self._seed_initial_book()

        mid = self._book.mid_price or self._initial_price
        self._price_history.append(mid)

        return self._build_observation(trades_this_step=[], reward=0.0, done=False)

    def step(self, action: LOBAction, **kwargs: Any) -> LOBObservation:  # type: ignore[override]
        """Execute one step of the simulation."""
        assert self._book is not None, "Must call reset() before step()."

        self._state.step_count += 1
        all_trades: List[Trade] = []

        # 1. Execute agent action
        agent_trades = self._execute_agent_action(action)
        all_trades.extend(agent_trades)

        # 2. Tick background traders
        mid = self._book.mid_price or self._initial_price
        agent_orders = self._book.get_orders_by_owner("agent")

        bg_trades: List[Trade] = []
        for trader in self._traders:
            t_trades = trader.act(
                book=self._book,
                mid_price=mid,
                step=self._state.step_count,
                price_history=list(self._price_history),
                agent_inventory=self._inventory,
                agent_orders=agent_orders,
            )
            bg_trades.extend(t_trades)

        # 2b. Process any background fills on agent's resting orders
        self._process_background_fills(bg_trades)
        all_trades.extend(bg_trades)

        # 3. Record trades and update history
        self._trade_history.extend(all_trades)
        mid_after = self._book.mid_price or mid
        self._price_history.append(mid_after)

        # 4. Update order-flow imbalance
        self._update_ofi(all_trades)

        # 5. Compute reward
        total_pnl = self._realized_pnl + self._unrealized_pnl(mid_after)
        delta_pnl = total_pnl - self._prev_total_pnl
        inventory_penalty = self._lambda * abs(self._inventory)
        reward = delta_pnl - inventory_penalty
        self._prev_total_pnl = total_pnl

        # 6. Check termination
        done = (
            self._state.step_count >= self._episode_length
            or abs(self._inventory) > self._max_inventory
        )

        return self._build_observation(
            trades_this_step=all_trades, reward=reward, done=done
        )

    @property
    def state(self) -> State:
        return self._state

    # ------------------------------------------------------------------
    # Agent action execution
    # ------------------------------------------------------------------

    def _execute_agent_action(self, action: LOBAction) -> List[Trade]:
        """Translate agent action into order book operations."""
        assert self._book is not None
        trades: List[Trade] = []

        act = action.action_type

        if act == "hold":
            return trades

        if act == "cancel":
            if action.order_id:
                self._book.cancel_order(action.order_id)
                self._agent_order_ids.discard(action.order_id)
            return trades

        qty = max(1, action.quantity or 1)
        agent_side: str | None = None

        if act == "limit_buy":
            price = action.price or (self._book.best_bid or self._initial_price)
            order_id, new_trades = self._book.add_limit_order(
                side=Side.BUY, price=price, quantity=qty, owner="agent"
            )
            self._agent_order_ids.add(order_id)
            trades.extend(new_trades)
            agent_side = "buy"

        elif act == "limit_sell":
            price = action.price or (self._book.best_ask or self._initial_price)
            order_id, new_trades = self._book.add_limit_order(
                side=Side.SELL, price=price, quantity=qty, owner="agent"
            )
            self._agent_order_ids.add(order_id)
            trades.extend(new_trades)
            agent_side = "sell"

        elif act == "market_buy":
            new_trades = self._book.add_market_order(
                side=Side.BUY, quantity=qty, owner="agent"
            )
            trades.extend(new_trades)
            agent_side = "buy"

        elif act == "market_sell":
            new_trades = self._book.add_market_order(
                side=Side.SELL, quantity=qty, owner="agent"
            )
            trades.extend(new_trades)
            agent_side = "sell"

        # Update agent portfolio — we know which side the agent is on
        if agent_side and trades:
            for trade in trades:
                self._update_position(trade.price, trade.quantity, side=agent_side)

        return trades

    def _process_background_fills(self, bg_trades: List[Trade]) -> None:
        """Check if background traders filled any of the agent's resting orders."""
        for trade in bg_trades:
            if trade.buyer_order_id in self._agent_order_ids:
                self._update_position(trade.price, trade.quantity, side="buy")
                # If fully filled, remove from tracking set
                order = self._book._orders.get(trade.buyer_order_id)
                if order is None or order.remaining == 0:
                    self._agent_order_ids.discard(trade.buyer_order_id)
            elif trade.seller_order_id in self._agent_order_ids:
                self._update_position(trade.price, trade.quantity, side="sell")
                order = self._book._orders.get(trade.seller_order_id)
                if order is None or order.remaining == 0:
                    self._agent_order_ids.discard(trade.seller_order_id)

    def _update_position(self, price: float, qty: int, side: str) -> None:
        """FIFO-style position update with average cost tracking."""
        if side == "buy":
            cost = price * qty
            self._cash -= cost
            if self._inventory >= 0:
                # Adding to long position
                total_cost = self._avg_entry_price * self._inventory + cost
                self._inventory += qty
                self._avg_entry_price = (
                    total_cost / self._inventory if self._inventory != 0 else price
                )
            else:
                # Covering short position
                cover_qty = min(qty, abs(self._inventory))
                pnl = (self._avg_entry_price - price) * cover_qty
                self._realized_pnl += pnl
                self._inventory += qty
                if self._inventory > 0:
                    self._avg_entry_price = price
                elif self._inventory == 0:
                    self._avg_entry_price = 0.0
        else:  # sell
            proceeds = price * qty
            self._cash += proceeds
            if self._inventory <= 0:
                # Adding to short position
                total_cost = abs(self._avg_entry_price * self._inventory) + proceeds
                self._inventory -= qty
                self._avg_entry_price = (
                    total_cost / abs(self._inventory)
                    if self._inventory != 0
                    else price
                )
            else:
                # Closing long position
                close_qty = min(qty, self._inventory)
                pnl = (price - self._avg_entry_price) * close_qty
                self._realized_pnl += pnl
                self._inventory -= qty
                if self._inventory < 0:
                    self._avg_entry_price = price
                elif self._inventory == 0:
                    self._avg_entry_price = 0.0

    # ------------------------------------------------------------------
    # Observation builder
    # ------------------------------------------------------------------

    def _build_observation(
        self,
        trades_this_step: List[Trade],
        reward: float,
        done: bool,
    ) -> LOBObservation:
        assert self._book is not None

        bid_p, bid_v, ask_p, ask_v = self._book.get_top_n(self._book_depth)
        mid = self._book.mid_price or self._initial_price
        spread = self._book.spread

        # Active agent orders
        agent_orders = self._book.get_orders_by_owner("agent")
        active_orders = [
            {
                "id": o.id,
                "side": o.side.value,
                "price": o.price,
                "qty": o.remaining,
            }
            for o in agent_orders
        ]

        # Recent trades for observation
        recent = [
            {
                "price": t.price,
                "qty": t.quantity,
                "aggressor": t.aggressor_side.value,
            }
            for t in trades_this_step[-20:]  # last 20
        ]

        return LOBObservation(
            # Book snapshot
            bid_prices=bid_p,
            bid_volumes=bid_v,
            ask_prices=ask_p,
            ask_volumes=ask_v,
            mid_price=mid,
            spread=spread,
            # Micro-structure
            order_flow_imbalance=self._current_ofi(),
            vwap=self._compute_vwap(),
            volatility=self._compute_volatility(),
            # Portfolio
            inventory=self._inventory,
            cash=self._cash,
            unrealized_pnl=self._unrealized_pnl(mid),
            realized_pnl=self._realized_pnl,
            active_orders=active_orders,
            # Episode
            step_number=self._state.step_count,
            total_steps=self._episode_length,
            recent_trades=recent,
            # Base observation fields
            done=done,
            reward=reward,
            metadata={
                "episode_id": self._state.episode_id,
                "mid_price": mid,
                "spread": spread,
                "total_pnl": self._realized_pnl + self._unrealized_pnl(mid),
            },
        )

    # ------------------------------------------------------------------
    # Market micro-structure helpers
    # ------------------------------------------------------------------

    def _unrealized_pnl(self, current_mid: float) -> float:
        if self._inventory == 0:
            return 0.0
        if self._inventory > 0:
            return (current_mid - self._avg_entry_price) * self._inventory
        else:
            return (self._avg_entry_price - current_mid) * abs(self._inventory)

    def _update_ofi(self, trades: List[Trade]) -> None:
        """Compute order-flow imbalance for this step."""
        buy_vol = sum(t.quantity for t in trades if t.aggressor_side == Side.BUY)
        sell_vol = sum(t.quantity for t in trades if t.aggressor_side == Side.SELL)
        total = buy_vol + sell_vol
        ofi = (buy_vol - sell_vol) / total if total > 0 else 0.0
        self._ofi_history.append(ofi)

    def _current_ofi(self) -> float:
        if not self._ofi_history:
            return 0.0
        return sum(self._ofi_history) / len(self._ofi_history)

    def _compute_vwap(self) -> float:
        """Volume-weighted average price over recent trades."""
        recent = list(self._trade_history)[-self._vwap_window :]
        if not recent:
            return self._initial_price
        total_value = sum(t.price * t.quantity for t in recent)
        total_vol = sum(t.quantity for t in recent)
        return total_value / total_vol if total_vol > 0 else self._initial_price

    def _compute_volatility(self) -> float:
        """Rolling standard deviation of log returns."""
        prices = list(self._price_history)
        n = min(len(prices), self._volatility_window)
        if n < 2:
            return 0.0
        window = prices[-n:]
        returns = [
            math.log(window[i] / window[i - 1])
            for i in range(1, len(window))
            if window[i - 1] > 0 and window[i] > 0
        ]
        if len(returns) < 2:
            return 0.0
        mean_r = sum(returns) / len(returns)
        var = sum((r - mean_r) ** 2 for r in returns) / (len(returns) - 1)
        return math.sqrt(var)

    # ------------------------------------------------------------------
    # Book seeding
    # ------------------------------------------------------------------

    def _seed_initial_book(self) -> None:
        """Populate the book with initial resting liquidity."""
        assert self._book is not None
        mid = self._initial_price

        for i in range(1, 20):
            offset = i * self._tick_size * random.uniform(0.8, 1.2)
            qty_bid = max(1, int(random.gauss(20, 8)))
            qty_ask = max(1, int(random.gauss(20, 8)))

            self._book.add_limit_order(
                side=Side.BUY,
                price=mid - offset,
                quantity=qty_bid,
                owner="init_seed",
            )
            self._book.add_limit_order(
                side=Side.SELL,
                price=mid + offset,
                quantity=qty_ask,
                owner="init_seed",
            )
