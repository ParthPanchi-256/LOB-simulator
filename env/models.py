# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the High-Frequency Limit Order Book (LOB) Simulator.

The LOB environment simulates a financial exchange micro-structure where
an agent acts as a market maker or algorithmic execution bot.
"""

from typing import List, Optional

from pydantic import Field, field_validator

from openenv.core.env_server.types import Action, Observation


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------

VALID_ACTION_TYPES = frozenset(
    {"limit_buy", "limit_sell", "market_buy", "market_sell", "cancel", "hold"}
)


class LOBAction(Action):
    """Action the agent can take on the exchange.

    Supported action types
    ----------------------
    * ``limit_buy``  / ``limit_sell``  – place a resting limit order at a
      specified *price* for *quantity* shares.
    * ``market_buy`` / ``market_sell`` – immediately execute against the
      opposite side of the book for *quantity* shares.
    * ``cancel`` – cancel the agent's existing limit order identified by
      *order_id*.
    * ``hold`` – do nothing this step.
    """

    action_type: str = Field(
        ...,
        description=(
            "Order type: 'limit_buy', 'limit_sell', 'market_buy', "
            "'market_sell', 'cancel', or 'hold'."
        ),
    )
    price: Optional[float] = Field(
        default=None,
        description="Limit price. Required for limit orders, ignored otherwise.",
    )
    quantity: Optional[int] = Field(
        default=None,
        ge=0,
        description="Number of shares (≥1 for orders, ignored for hold/cancel).",
    )
    order_id: Optional[str] = Field(
        default=None,
        description="ID of the order to cancel (required only for 'cancel').",
    )

    @field_validator("action_type")
    @classmethod
    def _validate_action_type(cls, v: str) -> str:
        v = v.strip().lower()
        if v not in VALID_ACTION_TYPES:
            raise ValueError(
                f"Invalid action_type '{v}'. "
                f"Must be one of {sorted(VALID_ACTION_TYPES)}."
            )
        return v


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------


class LOBObservation(Observation):
    """Observation returned by the LOB environment each step.

    Contains the visible state of the order book, the agent's portfolio,
    market micro-structure metrics, and episode progress.
    """

    # --- Order book snapshot (top-N levels) --------------------------------
    bid_prices: List[float] = Field(
        default_factory=list,
        description="Top-N bid (buy) prices, best first.",
    )
    bid_volumes: List[int] = Field(
        default_factory=list,
        description="Volumes at each bid level.",
    )
    ask_prices: List[float] = Field(
        default_factory=list,
        description="Top-N ask (sell) prices, best first.",
    )
    ask_volumes: List[int] = Field(
        default_factory=list,
        description="Volumes at each ask level.",
    )

    # --- Derived market data -----------------------------------------------
    mid_price: float = Field(default=0.0, description="Current mid-price.")
    spread: float = Field(default=0.0, description="Current bid-ask spread.")
    order_flow_imbalance: float = Field(
        default=0.0,
        description="Rolling order-flow imbalance metric in [-1, 1].",
    )
    vwap: float = Field(
        default=0.0,
        description="Volume-weighted average price of recent trades.",
    )
    volatility: float = Field(
        default=0.0,
        description="Rolling volatility estimate (std of returns).",
    )

    # --- Agent portfolio ---------------------------------------------------
    inventory: int = Field(
        default=0,
        description="Agent's net position (positive = long, negative = short).",
    )
    cash: float = Field(
        default=0.0,
        description="Agent's cash balance.",
    )
    unrealized_pnl: float = Field(
        default=0.0,
        description="Mark-to-market PnL of open position.",
    )
    realized_pnl: float = Field(
        default=0.0,
        description="Cumulative realized PnL from closed trades.",
    )
    active_orders: List[dict] = Field(
        default_factory=list,
        description="Agent's live limit orders [{id, side, price, qty}, ...].",
    )

    # --- Episode progress --------------------------------------------------
    step_number: int = Field(default=0, description="Current step in the episode.")
    total_steps: int = Field(default=1000, description="Total steps in the episode.")

    # --- Recent trades -----------------------------------------------------
    recent_trades: List[dict] = Field(
        default_factory=list,
        description="Trades executed this step [{price, qty, aggressor}, ...].",
    )
