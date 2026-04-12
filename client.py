# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""LOB Simulator Client."""

from typing import Any, Dict, List

from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State
from openenv.core import EnvClient

from models import LOBAction, LOBObservation


class LOBEnv(
    EnvClient[LOBAction, LOBObservation]
):
    """
    Client for the High-Frequency Limit Order Book Simulator.

    This client maintains a persistent WebSocket connection to the
    environment server, enabling efficient multi-step interactions
    with lower latency.  Each client instance has its own dedicated
    environment session on the server.

    Example:
        >>> # Connect to a running server
        >>> with LOBEnv(base_url="http://localhost:8000") as env:
        ...     result = env.reset()
        ...     print(f"Mid price: {result.observation.mid_price}")
        ...
        ...     # Place a limit buy
        ...     action = LOBAction(
        ...         action_type="limit_buy",
        ...         price=99.95,
        ...         quantity=10
        ...     )
        ...     result = env.step(action)
        ...     print(f"Inventory: {result.observation.inventory}")
        ...     print(f"PnL: {result.observation.realized_pnl}")

    Example with Docker:
        >>> client = LOBEnv.from_docker_image("lob-simulator:latest")
        >>> try:
        ...     result = client.reset()
        ...     # Run a market-making episode
        ...     for _ in range(100):
        ...         obs = result.observation
        ...         if obs.inventory > 5:
        ...             action = LOBAction(action_type="market_sell", quantity=3)
        ...         elif obs.inventory < -5:
        ...             action = LOBAction(action_type="market_buy", quantity=3)
        ...         else:
        ...             action = LOBAction(action_type="hold")
        ...         result = client.step(action)
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: LOBAction) -> Dict[str, Any]:
        """
        Convert LOBAction to JSON payload for step message.

        Args:
            action: LOBAction instance

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        payload: Dict[str, Any] = {
            "action_type": action.action_type,
        }
        if action.price is not None:
            payload["price"] = action.price
        if action.quantity is not None:
            payload["quantity"] = action.quantity
        if action.order_id is not None:
            payload["order_id"] = action.order_id
        if action.metadata:
            payload["metadata"] = action.metadata
        return payload

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[LOBObservation]:
        """
        Parse server response into StepResult[LOBObservation].

        Args:
            payload: JSON response data from server

        Returns:
            StepResult with LOBObservation
        """
        obs_data = payload.get("observation", {})

        observation = LOBObservation(
            # Book snapshot
            bid_prices=obs_data.get("bid_prices", []),
            bid_volumes=obs_data.get("bid_volumes", []),
            ask_prices=obs_data.get("ask_prices", []),
            ask_volumes=obs_data.get("ask_volumes", []),
            # Market data
            mid_price=obs_data.get("mid_price", 0.0),
            spread=obs_data.get("spread", 0.0),
            order_flow_imbalance=obs_data.get("order_flow_imbalance", 0.0),
            vwap=obs_data.get("vwap", 0.0),
            volatility=obs_data.get("volatility", 0.0),
            # Portfolio
            inventory=obs_data.get("inventory", 0),
            cash=obs_data.get("cash", 0.0),
            unrealized_pnl=obs_data.get("unrealized_pnl", 0.0),
            realized_pnl=obs_data.get("realized_pnl", 0.0),
            active_orders=obs_data.get("active_orders", []),
            # Episode
            step_number=obs_data.get("step_number", 0),
            total_steps=obs_data.get("total_steps", 1000),
            recent_trades=obs_data.get("recent_trades", []),
            # Base fields
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> State:
        """
        Parse server response into State object.

        Args:
            payload: JSON response from state request

        Returns:
            State object with episode_id and step_count
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
