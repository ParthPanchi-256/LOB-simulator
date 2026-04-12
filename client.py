# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""LOB Simulator Client."""

import asyncio
import subprocess
from dataclasses import dataclass
from typing import Any, Dict, Optional

import httpx

from models import LOBAction, LOBObservation


@dataclass
class StepResult:
    observation: LOBObservation
    reward: float
    done: bool


class LOBEnv:
    """
    HTTP client for the High-Frequency Limit Order Book Simulator.

    Communicates with the LOB server via plain HTTP POST requests to
    /reset and /step endpoints, matching the OpenEnv HTTP server spec.
    """

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")
        self._http = httpx.AsyncClient(timeout=30.0)
        self._container_id: Optional[str] = None

    @classmethod
    async def from_docker_image(cls, image_name: str, port: int = 8000) -> "LOBEnv":
        container_id = subprocess.check_output(
            ["docker", "run", "-d", "-p", f"{port}:8000", image_name]
        ).decode().strip()
        env = cls(base_url=f"http://localhost:{port}")
        env._container_id = container_id
        for _ in range(30):
            try:
                await env._http.get(f"{env.base_url}/health")
                break
            except Exception:
                await asyncio.sleep(1)
        return env

    def _parse_observation(self, payload: Dict[str, Any]) -> LOBObservation:
        obs_data = payload.get("observation", {})
        return LOBObservation(
            bid_prices=obs_data.get("bid_prices", []),
            bid_volumes=obs_data.get("bid_volumes", []),
            ask_prices=obs_data.get("ask_prices", []),
            ask_volumes=obs_data.get("ask_volumes", []),
            mid_price=obs_data.get("mid_price", 0.0),
            spread=obs_data.get("spread", 0.0),
            order_flow_imbalance=obs_data.get("order_flow_imbalance", 0.0),
            vwap=obs_data.get("vwap", 0.0),
            volatility=obs_data.get("volatility", 0.0),
            inventory=obs_data.get("inventory", 0),
            cash=obs_data.get("cash", 0.0),
            unrealized_pnl=obs_data.get("unrealized_pnl", 0.0),
            realized_pnl=obs_data.get("realized_pnl", 0.0),
            active_orders=obs_data.get("active_orders", []),
            step_number=obs_data.get("step_number", 0),
            total_steps=obs_data.get("total_steps", 1000),
            recent_trades=obs_data.get("recent_trades", []),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )

    async def reset(self, task_name: str = "") -> StepResult:
        body: Dict[str, Any] = {}
        if task_name:
            body["task_name"] = task_name
        resp = await self._http.post(f"{self.base_url}/reset", json=body)
        resp.raise_for_status()
        data = resp.json()
        obs = self._parse_observation(data)
        return StepResult(observation=obs, reward=0.0, done=data.get("done", False))

    async def step(self, action: LOBAction) -> StepResult:
        action_payload: Dict[str, Any] = {"action_type": action.action_type}
        if action.price is not None:
            action_payload["price"] = action.price
        if action.quantity is not None:
            action_payload["quantity"] = action.quantity
        if action.order_id is not None:
            action_payload["order_id"] = action.order_id
        resp = await self._http.post(
            f"{self.base_url}/step", json={"action": action_payload}
        )
        resp.raise_for_status()
        data = resp.json()
        obs = self._parse_observation(data)
        reward = data.get("reward") or 0.0
        done = data.get("done", False)
        return StepResult(observation=obs, reward=reward, done=done)

    async def close(self) -> None:
        await self._http.aclose()
        if self._container_id:
            subprocess.run(["docker", "stop", self._container_id], check=False)
            subprocess.run(["docker", "rm", self._container_id], check=False)
