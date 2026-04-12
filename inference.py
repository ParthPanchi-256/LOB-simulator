import asyncio
import os
import json
import textwrap
from typing import List, Optional, Tuple

from openai import OpenAI
from client import LOBEnv
from models import LOBAction

# Required Environment Variables
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o")
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
API_GATEWAY = os.getenv("API_GATEWAY", "https://parth256-lobsimulator.hf.space")
BENCHMARK = os.getenv("BENCHMARK", "lob-simulator")
MAX_STEPS = 50

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

client = OpenAI(
    api_key=HF_TOKEN,
    base_url=API_BASE_URL
)

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}", flush=True)

def build_system_prompt() -> str:
    return textwrap.dedent(
        """
        You are a high-frequency market-making bot on a Limit Order Book exchange.

        CORE PRINCIPLES:
        - Default: ACT (limit_buy or limit_sell). hold is rare.
        - Positive OFI (>0.2) = buy pressure -> lean limit_sell to capture spread.
        - Negative OFI (<-0.2) = sell pressure -> lean limit_buy to capture spread.
        - High inventory (>5) = risk -> reduce with market_sell.
        - Low inventory (<-5) = risk -> reduce with market_buy.
        - Wide spread = more profit opportunity; narrow spread = be selective.

        DECISION FLOW:
        1. If |inventory| > 8: use market_buy or market_sell to flatten.
        2. If OFI > 0.2: place limit_sell just above mid_price.
        3. If OFI < -0.2: place limit_buy just below mid_price.
        4. Otherwise: place limit orders symmetrically around mid_price.

        OUTPUT FORMAT (JSON only):
        {"action_type": "<action>", "price": <float_or_null>, "quantity": <int>}

        Valid action_type values: limit_buy, limit_sell, market_buy, market_sell, hold, cancel
        """
    ).strip()

def get_llm_action(obs, last_action: str = "hold") -> Tuple[LOBAction, str]:
    user_prompt = (
        f"Mid Price: {obs.mid_price:.4f} | Spread: {obs.spread:.4f} | "
        f"OFI: {obs.order_flow_imbalance:.4f} | Volatility: {obs.volatility:.6f}\n"
        f"Inventory: {obs.inventory} | Cash: {obs.cash:.2f} | "
        f"Realized PnL: {obs.realized_pnl:.4f} | Unrealized PnL: {obs.unrealized_pnl:.4f}\n"
        f"Step: {obs.step_number}/{obs.total_steps} | Last Action: {last_action}\n"
        f"Active Orders: {len(obs.active_orders)} | Recent Trades: {len(obs.recent_trades)}\n\n"
        f"Respond with JSON only: {{\"action_type\": \"...\", \"price\": null, \"quantity\": 5}}"
    )

    for attempt in range(2):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": build_system_prompt()},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
                max_tokens=100,
            )
            content = response.choices[0].message.content
            if content:
                data = json.loads(content)
                action_type = data.get("action_type", "hold")
                price = data.get("price")
                quantity = int(data.get("quantity", 5))
                action = LOBAction(action_type=action_type, price=price, quantity=quantity)
                action_str = f"{action_type}(qty={quantity})" if price is None else f"{action_type}(p={price},qty={quantity})"
                return action, action_str
        except Exception as e:
            if attempt == 0:
                continue
            print(f"[DEBUG] LLM call failed: {e}", flush=True)

    return LOBAction(action_type="hold"), "hold(qty=0)"

async def run_single_task(env: LOBEnv, task_name: str):
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    last_action = "hold"

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = await env.reset(task_name=task_name)
        obs = result.observation

        for step in range(1, MAX_STEPS + 1):
            if hasattr(result, "done") and result.done:
                break

            action, action_str = get_llm_action(obs, last_action)
            last_action = action.action_type

            result = await env.step(action)
            obs = result.observation if hasattr(result, "observation") else result
            reward = getattr(result, "reward", 0.0) or 0.0
            done = getattr(result, "done", getattr(obs, "done", False))

            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=action_str, reward=reward, done=done, error=None)

            if done:
                break

        total_pnl = getattr(obs, "realized_pnl", 0.0) + getattr(obs, "unrealized_pnl", 0.0)
        success = total_pnl > 0
        score = min(max(total_pnl / 100.0, 0.0), 1.0)

    except Exception as e:
        print(f"[DEBUG] Execution error: {e}", flush=True)
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

async def run_agent():
    try:
        if LOCAL_IMAGE_NAME:
            env = await LOBEnv.from_docker_image(LOCAL_IMAGE_NAME)
        else:
            env = LOBEnv(base_url=API_GATEWAY)
    except Exception as e:
        print(f"[DEBUG] Setup error: {e}", flush=True)
        return

    tasks = ["noise-survival", "momentum-capture", "adversarial-robustness"]

    single_task = os.getenv("TASK_NAME")
    if single_task and single_task in tasks:
        tasks = [single_task]

    for t in tasks:
        await run_single_task(env, t)

    try:
        await env.close()
    except Exception:
        pass

if __name__ == "__main__":
    asyncio.run(run_agent())
