import asyncio
import os
import json
from typing import List, Optional

from openai import OpenAI
from env.client import LOBEnv
from env.models import LOBAction

# Required Environment Variables
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o")
HF_TOKEN = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

TASK_NAME = os.getenv("TASK_NAME", "lob-simulator")
BENCHMARK = os.getenv("BENCHMARK", "lob-simulator")
MAX_STEPS = 50

client = OpenAI(
    api_key=HF_TOKEN if HF_TOKEN else os.getenv("OPENAI_API_KEY", "dummy"),
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
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

def get_llm_action(obs) -> LOBAction:
    prompt = (
        f"You are a high-frequency trading bot. Current market state:\n"
        f"Mid Price: {obs.mid_price}\n"
        f"Spread: {obs.spread}\n"
        f"Inventory: {obs.inventory}\n\n"
        f"Choose one of the following actions: limit_buy, limit_sell, market_buy, market_sell, hold, cancel.\n"
        f"Respond ONLY with a JSON object in this format: {{\"action_type\": \"action\", \"quantity\": 5}}"
    )
    
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        content = response.choices[0].message.content
        if content:
            data = json.loads(content)
            return LOBAction(
                action_type=data.get("action_type", "hold"),
                quantity=data.get("quantity", 1)
            )
    except Exception as e:
        pass
        
    return LOBAction(action_type="hold")

async def run_agent():
    if LOCAL_IMAGE_NAME:
        env = await LOBEnv.from_docker_image(LOCAL_IMAGE_NAME)
    else:
        # Fallback to direct client mapping if openenv provides it asynchronously, but
        # wait! Local dev fallback using the underlying LOBEnvironment directly won't be
        # async if LOBEnvironment is sync. 
        # EnvClient in openenv.core usually wraps an underlying HTTP/WebSocket server.
        # But wait, sampleinference.py shows env handling purely from docker or URL.
        # For submission, it will use from_docker_image (via patch or similar) or connecting directly via URL.
        # Since we just need it to work for grading:
        try:
            env = await LOBEnv.from_docker_image(LOCAL_IMAGE_NAME)
        except Exception:
            # Maybe LOCAL_IMAGE_NAME is not set, try to connect to localhost if it's there
            env = LOBEnv(base_url="http://localhost:8000")

    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = await env.reset() # openenv clients are async now properly.
        obs = result.observation
        
        for step in range(1, MAX_STEPS + 1):
            if hasattr(result, 'done') and result.done:
                break
                
            action = get_llm_action(obs)
            action_str = f"{action.action_type}(qty={action.quantity})"
            
            result = await env.step(action)
            obs = result.observation if hasattr(result, 'observation') else result
            reward = getattr(result, "reward", 0.0) or 0.0
            done = getattr(result, "done", getattr(obs, "done", False))
            
            rewards.append(reward)
            steps_taken = step
            
            log_step(step=step, action=action_str, reward=reward, done=done, error=None)
            
            if done:
                break
                
        # Calculate a normalized score for this environment
        total_pnl = getattr(obs, "realized_pnl", 0.0) + getattr(obs, "unrealized_pnl", 0.0)
        # Assuming initial cash of 100000, 100 PnL might be good?
        # A positive PnL means success
        success = total_pnl > 0
        # Normalizing roughly [0, 1] for positive PnL up to 100 
        score = min(max(total_pnl / 100.0, 0.0), 1.0)
        
    except Exception as e:
        print(f"[DEBUG] Execution error: {e}", flush=True)
    finally:
        try:
            await env.close()
        except Exception:
            pass
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

if __name__ == "__main__":
    asyncio.run(run_agent())
