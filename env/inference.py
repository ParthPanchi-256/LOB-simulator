import os
import json
from openai import OpenAI
from client import LOBEnv
from models import LOBAction

# Required Environment Variables
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o")
HF_TOKEN = os.getenv("HF_TOKEN")

# Optional - if you use from_docker_image()
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

# All LLM calls use the OpenAI client configured via these variables
client = OpenAI(
    api_key=HF_TOKEN if HF_TOKEN else os.getenv("OPENAI_API_KEY", "dummy"),
    base_url=API_BASE_URL
)

def get_llm_action(obs) -> LOBAction:
    """Use the OpenAI client to decide the next action based on the observation."""
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
        # Fallback on error
        pass
        
    return LOBAction(action_type="hold")

def run_agent():
    # Stdout logs follow the required structured format (START/STEP/END) exactly
    print("START")
    
    if LOCAL_IMAGE_NAME:
        env = LOBEnv.from_docker_image(LOCAL_IMAGE_NAME)
    else:
        # Local development fallback
        from server.env_environment import LOBEnvironment
        env = LOBEnvironment()

    try:
        result = env.reset(seed=42)
        
        # We assume the environment gives us observation directly or wrapped in a result
        # For our local LOBEnvironment, reset returns LOBObservation
        # For EnvClient, it returns StepResult
        obs = result.observation if hasattr(result, 'observation') else result
        
        for step_idx in range(50):
            print("STEP")
            
            action = get_llm_action(obs)
            result = env.step(action)
            obs = result.observation if hasattr(result, 'observation') else result
            done = result.done if hasattr(result, 'done') else obs.done
            
            if done:
                break
    finally:
        if hasattr(env, 'close'):
            env.close()

    print("END")

if __name__ == "__main__":
    run_agent()
