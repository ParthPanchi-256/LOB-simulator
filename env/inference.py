"""
Inference script demonstrating how to connect to the LOB-Simulator environment
and run a basic market-making agent.

Ensure the server is running locally first:
  python -m env.server.app --port 8000
"""

import sys
sys.path.insert(0, r"c:\Users\parth\Documents\My Projects\MetaHack")

import time
from env.client import LOBEnv
from env.models import LOBAction
from env.server.env_environment import LOBEnvironment

def run_agent():
    print("="*60)
    print("Starting LOB Simulator Agent")
    print("="*60)
    
    # We can use the environment directly for local testing without the server overhead, 
    # but normally you would use the LOBEnv client:
    # with LOBEnv(base_url="http://localhost:8000") as env:
    env = LOBEnvironment()
    obs = env.reset(seed=101)
    
    print(f"Initial Mid-Price: {obs.mid_price:.4f}")
    print(f"Initial Spread:    {obs.spread:.4f}")
    print(f"Initial Cash:      {obs.cash:.2f}")

    total_reward = 0.0

    # Run for 200 steps
    for step in range(200):
        # Extremely basic market making strategy with inventory control
        if obs.inventory > 10:
            # We are too long. aggressively sell to reduce inventory
            action = LOBAction(action_type="market_sell", quantity=5)
        elif obs.inventory < -10:
            # We are too short. aggressively buy
            action = LOBAction(action_type="market_buy", quantity=5)
        else:
            # Quote both sides just outside the spread
            if step % 2 == 0:
                action = LOBAction(
                    action_type="limit_buy",
                    price=obs.mid_price - 0.02, # 2 ticks below mid
                    quantity=3
                )
            else:
                action = LOBAction(
                    action_type="limit_sell",
                    price=obs.mid_price + 0.02, # 2 ticks above mid
                    quantity=3
                )
        
        obs = env.step(action)
        total_reward += (obs.reward or 0)
        
        if step % 10 == 0:
            print(f"[Step {step:03d}] PnL: {obs.realized_pnl:8.2f} | Inv: {obs.inventory:3d} | Cash: {obs.cash:9.2f} | Rwd: {obs.reward:7.4f}")

    print("="*60)
    print(f"Episode Done! Steps run: {obs.step_number}")
    print(f"Final Realized PnL:   {obs.realized_pnl:.2f}")
    print(f"Final Unrealized PnL: {obs.unrealized_pnl:.2f}")
    print(f"Total Cumulative Reward:{total_reward:.4f}")
    print("="*60)

if __name__ == "__main__":
    run_agent()
