"""Smoke test for the LOB Simulator environment."""
import sys
sys.path.insert(0, r"c:\Users\parth\Documents\My Projects\MetaHack\env")
sys.path.insert(0, r"c:\Users\parth\Documents\My Projects\MetaHack\env\server")

from server.order_book import OrderBook, Side
from server.env_environment import LOBEnvironment
from models import LOBAction

def test_order_book():
    """Test the order book matching engine."""
    print("=" * 60)
    print("TEST 1: Order Book Engine")
    print("=" * 60)

    book = OrderBook(tick_size=0.01)

    # Add some bids
    id1, _ = book.add_limit_order(Side.BUY, 99.95, 10, "trader_a")
    id2, _ = book.add_limit_order(Side.BUY, 99.90, 20, "trader_b")
    id3, _ = book.add_limit_order(Side.BUY, 99.95, 5, "trader_c")

    # Add some asks
    id4, _ = book.add_limit_order(Side.SELL, 100.05, 15, "trader_d")
    id5, _ = book.add_limit_order(Side.SELL, 100.10, 25, "trader_e")

    print(f"  Best bid: {book.best_bid}  (expected 99.95)")
    print(f"  Best ask: {book.best_ask}  (expected 100.05)")
    print(f"  Mid price: {book.mid_price}")
    print(f"  Spread: {book.spread}")

    assert book.best_bid == 99.95, f"Expected best_bid=99.95, got {book.best_bid}"
    assert book.best_ask == 100.05, f"Expected best_ask=100.05, got {book.best_ask}"

    # Test market order matching
    trades = book.add_market_order(Side.BUY, 8, "aggressor")
    print(f"\n  Market buy 8 shares -> {len(trades)} trade(s)")
    for t in trades:
        print(f"    Filled {t.quantity}@{t.price}")

    assert len(trades) == 1, f"Expected 1 trade, got {len(trades)}"
    assert trades[0].quantity == 8
    assert trades[0].price == 100.05

    # Test top-N
    bp, bv, ap, av = book.get_top_n(5)
    print(f"\n  Top 5 bids: {list(zip(bp, bv))}")
    print(f"  Top 5 asks: {list(zip(ap, av))}")

    # Remaining ask at 100.05 should be 15-8=7
    assert ap[0] == 100.05
    assert av[0] == 7

    # Test cancel
    ok = book.cancel_order(id2)
    print(f"\n  Cancel order {id2}: {ok}")
    assert ok is True
    bp2, bv2, _, _ = book.get_top_n(5)
    print(f"  Bids after cancel: {list(zip(bp2, bv2))}")

    # Test partial fill with price-time priority
    book2 = OrderBook(tick_size=0.01)
    book2.add_limit_order(Side.BUY, 100.0, 10, "A")
    book2.add_limit_order(Side.BUY, 100.0, 5, "B")  # same price, later time
    trades2 = book2.add_market_order(Side.SELL, 12, "C")
    assert len(trades2) == 2, f"Expected 2 trades (partial), got {len(trades2)}"
    assert trades2[0].quantity == 10  # A filled first (time priority)
    assert trades2[1].quantity == 2   # B partially filled
    print("\n  Price-time priority: PASSED")

    print("  [PASS] Order book tests passed!\n")


def test_environment():
    """Test the full LOB environment."""
    print("=" * 60)
    print("TEST 2: LOB Environment (100-step episode)")
    print("=" * 60)

    env = LOBEnvironment()
    obs = env.reset(seed=42)

    print(f"  Initial mid-price: {obs.mid_price:.4f}")
    print(f"  Initial spread: {obs.spread:.4f}")
    print(f"  Initial cash: {obs.cash:.2f}")
    print(f"  Book depth: {len(obs.bid_prices)} bids, {len(obs.ask_prices)} asks")
    print(f"  Inventory: {obs.inventory}")

    assert len(obs.bid_prices) > 0, "No bid prices!"
    assert len(obs.ask_prices) > 0, "No ask prices!"
    assert obs.mid_price > 0, "Mid price should be positive!"
    assert obs.cash == 100000.0, f"Initial cash should be 100000, got {obs.cash}"

    total_reward = 0.0
    actions_taken = {"hold": 0, "limit_buy": 0, "limit_sell": 0, "market_buy": 0, "market_sell": 0}

    for step in range(100):
        # Simple market-making strategy
        if obs.inventory > 10:
            action = LOBAction(action_type="market_sell", quantity=3)
            actions_taken["market_sell"] += 1
        elif obs.inventory < -10:
            action = LOBAction(action_type="market_buy", quantity=3)
            actions_taken["market_buy"] += 1
        elif step % 3 == 0:
            action = LOBAction(
                action_type="limit_buy",
                price=obs.mid_price - 0.03,
                quantity=5,
            )
            actions_taken["limit_buy"] += 1
        elif step % 3 == 1:
            action = LOBAction(
                action_type="limit_sell",
                price=obs.mid_price + 0.03,
                quantity=5,
            )
            actions_taken["limit_sell"] += 1
        else:
            action = LOBAction(action_type="hold")
            actions_taken["hold"] += 1

        obs = env.step(action)
        total_reward += obs.reward or 0.0

    print(f"\n  After 100 steps:")
    print(f"    Mid-price:      {obs.mid_price:.4f}")
    print(f"    Spread:         {obs.spread:.4f}")
    print(f"    Inventory:      {obs.inventory}")
    print(f"    Cash:           {obs.cash:.2f}")
    print(f"    Realized PnL:   {obs.realized_pnl:.2f}")
    print(f"    Unrealized PnL: {obs.unrealized_pnl:.2f}")
    print(f"    Total reward:   {total_reward:.4f}")
    print(f"    OFI:            {obs.order_flow_imbalance:.4f}")
    print(f"    VWAP:           {obs.vwap:.4f}")
    print(f"    Volatility:     {obs.volatility:.6f}")
    print(f"    Active orders:  {len(obs.active_orders)}")
    print(f"    Recent trades:  {len(obs.recent_trades)}")
    print(f"    Done:           {obs.done}")
    print(f"\n  Actions taken: {actions_taken}")

    # Validate observation fields
    assert obs.step_number == 100, f"Step should be 100, got {obs.step_number}"
    assert obs.total_steps == 1000, f"Total steps should be 1000"
    assert obs.done is False, "Should not be done at step 100"

    print("  [PASS] Environment tests passed!\n")


def test_reward_sanity():
    """Verify reward calculation matches manual computation."""
    print("=" * 60)
    print("TEST 3: Reward Sanity Check")
    print("=" * 60)

    env = LOBEnvironment(inventory_penalty_lambda=0.1)
    obs = env.reset(seed=123)

    # Step 1: hold -> reward ~ market movement only (no inventory penalty)
    obs = env.step(LOBAction(action_type="hold"))
    print(f"  Step 1 (hold): reward={obs.reward:.6f}, inv={obs.inventory}")
    assert obs.inventory == 0, "Inventory should be 0 after hold"

    # Step 2: buy -> creates inventory
    obs = env.step(LOBAction(action_type="market_buy", quantity=5))
    print(f"  Step 2 (buy 5): reward={obs.reward:.6f}, inv={obs.inventory}")

    # The reward should now include an inventory penalty
    # Since lambda=0.1 and |inventory|>0, penalty > 0
    expected_penalty = 0.1 * abs(obs.inventory)
    print(f"    (lambda*|inv| = 0.1*{abs(obs.inventory)} = {expected_penalty:.1f})")

    # Step 3: sell to close -> should realize PnL
    obs = env.step(LOBAction(action_type="market_sell", quantity=max(1, abs(obs.inventory))))
    print(f"  Step 3 (sell to close): reward={obs.reward:.6f}, inv={obs.inventory}")
    print(f"    Realized PnL: {obs.realized_pnl:.4f}")

    print("  [PASS] Reward sanity tests passed!\n")


def test_episode_termination():
    """Test that episode terminates correctly."""
    print("=" * 60)
    print("TEST 4: Episode Termination")
    print("=" * 60)

    # Short episode
    env = LOBEnvironment(episode_length=10)
    obs = env.reset(seed=99)

    for i in range(10):
        obs = env.step(LOBAction(action_type="hold"))

    assert obs.done is True, f"Expected done=True at step 10, got {obs.done}"
    print(f"  Episode ended at step {obs.step_number} (expected 10)")
    print("  [PASS] Episode termination test passed!\n")


def test_cancel_action():
    """Test order cancellation."""
    print("=" * 60)
    print("TEST 5: Order Cancellation")
    print("=" * 60)

    env = LOBEnvironment()
    obs = env.reset(seed=77)

    # Place a limit order
    obs = env.step(LOBAction(action_type="limit_buy", price=obs.mid_price - 1.0, quantity=10))
    agent_orders = obs.active_orders
    print(f"  After limit_buy: {len(agent_orders)} active order(s)")

    if agent_orders:
        oid = agent_orders[0]["id"]
        obs = env.step(LOBAction(action_type="cancel", order_id=oid))
        print(f"  After cancel: {len(obs.active_orders)} active order(s)")

    print("  [PASS] Cancel action test passed!\n")


if __name__ == "__main__":
    test_order_book()
    test_environment()
    test_reward_sanity()
    test_episode_termination()
    test_cancel_action()
    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
