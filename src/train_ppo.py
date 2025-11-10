# src/train_ppo.py
"""
Train PPO agent from scratch on portfolio optimization.

This script implements the complete training loop from Section 4.7:
1. Rollout: Collect data by running policy in environment
2. Compute advantages: Figure out what worked
3. Update: Improve policy and value networks
4. Repeat!
"""

import numpy as np
import pandas as pd
from env import PortfolioEnv
from ppo_agent import PPOAgent


def rollout(env, agent, max_steps=None):
    """
    Run the agent through the environment and collect data.
    
    This is the "ROLLOUT PHASE" from Section 4.3.
    We run the current policy and collect:
    - states
    - actions
    - rewards  
    - next_states
    - log probabilities
    - done flags
    
    Args:
        env: Portfolio environment
        agent: PPO agent
        max_steps: Maximum steps per episode (None = until done)
        
    Returns:
        Dictionary with all collected data
    """
    # Storage for trajectory data
    states = []
    actions = []
    rewards = []
    next_states = []
    log_probs = []
    dones = []
    
    # Reset environment
    state, info = env.reset()
    done = False
    steps = 0
    episode_reward = 0
    
    # Run episode
    while not done:
        # Get action from agent
        action, log_prob = agent.select_action(state, deterministic=False)
        
        # Take step in environment
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Store data
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        next_states.append(next_state if not done else state)  # If done, use dummy state
        log_probs.append(log_prob)
        dones.append(done)
        
        # Update for next iteration
        state = next_state
        episode_reward += reward
        steps += 1
        
        # Check if we hit max steps
        if max_steps is not None and steps >= max_steps:
            break
    
    return {
        "states": states,
        "actions": actions,
        "rewards": rewards,
        "next_states": next_states,
        "log_probs": log_probs,
        "dones": dones,
        "episode_reward": episode_reward,
        "steps": steps,
    }


def train_ppo(
    returns,
    total_episodes=100,
    lookback_window=6,
    transaction_cost=0.001,
    lr=3e-4,
    gamma=0.99,
    clip_range=0.2,
    update_epochs=10,
):
    """
    Train PPO agent on portfolio optimization.
    
    This implements the full training loop from Section 4.7:
    
    For each episode:
        1. ROLLOUT: Run policy, collect data
        2. FREEZE: Save old policy parameters (θ_old = θ)
        3. COMPUTE ADVANTAGES: Figure out what worked
        4. UPDATE: Improve policy and value (with clipping!)
        5. REPEAT
    
    Args:
        returns: DataFrame of monthly returns
        total_episodes: How many episodes to train
        lookback_window: Months of history for state
        transaction_cost: Trading cost penalty
        lr: Learning rate
        gamma: Discount factor
        clip_range: PPO clipping parameter (epsilon)
        update_epochs: How many times to update on each rollout
        
    Returns:
        Trained agent
    """
    
    print("=" * 70)
    print("PPO TRAINING FROM SCRATCH")
    print("=" * 70)
    
    # Step 1: Create environment
    print("\n1. Setting up environment...")
    env = PortfolioEnv(
        returns=returns,
        lookback_window=lookback_window,
        transaction_cost=transaction_cost,
    )
    
    state_dim = env.observation_space.shape[0]  # e.g., 12
    action_dim = env.action_space.shape[0]      # e.g., 6
    
    print(f"   ✓ Environment ready")
    print(f"   - State dimension: {state_dim}")
    print(f"   - Action dimension: {action_dim}")
    print(f"   - Transaction cost: {transaction_cost*100:.2f}%")
    
    # Step 2: Create PPO agent
    print("\n2. Creating PPO agent...")
    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=lr,
        gamma=gamma,
        clip_range=clip_range,
    )
    print(f"   ✓ Agent created with {clip_range*100:.0f}% clipping")
    
    # Step 3: Training loop
    print("\n3. Starting training...")
    print(f"   Total episodes: {total_episodes}")
    print(f"   Update epochs per rollout: {update_epochs}")
    print("")
    
    all_rewards = []
    
    for episode in range(total_episodes):
        
        # --- ROLLOUT PHASE (Section 4.3) ---
        # Run current policy through environment
        # Collect: states, actions, rewards, next_states
        
        trajectory = rollout(env, agent)
        episode_reward = trajectory["episode_reward"]
        all_rewards.append(episode_reward)
        
        # --- UPDATE PHASE (Section 4.5, 4.6) ---
        # Use collected data to improve the agent
        
        # This does:
        # 1. Compute advantages (A = r + γ*V(s_next) - V(s))
        # 2. Update policy with clipped objective
        # 3. Update value network
        metrics = agent.update(
            states=trajectory["states"],
            actions=trajectory["actions"],
            old_log_probs=trajectory["log_probs"],
            rewards=trajectory["rewards"],
            next_states=trajectory["next_states"],
            dones=trajectory["dones"],
            epochs=update_epochs,
        )
        
        # Print progress
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(all_rewards[-10:])
            print(f"Episode {episode + 1}/{total_episodes}")
            print(f"  Avg reward (last 10): {avg_reward:.4f}")
            print(f"  Policy loss: {metrics['policy_loss']:.4f}")
            print(f"  Value loss: {metrics['value_loss']:.4f}")
            print(f"  Entropy: {metrics['entropy']:.4f}")
            print("")
    
    print("=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    print(f"Final average reward (last 10 episodes): {np.mean(all_rewards[-10:]):.4f}")
    
    return agent, all_rewards


def evaluate_agent(agent, env, deterministic=True):
    """
    Evaluate trained agent.
    
    Run the agent through the environment and see how it performs.
    This is like "reviewing game footage" after training.
    
    Args:
        agent: Trained PPO agent
        env: Portfolio environment
        deterministic: Use best action (no exploration)
        
    Returns:
        Dictionary with results
    """
    print("\n" + "=" * 70)
    print("EVALUATING AGENT")
    print("=" * 70)
    
    # Run one episode
    state, info = env.reset()
    done = False
    
    episode_reward = 0
    all_weights = []
    all_returns = []
    steps = 0
    
    while not done:
        # Get action from agent
        action, _ = agent.select_action(state, deterministic=deterministic)
        
        # Take step
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Track results
        episode_reward += reward
        all_weights.append(action)
        all_returns.append(info["portfolio_return"])
        
        state = next_state
        steps += 1
    
    # Compute statistics
    avg_return = np.mean(all_returns)
    total_return = np.sum(all_returns)
    volatility = np.std(all_returns)
    sharpe = avg_return / (volatility + 1e-8) * np.sqrt(12)  # Annualized
    
    print(f"\nResults over {steps} months:")
    print(f"  Total reward: {episode_reward:.4f}")
    print(f"  Average monthly return: {avg_return:.4%}")
    print(f"  Total return: {total_return:.4%}")
    print(f"  Volatility: {volatility:.4%}")
    print(f"  Sharpe ratio: {sharpe:.2f}")
    
    return {
        "episode_reward": episode_reward,
        "avg_return": avg_return,
        "total_return": total_return,
        "volatility": volatility,
        "sharpe": sharpe,
        "weights": np.array(all_weights),
        "returns": np.array(all_returns),
    }


if __name__ == "__main__":
    """
    Main training script.
    """
    
    # Load data
    print("Loading data...")
    returns = pd.read_csv("../data/returns_monthly.csv")
    
    # Clean data
    if "Ticker" in str(returns.iloc[0].values):
        returns = returns.iloc[2:]
    if "Date" in returns.columns:
        returns = returns.set_index("Date")
    else:
        returns = returns.set_index(returns.columns[0])
    returns = returns.apply(pd.to_numeric, errors="coerce").dropna()
    
    print(f"✓ Loaded {returns.shape[0]} months of data")
    print(f"  Assets: {list(returns.columns)}\n")
    
    # Train agent
    agent, reward_history = train_ppo(
        returns=returns,
        total_episodes=100,         # Number of full episodes
        lookback_window=6,           # Use 6 months of history
        transaction_cost=0.001,      # 0.1% transaction cost
        lr=3e-4,                     # Learning rate
        gamma=0.99,                  # Discount factor
        clip_range=0.2,              # PPO clipping (20%)
        update_epochs=10,            # Update 10 times per rollout
    )
    
    # Save agent
    agent.save("ppo_agent.pth")
    
    # Evaluate agent
    env = PortfolioEnv(
        returns=returns,
        lookback_window=6,
        transaction_cost=0.001,
    )
    results = evaluate_agent(agent, env, deterministic=True)
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    print("\nNext steps:")
    print("1. Compare with baselines: python ../run.py")
    print("2. Visualize portfolio weights")
    print("3. Try different hyperparameters")