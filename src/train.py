# src/train.py
"""
Proper training pipeline with train/val/test splits.
Trains PPO on training set, evaluates on validation, and tests on held-out test set.
"""

import numpy as np
import pandas as pd
from env import PortfolioEnv
from ppo_agent import PPOAgent
from evaluate import split_data, compare_all, print_comparison


def train_ppo(
    train_returns,
    val_returns=None,
    total_episodes=100,
    lookback_window=6,
    transaction_cost=0.001,
    lr=3e-4,
    gamma=0.99,
    clip_range=0.2,
    update_epochs=10,
    save_path="ppo_agent.pth",
):
    """
    Train PPO agent with proper train/val split.
    
    Args:
        train_returns: Training data (DataFrame)
        val_returns: Validation data (DataFrame, optional)
        total_episodes: Number of training episodes
        lookback_window: Months of history for state
        transaction_cost: Trading cost penalty
        lr: Learning rate
        gamma: Discount factor
        clip_range: PPO clipping epsilon
        update_epochs: Update iterations per rollout
        save_path: Where to save trained model
        
    Returns:
        Trained agent, training rewards, validation metrics
    """
    
    print("=" * 70)
    print("PPO TRAINING WITH PROPER TRAIN/VAL SPLIT")
    print("=" * 70)
    
    # Create training environment
    print(f"\nTraining on {len(train_returns)} months")
    if val_returns is not None:
        print(f"Validating on {len(val_returns)} months")
    
    env = PortfolioEnv(
        returns=train_returns,
        lookback_window=lookback_window,
        transaction_cost=transaction_cost,
    )
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    print(f"\nEnvironment:")
    print(f"  State dim: {state_dim}")
    print(f"  Action dim: {action_dim}")
    print(f"  Transaction cost: {transaction_cost*100:.2f}%")
    
    # Create agent
    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=lr,
        gamma=gamma,
        clip_range=clip_range,
    )
    
    # Training loop
    print(f"\nTraining for {total_episodes} episodes...")
    print("")
    
    train_rewards = []
    val_metrics_history = []
    best_val_sharpe = -np.inf
    
    for episode in range(total_episodes):
        
        # Rollout
        state, _ = env.reset()
        done = False
        
        states, actions, rewards = [], [], []
        next_states, log_probs, dones = [], [], []
        episode_reward = 0
        
        while not done:
            action, log_prob = agent.select_action(state, deterministic=False)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state if not done else state)
            log_probs.append(log_prob)
            dones.append(done)
            
            state = next_state
            episode_reward += reward
        
        train_rewards.append(episode_reward)
        
        # Update agent
        metrics = agent.update(
            states=states,
            actions=actions,
            old_log_probs=log_probs,
            rewards=rewards,
            next_states=next_states,
            dones=dones,
            epochs=update_epochs,
        )
        
        # Validation every 10 episodes
        if val_returns is not None and (episode + 1) % 10 == 0:
            from evaluate import evaluate_ppo
            val_results = evaluate_ppo(agent, val_returns, lookback_window, transaction_cost)
            val_sharpe = val_results["metrics"]["sharpe"]
            val_metrics_history.append(val_sharpe)
            
            # Save best model
            if val_sharpe > best_val_sharpe:
                best_val_sharpe = val_sharpe
                agent.save(save_path)
                saved_marker = " [SAVED]"
            else:
                saved_marker = ""
            
            avg_train_reward = np.mean(train_rewards[-10:])
            print(f"Episode {episode + 1}/{total_episodes}")
            print(f"  Train reward: {avg_train_reward:.4f}")
            print(f"  Val Sharpe: {val_sharpe:.2f}{saved_marker}")
            print(f"  Policy loss: {metrics['policy_loss']:.4f}")
            print(f"  Value loss: {metrics['value_loss']:.4f}")
            print("")
        
        elif val_returns is None and (episode + 1) % 10 == 0:
            avg_reward = np.mean(train_rewards[-10:])
            print(f"Episode {episode + 1}/{total_episodes}")
            print(f"  Avg reward (last 10): {avg_reward:.4f}")
            print(f"  Policy loss: {metrics['policy_loss']:.4f}")
            print(f"  Value loss: {metrics['value_loss']:.4f}")
            print("")
    
    # Save final model if no validation
    if val_returns is None:
        agent.save(save_path)
    
    print("=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    
    return agent, train_rewards, val_metrics_history


if __name__ == "__main__":
    """
    Main training script with proper splits.
    """
    
    # Load data
    print("Loading data...")
    import os
    from pathlib import Path
    
    # Get project root (parent of src directory)
    project_root = Path(__file__).parent.parent
    data_path = project_root / "data" / "returns_monthly.csv"
    
    returns = pd.read_csv(data_path, index_col=0, parse_dates=True)
    returns = returns.apply(pd.to_numeric, errors="coerce").dropna()
    
    print(f"✓ Loaded {len(returns)} months of data")
    print(f"  Assets: {list(returns.columns)}")
    print(f"  Date range: {returns.index[0]} to {returns.index[-1]}")
    
    # Split data
    train, val, test = split_data(returns, train_pct=0.7, val_pct=0.15, test_pct=0.15)
    
    print(f"\nData split:")
    print(f"  Train: {train.index[0]} to {train.index[-1]} ({len(train)} months)")
    print(f"  Val:   {val.index[0]} to {val.index[-1]} ({len(val)} months)")
    print(f"  Test:  {test.index[0]} to {test.index[-1]} ({len(test)} months)")
    
    # Train agent
    agent, train_rewards, val_metrics = train_ppo(
        train_returns=train,
        val_returns=val,
        total_episodes=100,
        lookback_window=6,
        transaction_cost=0.001,
        lr=3e-4,
        gamma=0.99,
        clip_range=0.2,
        update_epochs=10,
        save_path="ppo_agent.pth",
    )
    
    # Evaluate on all splits
    print("\n" + "=" * 70)
    print("FINAL EVALUATION")
    print("=" * 70)
    
    print("\n--- Training Set ---")
    train_results = compare_all(train, agent_path="ppo_agent.pth")
    print_comparison(train_results, dataset_name="Train")
    
    print("\n--- Validation Set ---")
    val_results = compare_all(val, agent_path="ppo_agent.pth")
    print_comparison(val_results, dataset_name="Validation")
    
    print("\n--- Test Set (FINAL BENCHMARK) ---")
    test_results = compare_all(test, agent_path="ppo_agent.pth")
    print_comparison(test_results, dataset_name="Test")

