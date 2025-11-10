# src/inference.py
"""
Load trained PPO agent and run inference on new data.
"""

import numpy as np
import pandas as pd
from ppo_agent import PPOAgent
from env import PortfolioEnv


def load_agent(model_path, state_dim, action_dim):
    """
    Load trained PPO agent from file.
    
    Args:
        model_path: Path to saved model
        state_dim: State dimension
        action_dim: Action dimension
        
    Returns:
        Loaded PPOAgent
    """
    agent = PPOAgent(state_dim=state_dim, action_dim=action_dim)
    agent.load(model_path)
    return agent


def run_inference(agent, returns, lookback=6, transaction_cost=0.001):
    """
    Run trained agent on returns data.
    
    Args:
        agent: Trained PPOAgent
        returns: DataFrame of returns
        lookback: Lookback window
        transaction_cost: Transaction cost
        
    Returns:
        DataFrame with weights, returns, equity curve
    """
    env = PortfolioEnv(returns, lookback_window=lookback, transaction_cost=transaction_cost)
    
    state, _ = env.reset()
    done = False
    
    results = []
    
    while not done:
        # Get action from agent
        action, _ = agent.select_action(state, deterministic=True)
        
        # Take step
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Store results
        results.append({
            "weights": action,
            "portfolio_return": info["portfolio_return"],
            "turnover": info["turnover"],
        })
        
        state = next_state
    
    # Build results DataFrame
    dates = returns.index[lookback:lookback+len(results)]
    
    weights_df = pd.DataFrame(
        [r["weights"] for r in results],
        index=dates,
        columns=returns.columns,
    )
    
    returns_series = pd.Series(
        [r["portfolio_return"] for r in results],
        index=dates,
        name="portfolio_return"
    )
    
    turnover_series = pd.Series(
        [r["turnover"] for r in results],
        index=dates,
        name="turnover"
    )
    
    equity = (1 + returns_series).cumprod()
    equity.name = "equity"
    
    return weights_df, returns_series, equity, turnover_series


if __name__ == "__main__":
    """
    Example: Load agent and run on data.
    """
    
    # Load data
    from pathlib import Path
    project_root = Path(__file__).parent.parent
    data_path = project_root / "data" / "returns_monthly.csv"
    
    returns = pd.read_csv(data_path, index_col=0, parse_dates=True)
    returns = returns.apply(pd.to_numeric, errors="coerce").dropna()
    
    # Load agent
    state_dim = 2 * returns.shape[1]
    action_dim = returns.shape[1]
    agent = load_agent("ppo_agent.pth", state_dim, action_dim)
    
    # Run inference
    weights, portfolio_returns, equity, turnover = run_inference(agent, returns)
    
    print("Inference complete!")
    print(f"\nPortfolio weights (first 5 months):")
    print(weights.head())
    print(f"\nFinal equity: {equity.iloc[-1]:.2f}")
    print(f"Total return: {(equity.iloc[-1] - 1)*100:.2f}%")
    print(f"Avg turnover: {turnover.mean()*100:.2f}%")

