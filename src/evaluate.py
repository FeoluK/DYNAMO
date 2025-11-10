# src/evaluate.py
"""
Unified evaluation for all strategies (PPO + baselines).
Run everything on the same test set for fair comparison.
"""

import numpy as np
import pandas as pd

# Handle imports from both src/ and project root
try:
    from env import PortfolioEnv
    from ppo_agent import PPOAgent
    from baselines import equal_weight, sixty_forty
    from utils import equity_curve, summarize
except ImportError:
    from src.env import PortfolioEnv
    from src.ppo_agent import PPOAgent
    from src.baselines import equal_weight, sixty_forty
    from src.utils import equity_curve, summarize


def split_data(returns, train_pct=0.7, val_pct=0.15, test_pct=0.15):
    """
    Split returns into train/val/test chronologically.
    
    Args:
        returns: DataFrame of returns
        train_pct: Fraction for training (default 70%)
        val_pct: Fraction for validation (default 15%)
        test_pct: Fraction for testing (default 15%)
    
    Returns:
        train_df, val_df, test_df
    """
    n = len(returns)
    train_end = int(n * train_pct)
    val_end = int(n * (train_pct + val_pct))
    
    train = returns.iloc[:train_end]
    val = returns.iloc[train_end:val_end]
    test = returns.iloc[val_end:]
    
    return train, val, test


def evaluate_ppo(agent, returns, lookback=6, transaction_cost=0.001):
    """
    Evaluate trained PPO agent on given returns.
    
    Args:
        agent: Trained PPOAgent
        returns: DataFrame of returns to evaluate on
        lookback: Lookback window
        transaction_cost: Transaction cost
        
    Returns:
        dict with equity curve and metrics
    """
    env = PortfolioEnv(returns, lookback_window=lookback, transaction_cost=transaction_cost)
    
    state, _ = env.reset()
    done = False
    
    weights_history = []
    portfolio_returns = []
    
    while not done:
        action, _ = agent.select_action(state, deterministic=True)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        weights_history.append(action)
        portfolio_returns.append(info["portfolio_return"])
        
        state = next_state
    
    # Build equity curve
    port_rets = pd.Series(portfolio_returns, index=returns.index[lookback:lookback+len(portfolio_returns)])
    equity = (1 + port_rets).cumprod()
    metrics = summarize(equity)
    
    return {
        "equity": equity,
        "metrics": metrics,
        "weights": np.array(weights_history),
        "returns": port_rets,
    }


def evaluate_baseline(returns, strategy_fn, strategy_name):
    """
    Evaluate baseline strategy.
    
    Args:
        returns: DataFrame of returns
        strategy_fn: Function that returns (equity, metrics, weights)
        strategy_name: Name for display
        
    Returns:
        dict with equity curve and metrics
    """
    equity, metrics, weights = strategy_fn(returns)
    
    return {
        "name": strategy_name,
        "equity": equity,
        "metrics": metrics,
        "weights": weights,
    }


def compare_all(returns, agent_path=None, lookback=6, transaction_cost=0.001):
    """
    Compare PPO agent against all baselines on same data.
    
    Args:
        returns: DataFrame of returns to evaluate on
        agent_path: Path to trained PPO agent (None = skip PPO)
        lookback: Lookback window for PPO
        transaction_cost: Transaction cost for PPO
        
    Returns:
        dict mapping strategy_name → results
    """
    results = {}
    
    # Evaluate baselines
    print("Evaluating baselines...")
    results["Equal-Weight"] = evaluate_baseline(returns, equal_weight, "Equal-Weight")
    results["60/40"] = evaluate_baseline(returns, sixty_forty, "60/40")
    
    # Evaluate PPO if agent exists
    if agent_path:
        print(f"Evaluating PPO agent from {agent_path}...")
        
        # Load agent
        state_dim = 2 * returns.shape[1]
        action_dim = returns.shape[1]
        agent = PPOAgent(state_dim=state_dim, action_dim=action_dim)
        agent.load(agent_path)
        
        results["PPO"] = evaluate_ppo(agent, returns, lookback, transaction_cost)
        results["PPO"]["name"] = "PPO"
    
    return results


def print_comparison(results, dataset_name="Test"):
    """
    Print comparison table of all strategies.
    
    Args:
        results: dict from compare_all()
        dataset_name: Name of dataset (Train/Val/Test)
    """
    print("\n" + "=" * 80)
    print(f"RESULTS ON {dataset_name.upper()} SET")
    print("=" * 80)
    print(f"{'Strategy':<15} {'Annual Ret':<12} {'Annual Vol':<12} {'Sharpe':<10} {'Max DD':<10}")
    print("-" * 80)
    
    for name, res in results.items():
        m = res["metrics"]
        print(f"{name:<15} {m['ann_ret']:>10.2%}  {m['ann_vol']:>10.2%}  {m['sharpe']:>8.2f}  {m['mdd']:>8.2%}")
    
    print("=" * 80)


if __name__ == "__main__":
    """
    Example usage: Evaluate on test set
    """
    # Load data
    from pathlib import Path
    project_root = Path(__file__).parent.parent
    data_path = project_root / "data" / "returns_monthly.csv"
    
    returns = pd.read_csv(data_path, index_col=0, parse_dates=True)
    returns = returns.apply(pd.to_numeric, errors="coerce").dropna()
    
    # Split data
    train, val, test = split_data(returns, train_pct=0.7, val_pct=0.15, test_pct=0.15)
    
    print(f"Data split:")
    print(f"  Train: {train.index[0]} to {train.index[-1]} ({len(train)} months)")
    print(f"  Val:   {val.index[0]} to {val.index[-1]} ({len(val)} months)")
    print(f"  Test:  {test.index[0]} to {test.index[-1]} ({len(test)} months)")
    
    # Compare all strategies on test set
    results = compare_all(test, agent_path="ppo_agent.pth")
    print_comparison(results, dataset_name="Test")

