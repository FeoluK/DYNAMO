# run.py
"""
Main entry point: Fetch data, train PPO, evaluate against baselines.
"""

import pandas as pd
from src.data import main as fetch_data
from src.evaluate import split_data, compare_all, print_comparison


def load_returns_clean(path="data/returns_monthly.csv"):
    """Load and clean returns data."""
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df = df.apply(pd.to_numeric, errors="coerce").dropna()
    return df


if __name__ == "__main__":
    """
    Full pipeline: Fetch data → Compare all strategies on test set
    """
    
    print("=" * 70)
    print("DYNAMO: Deep Yield-focused Adaptive Market Optimizer")
    print("=" * 70)
    
    # Fetch latest data
    print("\nFetching latest market data...")
    fetch_data()
    
    # Load returns
    returns = load_returns_clean()
    print(f"✓ Loaded {len(returns)} months of data")
    print(f"  Date range: {returns.index[0]} to {returns.index[-1]}")
    print(f"  Assets: {list(returns.columns)}")
    
    # Split data
    train, val, test = split_data(returns, train_pct=0.7, val_pct=0.15, test_pct=0.15)
    
    print(f"\nData split:")
    print(f"  Train: {train.index[0]} to {train.index[-1]} ({len(train)} months)")
    print(f"  Val:   {val.index[0]} to {val.index[-1]} ({len(val)} months)")
    print(f"  Test:  {test.index[0]} to {test.index[-1]} ({len(test)} months)")
    
    # Compare all strategies on test set
    print("\n" + "=" * 70)
    print("EVALUATING ALL STRATEGIES ON TEST SET")
    print("=" * 70)
    
    results = compare_all(
        test,
        agent_path="src/ppo_agent.pth",  # Load trained agent if exists
        lookback=6,
        transaction_cost=0.001,
    )
    
    print_comparison(results, dataset_name="Test")
    
    print("\n" + "=" * 70)
    print("To train PPO agent: python src/train.py")
    print("To run inference: python src/inference.py")
    print("=" * 70)
