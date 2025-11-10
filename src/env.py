# src/env.py
"""
Basic RL Environment for Portfolio Optimization

This wraps your market data so an RL agent can interact with it step-by-step.
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces


class PortfolioEnv(gym.Env):
    """
    Simple portfolio environment.
    
    How it works:
    1. Agent sees current market conditions (state)
    2. Agent chooses portfolio weights (action)
    3. Environment moves forward 1 month
    4. Agent gets reward based on portfolio return
    """
    
    def __init__(self, returns, lookback_window=6, transaction_cost=0.001):
        """
        Args:
            returns: DataFrame of monthly returns (your data/returns_monthly.csv)
            lookback_window: How many months of history the agent sees
            transaction_cost: Penalty for trading (default 0.1%)
        """
        super().__init__()
        
        # Clean and store data
        self.returns_df = returns.apply(pd.to_numeric, errors="coerce").dropna()
        self.returns = self.returns_df.values  # Convert to numpy array
        
        # Dimensions
        self.T, self.N = self.returns.shape  # T = time steps, N = number of assets
        self.lookback = lookback_window
        self.cost = transaction_cost
        
        # Define what the agent sees and does
        # State = [average returns for each asset, volatility for each asset]
        state_size = 2 * self.N  # mean + std for each asset
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(state_size,), 
            dtype=np.float32
        )
        
        # Action = portfolio weights (must sum to 1)
        self.action_space = spaces.Box(
            low=0.0, 
            high=1.0, 
            shape=(self.N,), 
            dtype=np.float32
        )
        
        # Episode tracking
        self.current_step = None
        self.current_weights = None
    
    def reset(self, seed=None, options=None):
        """
        Start a new episode.
        
        Returns:
            state: What the agent observes
            info: Extra information (empty for now)
        """
        super().reset(seed=seed)
        
        # Start after lookback window (need history to compute state)
        self.current_step = self.lookback
        
        # Start with equal weights
        self.current_weights = np.ones(self.N) / self.N
        
        # Get initial state
        state = self._get_state()
        
        return state, {}
    
    def _get_state(self):
        """
        Compute what the agent sees.
        
        State = [avg_return_asset1, avg_return_asset2, ..., 
                 volatility_asset1, volatility_asset2, ...]
        """
        # Get last 'lookback' months of returns
        start_idx = max(0, self.current_step - self.lookback)
        recent_returns = self.returns[start_idx:self.current_step, :]
        
        # Compute statistics
        if recent_returns.shape[0] > 1:
            avg_returns = recent_returns.mean(axis=0)  # Average return per asset
            volatility = recent_returns.std(axis=0)    # Volatility per asset
        else:
            avg_returns = np.zeros(self.N)
            volatility = np.ones(self.N) * 0.01
        
        # Combine into state vector
        state = np.concatenate([avg_returns, volatility])
        
        return state.astype(np.float32)
    
    def step(self, action):
        """
        Agent takes an action (chooses weights), environment responds.
        
        Args:
            action: Portfolio weights chosen by agent
            
        Returns:
            next_state: What agent sees next
            reward: How good was the action
            terminated: Is episode done?
            truncated: Did we stop early? (always False)
            info: Extra details
        """
        # Normalize weights to sum to 1
        new_weights = np.clip(action, 0, 1)
        new_weights = new_weights / (new_weights.sum() + 1e-8)
        
        # Calculate how much we traded
        turnover = np.abs(new_weights - self.current_weights).sum()
        
        # Get this month's returns
        monthly_returns = self.returns[self.current_step, :]
        
        # Calculate portfolio return
        portfolio_return = (new_weights * monthly_returns).sum()
        
        # Reward = return - penalty for trading
        reward = portfolio_return - (self.cost * turnover)
        
        # Update state
        self.current_weights = new_weights
        self.current_step += 1
        
        # Check if we're done
        terminated = (self.current_step >= self.T)
        
        # Get next state
        if not terminated:
            next_state = self._get_state()
        else:
            next_state = np.zeros(self.observation_space.shape, dtype=np.float32)
        
        info = {
            "portfolio_return": float(portfolio_return),
            "turnover": float(turnover),
            "weights": new_weights.copy(),
        }
        
        return next_state, float(reward), terminated, False, info