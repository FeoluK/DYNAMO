# src/ppo_agent.py
"""
PPO Agent built from scratch using PyTorch.

This implements Section 4 of the background document:
- Policy network (outputs portfolio weights)
- Value network (estimates state value)
- PPO training algorithm with clipped objective
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical


class PolicyNetwork(nn.Module):
    """
    Policy Network: Maps state → portfolio weights
    
    This is π_θ(a|s) from the paper.
    Takes in state (market conditions) and outputs logits that become weights.
    """
    
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        """
        Args:
            state_dim: Size of state vector (e.g., 12 = 6 avg returns + 6 volatilities)
            action_dim: Number of assets (e.g., 6)
            hidden_dim: Size of hidden layers
        """
        super(PolicyNetwork, self).__init__()
        
        # Neural network layers
        # state → hidden → hidden → logits
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),  # Output: logits for each asset
        )
        
    def forward(self, state):
        """
        Forward pass: state → logits
        
        Args:
            state: Tensor of shape (batch_size, state_dim)
            
        Returns:
            logits: Tensor of shape (batch_size, action_dim)
        """
        logits = self.network(state)
        return logits
    
    def get_action(self, state):
        """
        Get action (portfolio weights) from state.
        
        This applies softmax to logits to ensure weights sum to 1.
        
        Args:
            state: Tensor of shape (state_dim,) or (batch_size, state_dim)
            
        Returns:
            action: Portfolio weights (sums to 1)
            log_prob: Log probability of this action (needed for PPO)
        """
        # Get logits from network
        logits = self.forward(state)
        
        # Apply softmax to get weights that sum to 1
        # This ensures: w_i >= 0 and Σw_i = 1
        probs = torch.softmax(logits, dim=-1)
        
        # Sample from the distribution (during training, add randomness)
        # During evaluation, we can just take the probabilities directly
        dist = Categorical(probs)
        
        # For continuous weights, we'll just use the probabilities directly
        # (In true continuous action spaces, you'd use a different distribution)
        action = probs
        
        # Compute log probability (needed for PPO update)
        # For our case, we'll compute it differently (see below)
        log_prob = torch.log(probs + 1e-8)  # Add small epsilon to avoid log(0)
        
        return action, log_prob


class ValueNetwork(nn.Module):
    """
    Value Network: Maps state → expected total future return
    
    This is V_φ(s) from the paper.
    
    ============================================================================
    WHAT DOES V(s) REPRESENT?
    ============================================================================
    
    V(s) is NOT just the next month's return. It's the SUM of ALL future returns!
    
    Mathematically:
        V(s_t) = E[r_{t+1} + γ*r_{t+2} + γ²*r_{t+3} + γ³*r_{t+4} + ... | state s_t]
                 ↑       ↑           ↑            ↑             ↑
               next   month      month        month        infinite
               month  after      after        after        horizon!
    
    Example:
        If V(s_10) = 0.05, this means:
        "Starting from state 10 (month 10), if I follow my policy,
         I expect to earn 5% TOTAL over ALL remaining months"
        
        This includes:
        - Month 11: +1.0%
        - Month 12: +0.99%  (discounted by γ)
        - Month 13: +0.98%
        - Month 14: +0.97%
        - ... all the way to the end!
    
    ============================================================================
    HOW DOES IT KNOW ABOUT THE ENTIRE FUTURE?
    ============================================================================
    
    Through the Bellman equation (recursive property):
        V(s_t) = r_{t+1} + γ*V(s_{t+1})
    
    The value at state t equals:
        Immediate reward + Discounted value of next state
    
    This means V(s_{t+1}) contains information about V(s_{t+2}), which contains
    info about V(s_{t+3}), and so on. Information propagates BACKWARDS!
    
    Training process:
        Month 133: V(s_133) learns about immediate future (end of data)
        Month 132: V(s_132) learns from V(s_133) → knows 2 steps ahead
        Month 131: V(s_131) learns from V(s_132) → knows 3 steps ahead
        ...
        Month 10: V(s_10) learns from V(s_11) → knows 123 steps ahead!
    
    ============================================================================
    IS THIS A PREDICTION OR REALITY?
    ============================================================================
    
    V(s) is a PREDICTION (the value network's OPINION), NOT reality!
    
    Example:
        State: Market just crashed, high volatility
        V(state) = 0.08
        
        This means the network THINKS: "From here, I expect 8% total return"
        
        But what ACTUALLY happens might be:
        - 15% total return (network was too pessimistic!)
        - 3% total return (network was too optimistic!)
        
    We train the network to make better predictions by comparing:
        Prediction: V(s_t)
        Reality: r_{t+1} + γ*V(s_{t+1})  (what we actually experienced)
        Loss: (prediction - reality)²
    
    Over time, the network learns to predict accurately!
    
    ============================================================================
    """
    
    def __init__(self, state_dim, hidden_dim=64):
        """
        Args:
            state_dim: Size of state vector (e.g., 12)
            hidden_dim: Size of hidden layers (64 neurons)
        """
        super(ValueNetwork, self).__init__()
        
        # Neural network layers
        # state → hidden → hidden → value (single number)
        #
        # Architecture:
        #   Input: [avg_returns, volatilities] (12 numbers)
        #   Hidden layer 1: 64 neurons with ReLU activation
        #   Hidden layer 2: 64 neurons with ReLU activation  
        #   Output: Single value (expected total future return)
        #
        # Example flow:
        #   [0.02, -0.01, 0.03, ...] → [64 neurons] → [64 neurons] → 0.0523
        #                                                                ↑
        #                                                    "Expect 5.23% total return"
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),  # Output: single value
        )
        
    def forward(self, state):
        """
        Forward pass: state → value
        
        Takes in market conditions, outputs expected total future return.
        
        Args:
            state: Tensor of shape (batch_size, state_dim)
                  e.g., [[0.02, -0.01, 0.03, ...],  ← State 1
                         [0.025, -0.005, 0.035, ...],  ← State 2
                         ...]
            
        Returns:
            value: Tensor of shape (batch_size,)
                  e.g., [0.0523, 0.0467, ...]  ← Expected returns for each state
        """
        value = self.network(state)
        return value.squeeze(-1)  # Remove last dimension (batch_size,)


class PPOAgent:
    """
    Complete PPO Agent.
    
    Combines policy and value networks with PPO training algorithm.
    """
    
    def __init__(
        self,
        state_dim,
        action_dim,
        lr=3e-4,
        gamma=0.99,
        clip_range=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
    ):
        """
        Initialize PPO agent.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space (number of assets)
            lr: Learning rate
            gamma: Discount factor (how much we value future rewards)
            clip_range: Clipping parameter for PPO (epsilon in paper)
            value_coef: Coefficient for value loss
            entropy_coef: Coefficient for entropy bonus (encourages exploration)
        """
        # Store hyperparameters
        self.gamma = gamma
        self.clip_range = clip_range
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        
        # Create networks
        self.policy = PolicyNetwork(state_dim, action_dim)
        self.value = ValueNetwork(state_dim)
        
        # Create optimizers (one for each network)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.value_optimizer = optim.Adam(self.value.parameters(), lr=lr)
        
        print("PPO Agent initialized:")
        print(f"  State dim: {state_dim}")
        print(f"  Action dim: {action_dim}")
        print(f"  Clip range: {clip_range}")
        print(f"  Gamma: {gamma}")
    
    def select_action(self, state, deterministic=False):
        """
        Select action given state.
        
        Args:
            state: numpy array of shape (state_dim,)
            deterministic: If True, use best action (no randomness)
            
        Returns:
            action: numpy array of portfolio weights
            log_prob: log probability of action (for training)
        """
        # Convert state to tensor
        state_tensor = torch.FloatTensor(state).unsqueeze(0)  # Add batch dimension
        
        # Get action from policy
        with torch.no_grad():  # Don't track gradients during action selection
            logits = self.policy(state_tensor)
            probs = torch.softmax(logits, dim=-1)
            
            if deterministic:
                # Use most likely action (evaluation mode)
                action = probs
            else:
                # Add some noise for exploration (training mode)
                dist = Categorical(probs)
                # For portfolio weights, we use the probabilities directly
                action = probs
            
            # Compute log probability
            log_prob = torch.log(probs + 1e-8).sum(dim=-1)
        
        return action.squeeze(0).numpy(), log_prob.item()
    
    def compute_advantages(self, rewards, values, next_values, dones):
        """
        Compute advantages using the fundamental RL advantage formula.
        
        INTUITION: Advantage answers "Was this action better or worse than expected?"
        
        The formula: A_t = r_{t+1} + γ*V(s_{t+1}) - V(s_t)
        
        Breaking it down:
        - V(s_t): "What I EXPECTED when I started" (baseline expectation)
            → This is the value network's PREDICTION of total future return
            → Includes ALL future rewards: r_{t+1} + γ*r_{t+2} + γ²*r_{t+3} + ...
        
        - r_{t+1}: "What I EARNED immediately" (the journey/experience)
            → This is the ACTUAL reward we got this timestep
            → Without this, we'd miss whether we earned/lost money along the way
        
        - γ*V(s_{t+1}): "What the FUTURE looks like from where I ended up"
            → V(s_{t+1}) represents ALL future returns from the next state onwards
            → γ=0.99 means future is worth 99% of present
            → This captures whether we "set up for future gains" or "ruined our position"
        
        EXAMPLE:
            Starting state: Market neutral, V(s_t) = 0.01 (expect 1% total)
            Action: Buy stocks aggressively
            Reward: r = -0.02 (-2% this month, ouch!)
            Next state: Positioned for bull run, V(s_{t+1}) = 0.25 (expect 25% total!)
            
            Advantage = -0.02 + 0.99*0.25 - 0.01
                      = -0.02 + 0.2475 - 0.01
                      = 0.2175  ← Highly positive! Short-term pain, long-term gain!
        
        Why this works:
        - Positive advantage → action was better than baseline, do MORE of it
        - Negative advantage → action was worse than baseline, do LESS of it
        - Zero advantage → action was exactly as expected, no change needed
        
        Args:
            rewards: List of rewards (what we actually earned each step)
            values: List of V(s_t) - predictions for starting states
            next_values: List of V(s_{t+1}) - predictions for ending states
            dones: List of done flags (episode ended?)
            
        Returns:
            advantages: Tensor of advantages (normalized for stability)
        """
        advantages = []
        
        for i in range(len(rewards)):
            # If episode is done, there's no future, so V(s_{t+1}) = 0
            next_value = 0 if dones[i] else next_values[i]
            
            # THE CORE ADVANTAGE FORMULA
            # This answers: "Was this action better/worse than what I expected?"
            #
            # rewards[i]: What we ACTUALLY got (the journey)
            # next_value: Where we ENDED UP (discounted future)
            # values[i]: What we EXPECTED (the baseline)
            #
            # If (journey + destination) > starting_expectations → Positive advantage
            advantage = rewards[i] + self.gamma * next_value - values[i]
            advantages.append(advantage)
        
        # Normalize advantages for training stability
        # This makes positive and negative advantages have similar scales
        advantages = torch.FloatTensor(advantages)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages
    
    def update(self, states, actions, old_log_probs, rewards, next_states, dones, epochs=10):
        """
        Update policy and value networks using PPO algorithm.
        
        This implements the complete training loop from Section 4.7:
        1. Compute advantages (what worked vs what didn't)
        2. Update policy with clipped objective (do more of what worked, but don't change too fast)
        3. Update value network (make better predictions)
        
        ============================================================================
        DEEP DIVE: What's Happening Here
        ============================================================================
        
        We collected data during ROLLOUT:
        - states: [s_6, s_7, ..., s_133]  (128 states we visited)
        - actions: [a_6, a_7, ..., a_133]  (128 actions we took)
        - rewards: [r_7, r_8, ..., r_134]  (128 rewards we earned)
        - old_log_probs: [lp_6, lp_7, ..., lp_133]  (how likely were those actions?)
        
        Now we LEARN from this data by:
        1. Computing how good each action was (advantages)
        2. Adjusting policy to do more of good actions, less of bad ones
        3. But preventing overly aggressive changes (clipping!)
        
        ============================================================================
        WHY LOG PROBABILITIES?
        ============================================================================
        
        We use log probabilities for numerical stability and easy math:
        
        Normal probabilities can be tiny:
            prob = 0.00000123  (hard to work with!)
        
        Log probabilities are reasonable numbers:
            log_prob = log(0.00000123) = -13.6  (much better!)
        
        To compute ratio = new_prob / old_prob, we use:
            ratio = exp(log_new - log_old)
        
        Why this works:
            ratio = new_prob / old_prob
                  = exp(log(new_prob)) / exp(log(old_prob))
                  = exp(log(new_prob) - log(old_prob))  ← Subtraction instead of division!
        
        ============================================================================
        
        Args:
            states: List of states [s_t] we visited
            actions: List of actions [a_t] we took
            old_log_probs: List of log(π_θ_old(a_t|s_t)) from rollout
            rewards: List of rewards [r_{t+1}] we earned
            next_states: List of next states [s_{t+1}] we reached
            dones: List of done flags
            epochs: Number of times to update on this data (typically 10)
            
        Returns:
            dict with training metrics
        """
        # Convert everything to tensors for PyTorch
        states = torch.FloatTensor(np.array(states))
        actions = torch.FloatTensor(np.array(actions))
        old_log_probs = torch.FloatTensor(old_log_probs)
        next_states = torch.FloatTensor(np.array(next_states))
        
        # ========================================================================
        # STEP 1: Compute state values (these are PREDICTIONS, not reality!)
        # ========================================================================
        
        # V(s_t) = "Expected total return if I start from state s_t and follow my policy"
        # V(s_{t+1}) = "Expected total return from the next state"
        #
        # These are the value network's OPINIONS/GUESSES about future returns.
        # They incorporate the ENTIRE future: r_{t+1} + γ*r_{t+2} + γ²*r_{t+3} + ...
        #
        # Example:
        #   V(s_50) = 0.10 → "From state 50, I expect 10% total return over all future"
        #   This includes months 51, 52, 53, ..., all the way to the end!
        #
        # The value network learns this through bootstrapping:
        #   V(s_t) ≈ r_{t+1} + γ*V(s_{t+1})
        #   So V(s_50) contains info from V(s_51), which contains info from V(s_52), etc.
        
        with torch.no_grad():
            values = self.value(states).numpy()       # V(s_t) for all timesteps
            next_values = self.value(next_states).numpy()  # V(s_{t+1}) for all timesteps
        
        # ========================================================================
        # STEP 2: Compute advantages (what worked vs what didn't)
        # ========================================================================
        
        # Advantages tell us: "Was each action better or worse than expected?"
        #
        # A_t = r_{t+1} + γ*V(s_{t+1}) - V(s_t)
        #
        # Positive advantage: Action was BETTER than expected → Increase its probability
        # Negative advantage: Action was WORSE than expected → Decrease its probability
        #
        # This captures BOTH immediate payoff (r) AND long-term consequences (V(s_{t+1}))
        
        advantages = self.compute_advantages(rewards, values, next_values, dones)
        
        # Compute returns (targets for value network training)
        # returns ≈ r_{t+1} + γ*V(s_{t+1}) (what actually happened)
        returns = advantages + torch.FloatTensor(values)
        
        # ========================================================================
        # STEP 3: Update networks multiple times on the same data
        # ========================================================================
        
        # Why multiple epochs?
        # - Collecting data is expensive (we ran the policy through the environment)
        # - Let's learn thoroughly from it!
        # - But not too much (risk overfitting if epochs too high)
        
        for epoch in range(epochs):
            
            # ====================================================================
            # STEP 3A: UPDATE POLICY NETWORK (the heart of PPO!)
            # ====================================================================
            
            # Get new log probabilities from CURRENT policy
            # (The policy has changed slightly since the rollout!)
            logits = self.policy(states)
            probs = torch.softmax(logits, dim=-1)
            new_log_probs = torch.log(probs + 1e-8).sum(dim=-1)
            
            # ----------------------------------------------------------------
            # Compute RATIO: How much has the policy changed?
            # ----------------------------------------------------------------
            
            # ratio = π_θ(a|s) / π_θ_old(a|s)
            #       = exp(log(π_θ) - log(π_θ_old))
            #       = exp(new_log_probs - old_log_probs)
            #
            # What ratio means:
            #   ratio = 1.0  → Policy hasn't changed
            #   ratio = 1.5  → New policy is 50% more likely to take this action
            #   ratio = 0.6  → New policy is 40% less likely to take this action
            #   ratio = 3.0  → New policy is 3x more likely (BIG change!)
            #
            # Example:
            #   old_prob = 10%, new_prob = 15%
            #   ratio = 0.15/0.10 = 1.5
            
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            # ----------------------------------------------------------------
            # PPO CLIPPED OBJECTIVE (the innovation that makes PPO stable!)
            # ----------------------------------------------------------------
            
            # We compute TWO versions of the objective:
            #
            # surrogate1: Unclipped version
            #   = ratio * advantage
            #   = "How much do we want to change the policy?"
            #   If advantage is positive and ratio is large, this could be HUGE
            #
            # surrogate2: Clipped version
            #   = clipped_ratio * advantage
            #   = "Same, but don't allow extreme changes"
            #   We clip ratio to [1-ε, 1+ε] = [0.8, 1.2] with ε=0.2
            #   This limits policy changes to ±20% per update
            #
            # We take the MINIMUM (most conservative) of the two.
            #
            # Example 1: Good action, policy wants to increase too much
            #   advantage = +0.05 (action was good!)
            #   ratio = 2.5 (policy wants to make it 2.5x more likely)
            #   
            #   surrogate1 = 2.5 * 0.05 = 0.125  (aggressive update)
            #   surrogate2 = 1.2 * 0.05 = 0.06   (clipped to 20% change)
            #   
            #   min(0.125, 0.06) = 0.06  ← Take the conservative one
            #   We increase the probability, but only by 20%, not 150%!
            #
            # Example 2: Bad action, policy wants to decrease too much
            #   advantage = -0.03 (action was bad)
            #   ratio = 0.4 (policy wants to make it 60% less likely)
            #   
            #   surrogate1 = 0.4 * -0.03 = -0.012  (aggressive)
            #   surrogate2 = 0.8 * -0.03 = -0.024  (clipped)
            #   
            #   min(-0.012, -0.024) = -0.024  ← More negative, but conservative
            #   We decrease, but only by 20%, not 60%!
            
            surrogate1 = ratio * advantages
            surrogate2 = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * advantages
            
            # Take the minimum (most conservative update)
            # Negate because we minimize loss (which maximizes the objective)
            policy_loss = -torch.min(surrogate1, surrogate2).mean()
            
            # Add entropy bonus (encourages exploration)
            # Entropy = how "random" the policy is
            # High entropy = policy is uncertain, explores different actions
            # Low entropy = policy is confident, always does the same thing
            # We want some randomness so the agent doesn't get stuck!
            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()
            policy_loss = policy_loss - self.entropy_coef * entropy
            
            # Update policy network via gradient descent
            self.policy_optimizer.zero_grad()
            policy_loss.backward()  # Compute gradients
            self.policy_optimizer.step()  # Update network weights
            
            # ====================================================================
            # STEP 3B: UPDATE VALUE NETWORK (make better predictions!)
            # ====================================================================
            
            # The value network's job: predict V(s) accurately
            #
            # Currently it predicts: V(s_t)
            # The target (what actually happened): r_{t+1} + γ*V(s_{t+1})
            #
            # We train it like a regression problem: minimize (prediction - target)²
            #
            # Example:
            #   State s_7: [market conditions]
            #   Prediction: V(s_7) = 0.01 (network thinks "expect 1% future return")
            #   What happened: r_8 + γ*V(s_8) = 0.025 + 0.99*0.015 = 0.03985
            #   Target: 0.03985 (we actually got 4% total value)
            #   
            #   Loss: (0.01 - 0.03985)² = 0.000887
            #   
            #   Gradient descent adjusts network weights
            #   Next time, V(s_7) will predict closer to 0.04!
            #
            # Over time, the value network gets better at predicting future returns.
            # This helps compute better advantages!
            
            values_pred = self.value(states)
            
            # MSE loss between prediction and target
            value_loss = ((values_pred - returns) ** 2).mean()
            
            # Update value network
            self.value_optimizer.zero_grad()
            value_loss.backward()
            self.value_optimizer.step()
        
        # ========================================================================
        # What just happened? (Summary for one timestep)
        # ========================================================================
        #
        # Before update:
        #   Policy: "In volatile markets, allocate 30% to bonds"
        #   Value: "Volatile markets → expect 1% future return"
        #
        # Rollout data (month 7):
        #   State: Volatile market
        #   Action: 30% bonds
        #   Reward: +2.5% (did well!)
        #   Next state: Good market positioning
        #   V(s_7) = 0.01, V(s_8) = 0.03
        #
        # Advantage:
        #   A_7 = 0.025 + 0.99*0.03 - 0.01 = 0.0447 (very positive!)
        #
        # Policy update:
        #   Old policy: 30% probability of choosing that action
        #   Ratio wants: 1.5 (50% increase)
        #   Clipped to: 1.2 (20% increase)
        #   New policy: ~36% probability (was 30%)
        #
        # Value update:
        #   Prediction: V(s_7) = 0.01
        #   Target: 0.025 + 0.99*0.03 = 0.0545
        #   After update: V(s_7) moves toward 0.0545
        #
        # Result:
        #   Policy learned: "30% bonds in volatile markets is good, do slightly more"
        #   Value learned: "Volatile markets are worth more than I thought"
        #
        # ========================================================================
        
        # Return metrics for logging
        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.item(),
            "avg_advantage": advantages.mean().item(),
        }
    
    def save(self, path):
        """Save agent to file."""
        torch.save({
            'policy': self.policy.state_dict(),
            'value': self.value.state_dict(),
        }, path)
        print(f"Model saved to {path}")
    
    def load(self, path):
        """Load agent from file."""
        checkpoint = torch.load(path)
        self.policy.load_state_dict(checkpoint['policy'])
        self.value.load_state_dict(checkpoint['value'])
        print(f"Model loaded from {path}")