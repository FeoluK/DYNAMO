# DYNAMO: Deep Yield-focused Adaptive Market Optimizer

**Reinforcement Learning for Portfolio Optimization**

Train a PPO (Proximal Policy Optimization) agent to dynamically allocate portfolios across multiple asset classes, beating static baseline strategies.

---

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train agent with proper train/val/test splits
python src/train.py

# Evaluate all strategies on test set
python run.py

# Run inference with trained model
python src/inference.py
```

---

## 📁 Project Structure

```
DYNAMO/
├── data/
│   ├── prices_monthly.csv      # Historical prices
│   └── returns_monthly.csv     # Monthly returns
├── src/
│   ├── data.py                 # Fetch data from Yahoo Finance
│   ├── env.py                  # Portfolio RL environment
│   ├── ppo_agent.py            # PPO agent implementation
│   ├── train.py                # Training with train/val/test splits
│   ├── evaluate.py             # Unified evaluation framework
│   ├── inference.py            # Load model and run predictions
│   ├── baselines.py            # Equal-weight & 60/40 strategies
│   └── utils.py                # Metrics (Sharpe, drawdown, etc.)
├── run.py                      # Main entry point
└── requirements.txt
```

---

## 📊 Results (Test Set)

| Strategy | Annual Return | Volatility | Sharpe | Max Drawdown |
|----------|--------------|------------|--------|--------------|
| **PPO** | **38.54%** | **8.07%** | **4.78** | **-1.62%** |
| Equal-Weight | 23.54% | 11.19% | 2.10 | -4.66% |
| 60/40 | 13.45% | 11.08% | 1.21 | -6.30% |

**PPO wins on all metrics** - highest returns, lowest volatility, best Sharpe ratio, smallest drawdowns.

---

## 🎯 How It Works

### **Data Split**
```
Total: 134 months (Oct 2014 - Nov 2024)
├── Train:  93 months (70%) - Learn optimal policy
├── Val:    20 months (15%) - Early stopping
└── Test:   21 months (15%) - Final benchmark
```

### **Environment**
- **State**: Recent returns + volatilities for each asset
- **Action**: Portfolio weights (sum to 1, no shorting)
- **Reward**: Portfolio return - transaction costs

### **PPO Agent**
- **Policy Network**: Maps state → portfolio weights (softmax)
- **Value Network**: Estimates expected future returns
- **Training**: Clipped objective (ε=0.2), entropy bonus, GAE advantages

### **Training Pipeline**
1. **Rollout**: Run policy, collect trajectories
2. **Update**: Improve policy and value networks (PPO)
3. **Validate**: Evaluate on validation set every 10 episodes
4. **Save**: Keep best model by validation Sharpe ratio

---

## 🔧 Key Features

✅ **Proper train/val/test splits** - No data leakage  
✅ **Validation-based early stopping** - Prevents overfitting  
✅ **Fair benchmarking** - All strategies on same test set  
✅ **Transaction costs** - Penalizes excessive trading  
✅ **Inference mode** - Load and run trained models  

---

## 🛠️ Customization

### **Change Assets**

Edit `src/data.py`:
```python
TICKERS = ["SPY", "TLT", "GLD", "XLE", "XLK", "BTC-USD"]
```

### **Hyperparameters**

Edit `src/train.py` or modify training call:
```python
train_ppo(
    train_returns=train,
    val_returns=val,
    total_episodes=200,        # More training
    lookback_window=12,        # 1 year history
    transaction_cost=0.002,    # 0.2% trading cost
    lr=1e-4,                   # Learning rate
    gamma=0.99,                # Discount factor
    clip_range=0.2,            # PPO clipping
    update_epochs=10,          # Updates per rollout
)
```

### **Add New Baselines**

Add to `src/baselines.py`:
```python
def risk_parity(returns):
    """Allocate inversely to volatility."""
    vols = returns.std()
    w = (1/vols) / (1/vols).sum()
    eq = equity_curve(returns, w)
    return eq, summarize(eq), w
```

---

## 📖 Documentation

- **`README.md`** - This file (quick start)
- **`SUMMARY.md`** - Implementation details
- **`ARCHITECTURE.md`** - System design and algorithms

---

## 🎓 Technical Details

### **PPO Implementation**
- Policy: state (12) → hidden (64) → hidden (64) → logits (6) → softmax
- Value: state (12) → hidden (64) → hidden (64) → value (1)
- Advantages: A = r + γ*V(s') - V(s)
- Clipped objective: Limits policy changes to ±20%

### **Environment Design**
- State: [avg_returns (6), volatilities (6)] from last 6 months
- Action: [w_SPY, w_TLT, w_GLD, w_XLE, w_XLK, w_BTC]
- Reward: portfolio_return - 0.001 × turnover

### **Assets (6)**
- SPY: S&P 500 ETF
- TLT: Long-term Treasury Bonds
- GLD: Gold
- XLE: Energy Sector
- XLK: Technology Sector
- BTC-USD: Bitcoin

---

## 🚀 Next Steps

**Improve Performance:**
- Add more features (momentum, correlation, macro indicators)
- Try LSTM/Transformers for temporal patterns
- Ensemble multiple agents
- Multi-objective optimization

**Extend Assets:**
- International equities
- Real estate (REITs)
- Commodities
- More cryptocurrencies

**Deploy:**
- Live trading interface
- Real-time monitoring
- Automated retraining
- Risk management system

---

## ⚠️ Disclaimer

This is a research project for educational purposes. Not financial advice. Past performance doesn't guarantee future results. Always do your own research before making investment decisions.

---

**Happy optimizing! 🚀📈**
