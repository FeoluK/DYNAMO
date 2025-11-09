# src/baselines.py
import numpy as np
import pandas as pd
from .utils import equity_curve, summarize

STOCKS = ["SPY","XLK"]
BONDS  = ["TLT"]

def equal_weight(returns: pd.DataFrame):
    n = returns.shape[1]
    w = np.repeat(1/n, n)
    eq = equity_curve(returns, w)
    return eq, summarize(eq), w

def sixty_forty(returns: pd.DataFrame):
    cols = returns.columns.tolist()
    w = np.zeros(len(cols))
    # split 60% equally across STOCKS, 40% in BONDS
    for s in STOCKS:
        if s in cols: w[cols.index(s)] += 0.60 / max(1, sum(c in cols for c in STOCKS))
    for b in BONDS:
        if b in cols: w[cols.index(b)] += 0.40 / max(1, sum(c in cols for c in BONDS))
    w = w / w.sum()  # normalize if some tickers missing
    eq = equity_curve(returns, w)
    return eq, summarize(eq), w
