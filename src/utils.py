# src/utils.py
import numpy as np
import pandas as pd

def equity_curve(returns: pd.DataFrame, weights: np.ndarray) -> pd.Series:
    # force numeric and aligned dot-product
    R = returns.apply(pd.to_numeric, errors="coerce").dropna()
    w = np.asarray(weights, dtype=float)
    port = pd.Series(R.values @ w, index=R.index)  # (T x N) @ (N,) -> (T,)
    return (1 + port).cumprod()

def summarize(equity: pd.Series) -> dict:
    rets = equity.pct_change().dropna()
    ann_ret = equity.iloc[-1]**(12/len(equity)) - 1
    ann_vol = rets.std() * np.sqrt(12)
    sharpe = float(ann_ret / (ann_vol + 1e-12))
    mdd = float(((equity / equity.cummax()) - 1).min())
    return {"ann_ret": float(ann_ret), "ann_vol": float(ann_vol), "sharpe": sharpe, "mdd": mdd}
