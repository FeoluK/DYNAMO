# src/data.py
from pathlib import Path
import time
import pandas as pd
import yfinance as yf

TICKERS = ["SPY","TLT","GLD","XLE","XLK","BTC-USD"]

def fetch_one(ticker, start="2010-01-01", end=None, retries=3, wait=1.5):
    for i in range(retries):
        try:
            # auto_adjust=False so "Adj Close" exists
            df = yf.download(
                ticker,
                start=start,
                end=end,
                interval="1mo",
                auto_adjust=False,
                progress=False,
                threads=False,   # avoid sqlite contention
            )
            if df.empty:
                raise RuntimeError(f"No data for {ticker}")
            # use Adj Close; fallback to Close if needed
            col = "Adj Close" if "Adj Close" in df.columns else "Close"
            out = df[[col]].rename(columns={col: ticker})
            return out
        except Exception as e:
            if i == retries - 1:
                raise
            time.sleep(wait * (i + 1))
    raise RuntimeError(f"Failed to fetch {ticker}")

def fetch_monthly_prices(start="2010-01-01", end=None):
    frames = []
    for t in TICKERS:
        print(f"Fetching {t}...")
        frames.append(fetch_one(t, start=start, end=end))
    prices = pd.concat(frames, axis=1)
    prices.index = prices.index.to_period("M").to_timestamp("M")  # month-end index
    return prices

def to_monthly_returns(prices: pd.DataFrame) -> pd.DataFrame:
    returns = prices.pct_change().dropna()
    return returns

def save(df: pd.DataFrame, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)

def main():
    prices = fetch_monthly_prices()
    # align on months where all series are present
    prices = prices.dropna(how="any")
    returns = to_monthly_returns(prices)
    save(prices, "data/prices_monthly.csv")
    save(returns, "data/returns_monthly.csv")
    print("Saved: data/prices_monthly.csv, data/returns_monthly.csv")

if __name__ == "__main__":
    main()
