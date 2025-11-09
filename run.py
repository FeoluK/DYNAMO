# run.py
import pandas as pd
from src.data import main as fetch_data
from src.baselines import equal_weight, sixty_forty

def load_returns_clean(path="data/returns_monthly.csv"):
    df = pd.read_csv(path)
    # If a Date column exists, use it; otherwise use first column as index
    if "Date" in df.columns:
        df = df.rename(columns={"Date": "date"}).set_index("date")
    else:
        df = df.set_index(df.columns[0])
    # keep only numeric, drop rows with NaNs (e.g., partial months)
    df = df.apply(pd.to_numeric, errors="coerce").dropna()
    df.index = pd.to_datetime(df.index)
    return df

if __name__ == "__main__":
    fetch_data()  # ensures fresh files
    rets = load_returns_clean()

    eq_eqw, m_eqw, _ = equal_weight(rets)
    eq_6040, m_6040, _ = sixty_forty(rets)

    print("\n=== Baseline Metrics ===")
    print("Equal-Weight :", m_eqw)
    print("60/40        :", m_6040)
