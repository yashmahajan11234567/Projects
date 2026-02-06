import yfinance as yf
import pandas as pd
import os

def fetch_nifty50(start="2019-01-01", end="2024-01-01"):
    os.makedirs("data", exist_ok=True)

    nifty = yf.download(
        "^NSEI",
        start=start,
        end=end,
        auto_adjust=True,
        progress=True
    )

    # Handle MultiIndex safely
    if isinstance(nifty.columns, pd.MultiIndex):
        prices = nifty["Close"]["^NSEI"]
    else:
        prices = nifty["Close"]

    prices = prices.dropna()
    prices.to_csv("data/nifty50.csv")
    return prices


def load_nifty_returns():
    prices = pd.read_csv("data/nifty50.csv", index_col=0)
    returns = prices.pct_change().dropna()
    return returns
