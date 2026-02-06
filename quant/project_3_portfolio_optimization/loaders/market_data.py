import yfinance as yf
import pandas as pd
import os

def fetch_data(tickers, start="2019-01-01", end="2024-01-01"):
    os.makedirs("data", exist_ok=True)

    data = yf.download(tickers, start=start, end=end, progress=True)
    prices = data["Adj Close"].dropna()
    prices.to_csv("data/prices.csv")
    return prices


def load_returns():
    prices = pd.read_csv("data/prices.csv", index_col=0)
    returns = prices.pct_change().dropna()
    return returns
