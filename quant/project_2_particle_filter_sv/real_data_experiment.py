import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from particle_filter import particle_filter_sv
from diagnostics import plot_diagnostics

os.makedirs("data", exist_ok=True)
os.makedirs("results", exist_ok=True)

# ==========================
# Download real market data
# ==========================
ticker = "AAPL"
start_date = "2020-01-01"
end_date = "2024-01-01"

data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False)

# ==========================
# Handle MultiIndex columns safely
# ==========================
if isinstance(data.columns, pd.MultiIndex):
    if ("Adj Close", ticker) in data.columns:
        prices = data[("Adj Close", ticker)]
    else:
        prices = data[("Close", ticker)]
else:
    prices = data["Adj Close"] if "Adj Close" in data.columns else data["Close"]

prices = prices.dropna()

# ==========================
# Compute log returns
# ==========================
returns = np.log(prices / prices.shift(1)).dropna().values

pd.DataFrame(returns, columns=["returns"]).to_csv(
    "data/market_returns.csv", index=False
)

# ==========================
# Run particle filter
# ==========================
sigma_est, ess, loglikelihood = particle_filter_sv(
    returns,
    N=2000,
    tau=0.12,
    sigma0=0.2
)

# ==========================
# Plot estimated volatility
# ==========================
plt.figure(figsize=(10, 5))
plt.plot(sigma_est, label="Filtered Volatility (Particle Filter)")
plt.xlabel("Time")
plt.ylabel("Volatility")
plt.title(f"Particle Filter Volatility Estimate — {ticker}")
plt.legend()
plt.grid(True)
plt.savefig("results/real_volatility_pf.png", dpi=150)
plt.close()

# ==========================
# Diagnostics
# ==========================
plot_diagnostics(ess, loglikelihood)

print("✅ Real data particle filtering complete.")
print("📁 Results saved in /results")
