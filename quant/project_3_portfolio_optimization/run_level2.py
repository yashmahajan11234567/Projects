import matplotlib.pyplot as plt
import os

from loaders.market_data import fetch_data, load_returns
from optimization.constrained_optimizer import optimize_portfolio

# ==========================
# Assets
# ==========================
tickers = [
    "RELIANCE.NS",
    "TCS.NS",
    "INFY.NS",
    "HDFCBANK.NS",
    "ITC.NS"
]

# ==========================
# Data
# ==========================
fetch_data(tickers)
returns = load_returns()

# ==========================
# Optimization
# ==========================
weights = optimize_portfolio(
    returns,
    risk_aversion=1.0,
    min_weight=0.05,
    max_weight=0.40
)

# ==========================
# Save results
# ==========================
os.makedirs("results", exist_ok=True)

plt.figure(figsize=(8, 5))
plt.bar(tickers, weights)
plt.title("Level 2 Portfolio Weights (Constrained)")
plt.ylabel("Weight")
plt.grid(True)
plt.savefig("results/level2_weights.png", dpi=150)
plt.close()

print("Level 2 Portfolio Weights:")
for t, w in zip(tickers, weights):
    print(f"{t}: {w:.4f}")
