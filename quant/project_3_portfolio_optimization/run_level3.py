import matplotlib.pyplot as plt
import os

from loaders.market_data import load_returns
from optimization.miqp_optimizer import optimize_portfolio_miqp

# ==========================
# Assets (same as before)
# ==========================
tickers = [
    "RELIANCE.NS",
    "TCS.NS",
    "INFY.NS",
    "HDFCBANK.NS",
    "ITC.NS"
]

# ==========================
# Load data
# ==========================
returns = load_returns()

# ==========================
# Run MIQP Optimization
# ==========================
weights, selected = optimize_portfolio_miqp(
    returns,
    risk_aversion=1.0,
    max_assets=3
)

# ==========================
# Save results
# ==========================
os.makedirs("results", exist_ok=True)

plt.figure(figsize=(8, 5))
plt.bar(tickers, weights)
plt.title("Level 3 Portfolio (Cardinality Constrained)")
plt.ylabel("Weight")
plt.grid(True)
plt.savefig("results/level3_weights.png", dpi=150)
plt.close()

print("Level 3 Portfolio Weights:")
for t, w, z in zip(tickers, weights, selected):
    status = "SELECTED" if z > 0.5 else "NOT SELECTED"
    print(f"{t}: {w:.4f}  |  {status}")
