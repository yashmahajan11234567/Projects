import numpy as np
import pandas as pd
from pricing.pf_pricing import pricing_comparison_plot

# ==========================
# Inputs
# ==========================
S0 = 180.0
K = 180.0
r = 0.05
T = 0.5

# Load PF volatility
sigma_pf = pd.read_csv(
    "data/market_returns.csv"
).values.flatten()

sigma_const = np.std(sigma_pf) * np.sqrt(252)

pricing_comparison_plot(
    S0, K, r, T, sigma_pf, sigma_const
)

print("✅ Pricing comparison complete.")
