import numpy as np
import matplotlib.pyplot as plt

from models.black_scholes import bs_price_call
from models.leland_adjustment import leland_volatility
from simulation.gbm_simulator import simulate_gbm
from simulation.delta_hedge import delta_hedge
from simulation.no_hedge import no_hedge
from analysis.hedging_metrics import hedging_statistics

# ======================
# Parameters
# ======================
S0 = 100
K = 100
r = 0.05
sigma = 0.2
T = 1.0

steps = 252
paths = 5000
transaction_cost = 0.001
dt = T / steps

# ======================
# Simulate prices
# ======================
prices = simulate_gbm(S0, r, sigma, T, steps, paths)

# ======================
# Strategies
# ======================
pnl_no_hedge = no_hedge(prices, K)

pnl_delta = delta_hedge(
    prices, K, r, sigma, T,
    transaction_cost=transaction_cost
)

sigma_leland = leland_volatility(sigma, transaction_cost, dt)

pnl_leland = delta_hedge(
    prices, K, r, sigma_leland, T,
    transaction_cost=transaction_cost
)

# ======================
# Metrics
# ======================
print("\n📊 HEDGING STRATEGY COMPARISON\n")

for name, pnl in {
    "No Hedge": pnl_no_hedge,
    "Delta Hedge": pnl_delta,
    "Leland Hedge": pnl_leland
}.items():
    stats = hedging_statistics(pnl)
    print(f"{name}")
    for k, v in stats.items():
        print(f"  {k}: {v:.4f}")
    print()

# ======================
# Plot
# ======================
plt.figure(figsize=(10, 6))
plt.hist(pnl_no_hedge, bins=50, alpha=0.5, label="No Hedge", density=True)
plt.hist(pnl_delta, bins=50, alpha=0.5, label="Delta Hedge", density=True)
plt.hist(pnl_leland, bins=50, alpha=0.7, label="Leland Hedge", density=True)
plt.legend()
plt.title("PnL Distribution: Hedging Strategy Comparison")
plt.xlabel("PnL")
plt.ylabel("Density")
plt.savefig("hedging_comparison.png")
plt.close()
