import sys
import os
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sv_model import simulate_sv
from particle_filter import particle_filter_sv
from diagnostics import plot_diagnostics

os.makedirs("results", exist_ok=True)

# ==========================
# Run simulation
# ==========================
returns, sigma_true, _ = simulate_sv()

sigma_est, ess, loglikelihood = particle_filter_sv(returns)

# ==========================
# Plot volatility
# ==========================
plt.figure(figsize=(10, 5))
plt.plot(sigma_true, label="True Volatility", linewidth=2)
plt.plot(sigma_est, label="Filtered Volatility", linestyle="--")
plt.xlabel("Time")
plt.ylabel("Volatility")
plt.title("Particle Filter for Stochastic Volatility")
plt.legend()
plt.grid(True)
plt.savefig("results/volatility_estimation.png", dpi=150)
plt.close()

# ==========================
# Diagnostics
# ==========================
plot_diagnostics(ess, loglikelihood)
