import matplotlib.pyplot as plt
import os

def plot_diagnostics(ess, loglikelihood):
    os.makedirs("results", exist_ok=True)

    # ESS plot
    plt.figure(figsize=(10, 4))
    plt.plot(ess)
    plt.axhline(0.5 * max(ess), linestyle="--", color="red")
    plt.title("Effective Sample Size (ESS)")
    plt.xlabel("Time")
    plt.ylabel("ESS")
    plt.grid(True)
    plt.savefig("results/ess_plot.png", dpi=150)
    plt.close()

    # Log-likelihood plot
    plt.figure(figsize=(10, 4))
    plt.plot(loglikelihood)
    plt.title("Log-Likelihood Over Time")
    plt.xlabel("Time")
    plt.ylabel("Log-Likelihood")
    plt.grid(True)
    plt.savefig("results/loglikelihood.png", dpi=150)
    plt.close()
