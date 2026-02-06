import numpy as np
import matplotlib.pyplot as plt
from pricing.black_scholes import black_scholes_call

def price_with_pf_vol(
    S0,
    K,
    r,
    T,
    sigma_pf,
    steps_per_year=252
):
    """
    Price option using time-varying volatility
    by converting PF volatility into an effective sigma
    """
    horizon = int(T * steps_per_year)
    sigma_eff = np.sqrt(np.mean(sigma_pf[:horizon]**2))
    return black_scholes_call(S0, K, T, r, sigma_eff)

def pricing_comparison_plot(
    S0,
    K,
    r,
    T,
    sigma_pf,
    sigma_const
):
    strikes = np.linspace(0.7 * K, 1.3 * K, 40)

    pf_prices = []
    bs_prices = []

    for K_ in strikes:
        pf_prices.append(
            price_with_pf_vol(S0, K_, r, T, sigma_pf)
        )
        bs_prices.append(
            black_scholes_call(S0, K_, T, r, sigma_const)
        )

    plt.figure(figsize=(9, 5))
    plt.plot(strikes, pf_prices, label="Particle Filter Volatility")
    plt.plot(strikes, bs_prices, "--", label="Constant Volatility (BS)")
    plt.xlabel("Strike")
    plt.ylabel("Option Price")
    plt.title("Option Pricing: PF Volatility vs Constant Vol")
    plt.legend()
    plt.grid(True)
    plt.savefig("results/pricing_comparison.png", dpi=150)
    plt.close()
