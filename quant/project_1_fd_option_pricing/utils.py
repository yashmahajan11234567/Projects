import numpy as np
from scipy.stats import norm

def call_payoff(S, K):
    return np.maximum(S - K, 0.0)

def put_payoff(S, K):
    return np.maximum(K - S, 0.0)

def black_scholes_call(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

PARAMS = {
    "S0": 100.0,
    "K": 100.0,
    "r": 0.05,
    "sigma": 0.2,
    "T": 1.0
}
