import numpy as np

def simulate_sv(T=200, tau=0.15, sigma0=0.2, seed=42):
    """
    Simulate a simple stochastic volatility model.
    """
    np.random.seed(seed)

    x = np.zeros(T)        # log-variance
    sigma = np.zeros(T)
    returns = np.zeros(T)

    x[0] = np.log(sigma0**2)
    sigma[0] = sigma0

    for t in range(1, T):
        x[t] = x[t-1] + np.random.normal(0, tau)
        sigma[t] = np.exp(x[t] / 2)
        returns[t] = sigma[t] * np.random.normal()

    return returns, sigma, x
