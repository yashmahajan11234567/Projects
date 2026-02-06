import numpy as np

def simulate_gbm(S0, r, sigma, T, steps, n_paths):
    dt = T / steps
    paths = np.zeros((n_paths, steps + 1))
    paths[:, 0] = S0

    for t in range(1, steps + 1):
        Z = np.random.normal(size=n_paths)
        paths[:, t] = paths[:, t-1] * np.exp(
            (r - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*Z
        )
    return paths
