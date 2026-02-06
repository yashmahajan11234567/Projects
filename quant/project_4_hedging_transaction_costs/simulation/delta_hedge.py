import numpy as np
from models.black_scholes import bs_delta_call

def delta_hedge(
    price_paths,
    K, r, sigma, T,
    transaction_cost=0.001
):
    n_paths, steps = price_paths.shape
    dt = T / (steps - 1)

    pnl = np.zeros(n_paths)

    for i in range(n_paths):
        cash = 0.0
        delta_prev = 0.0

        for t in range(steps - 1):
            S = price_paths[i, t]
            tau = T - t * dt

            delta = bs_delta_call(S, K, tau, r, sigma)
            d_delta = delta - delta_prev

            cost = transaction_cost * abs(d_delta) * S
            cash -= d_delta * S + cost
            cash *= np.exp(r * dt)

            delta_prev = delta

        payoff = max(price_paths[i, -1] - K, 0)
        pnl[i] = cash + delta_prev * price_paths[i, -1] - payoff

    return pnl
