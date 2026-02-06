import numpy as np

def no_hedge(price_paths, K):
    """
    PnL of holding option without hedging
    """
    payoff = np.maximum(price_paths[:, -1] - K, 0)
    premium = payoff.mean()   # fair price approximation
    return premium - payoff
