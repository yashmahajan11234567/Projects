import numpy as np

def leland_volatility(sigma, transaction_cost, dt):
    """
    Leland-adjusted volatility
    """
    return sigma * np.sqrt(
        1 + (2 / np.pi) * (transaction_cost / sigma) * np.sqrt(1 / dt)
    )
