import numpy as np

def hedging_statistics(pnl):
    return {
        "Mean PnL": np.mean(pnl),
        "Std PnL": np.std(pnl),
        "VaR (5%)": np.percentile(pnl, 5),
        "CVaR (5%)": np.mean(pnl[pnl <= np.percentile(pnl, 5)])
    }
