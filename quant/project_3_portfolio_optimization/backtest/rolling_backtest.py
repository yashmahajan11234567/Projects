import numpy as np
import pandas as pd
from optimization.miqp_optimizer import optimize_portfolio_miqp

def rolling_backtest(
    returns,
    window=252,
    rebalance_freq=21,
    risk_aversion=1.0,
    max_assets=3,
    transaction_cost=0.001  # 0.1% per unit turnover
):
    portfolio_value = 1.0
    values = []
    dates = []

    prev_weights = None

    for t in range(window, len(returns), rebalance_freq):
        train = returns.iloc[t-window:t]
        weights, _ = optimize_portfolio_miqp(
            train,
            risk_aversion=risk_aversion,
            max_assets=max_assets
        )

        # Compute turnover
        if prev_weights is None:
            turnover = np.sum(np.abs(weights))
        else:
            turnover = np.sum(np.abs(weights - prev_weights))

        cost = transaction_cost * turnover
        prev_weights = weights.copy()

        test = returns.iloc[t:t+rebalance_freq]

        for i in range(len(test)):
            gross_return = np.dot(weights, test.iloc[i])
            net_return = gross_return - cost / rebalance_freq
            portfolio_value *= (1 + net_return)

            values.append(portfolio_value)
            dates.append(test.index[i])

    return pd.Series(values, index=dates)
