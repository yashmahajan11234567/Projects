import numpy as np
import pandas as pd

TRADING_DAYS = 252

def annualized_return(returns):
    compounded = (1 + returns).prod()
    n_years = len(returns) / TRADING_DAYS
    return compounded**(1 / n_years) - 1


def annualized_volatility(returns):
    return returns.std() * np.sqrt(TRADING_DAYS)


def sharpe_ratio(returns, risk_free_rate=0.0):
    excess = returns - risk_free_rate / TRADING_DAYS
    return excess.mean() / excess.std() * np.sqrt(TRADING_DAYS)


def max_drawdown(equity_curve):
    running_max = equity_curve.cummax()
    drawdown = (equity_curve - running_max) / running_max
    return drawdown.min()


def compute_all_metrics(equity_curve):
    returns = equity_curve.pct_change().dropna()

    return {
        "Annualized Return": annualized_return(returns),
        "Annualized Volatility": annualized_volatility(returns),
        "Sharpe Ratio": sharpe_ratio(returns),
        "Max Drawdown": max_drawdown(equity_curve)
    }
