import cvxpy as cp
import numpy as np

def optimize_portfolio(returns, risk_aversion=1.0):
    mu = returns.mean().values
    Sigma = returns.cov().values
    n = len(mu)

    w = cp.Variable(n)

    objective = cp.Maximize(mu @ w - risk_aversion * cp.quad_form(w, Sigma))
    constraints = [
        cp.sum(w) == 1,
        w >= 0,
        w <= 0.4
    ]

    problem = cp.Problem(objective, constraints)
    problem.solve()

    return w.value
