import cvxpy as cp
import numpy as np

def optimize_portfolio_miqp(returns, risk_aversion=1.0, max_assets=3):
    mu = returns.mean().values
    Sigma = returns.cov().values
    n = len(mu)

    w = cp.Variable(n)
    z = cp.Variable(n, boolean=True)

    objective = cp.Maximize(mu @ w - risk_aversion * cp.quad_form(w, Sigma))

    constraints = [
        cp.sum(w) == 1,
        w >= 0,
        w <= z,
        cp.sum(z) <= max_assets
    ]

    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.ECOS_BB)

    return w.value, z.value
