import matplotlib.pyplot as plt
import os
import pandas as pd

from loaders.market_data import load_returns
from loaders.benchmark_data import fetch_nifty50, load_nifty_returns
from backtest.rolling_backtest import rolling_backtest
from metrics.performance_metrics import compute_all_metrics

returns = load_returns()

# ==========================
# Backtests
# ==========================
portfolio_no_cost = rolling_backtest(
    returns,
    transaction_cost=0.0
)

portfolio_with_cost = rolling_backtest(
    returns,
    transaction_cost=0.001
)

# ==========================
# Benchmark
# ==========================
fetch_nifty50()
nifty_returns = load_nifty_returns()
nifty_returns = nifty_returns.loc[portfolio_with_cost.index]
nifty_curve = (1 + nifty_returns).cumprod()

# ==========================
# Metrics
# ==========================
metrics_df = pd.DataFrame({
    "No Cost Portfolio": compute_all_metrics(portfolio_no_cost),
    "With Cost Portfolio": compute_all_metrics(portfolio_with_cost),
    "NIFTY 50": compute_all_metrics(nifty_curve)
}).T

print("\n📊 PERFORMANCE WITH TRANSACTION COSTS\n")
print(metrics_df.round(4))

# ==========================
# Plot
# ==========================
os.makedirs("results", exist_ok=True)

plt.figure(figsize=(10, 5))
plt.plot(portfolio_no_cost, label="Portfolio (No Cost)", alpha=0.7)
plt.plot(portfolio_with_cost, label="Portfolio (With Cost)", linewidth=2)
plt.plot(nifty_curve, label="NIFTY 50", linestyle="--")
plt.title("Impact of Transaction Costs on Portfolio Performance")
plt.xlabel("Date")
plt.ylabel("Portfolio Value")
plt.grid(True)
plt.legend()
plt.savefig("results/backtest_with_transaction_costs.png", dpi=150)
plt.close()

print("\n✅ Transaction cost analysis complete.")
