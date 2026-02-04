import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from utils import call_payoff, black_scholes_call, PARAMS

# ==========================
# Parameters
# ==========================
S0, K, r, sigma, T = PARAMS.values()

S_max = 4 * K
M, N = 200, 200
dS, dt = S_max / M, T / N

# ==========================
# Grid
# ==========================
S = np.linspace(0, S_max, M + 1)
V = call_payoff(S, K)

# ==========================
# Coefficients (STANDARD FORM)
# ==========================
i = np.arange(1, M)
sigma2 = sigma * sigma

a = 0.25 * dt * (sigma2 * i**2 - r * i)
b = -0.5 * dt * (sigma2 * i**2 + r)
c = 0.25 * dt * (sigma2 * i**2 + r * i)

# ==========================
# Crank–Nicolson matrices
# ==========================
A = diags(
    diagonals=[-a[1:], 1 - b, -c[:-1]],
    offsets=[-1, 0, 1],
    shape=(M-1, M-1),
    format="csc"
)

B = diags(
    diagonals=[a[1:], 1 + b, c[:-1]],
    offsets=[-1, 0, 1],
    shape=(M-1, M-1),
    format="csc"
)

# ==========================
# Time stepping
# ==========================
for n in range(N):
    t = n * dt

    # Boundary conditions
    V[0] = 0.0
    V[-1] = S_max - K * np.exp(-r * (T - t))

    rhs = B.dot(V[1:M])

    # Boundary contributions
    rhs[0]  += a[0] * V[0]
    rhs[-1] += c[-1] * V[-1]

    V[1:M] = spsolve(A, rhs)

# ==========================
# Prices
# ==========================
fd_price = np.interp(S0, S, V)
bs_price = black_scholes_call(S0, K, T, r, sigma)

print("European Call Option")
print(f"FD Price : {fd_price:.6f}")
print(f"BS Price : {bs_price:.6f}")
print(f"Error    : {abs(fd_price - bs_price):.6e}")

# ==========================
# Plot
# ==========================
S_plot = S[1:]

plt.plot(S, V, label="Crank-Nicolson FD")
plt.plot(
    S_plot,
    black_scholes_call(S_plot, K, T, r, sigma),
    "--",
    label="Analytical BS"
)
plt.xlabel("Asset Price")
plt.ylabel("Option Value")
plt.legend()
plt.savefig("results/european_option_cn.png")

