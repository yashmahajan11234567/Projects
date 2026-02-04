import os
import numpy as np
import matplotlib.pyplot as plt
from utils import put_payoff, PARAMS

# ==========================
# Setup
# ==========================
os.makedirs("results", exist_ok=True)

S0, K, r, sigma, T = PARAMS.values()

S_max = 4 * K
M, N = 200, 200
dS = S_max / M
dt = T / N

omega = 1.2        # relaxation parameter (1 < omega < 2)
tol = 1e-6
max_iter = 10000

# ==========================
# Grid
# ==========================
S = np.linspace(0, S_max, M + 1)
V = put_payoff(S, K)

# ==========================
# Coefficients (Implicit scheme)
# ==========================
i = np.arange(1, M)
sigma2 = sigma * sigma

a = 0.5 * dt * (sigma2 * i**2 - r * i)
b = 1 + dt * (sigma2 * i**2 + r)
c = 0.5 * dt * (sigma2 * i**2 + r * i)

# ==========================
# Time stepping
# ==========================
exercise_boundary = []

for n in range(N):
    t = n * dt

    # Boundary conditions
    V[0] = K                # deep in-the-money put
    V[-1] = 0.0             # worthless at high S

    payoff = put_payoff(S, K)
    V_old = V.copy()

    rhs = V_old[1:M]

    # ==========================
    # PSOR iteration
    # ==========================
    for _ in range(max_iter):
        V_prev = V.copy()

        for j in range(1, M):
            cont = (
                a[j-1] * V[j-1] +
                c[j-1] * V_prev[j+1]
            )

            V_new = (rhs[j-1] + cont) / b[j-1]

            # Projection (American constraint)
            V[j] = max(
                payoff[j],
                (1 - omega) * V_prev[j] + omega * V_new
            )

        if np.linalg.norm(V - V_prev, np.inf) < tol:
            break

    # ==========================
    # Early-exercise boundary
    # ==========================
    exercised = np.where(V[1:M] <= payoff[1:M] + 1e-8)[0]
    exercise_boundary.append(S[exercised[-1] + 1] if len(exercised) else 0.0)

# ==========================
# Price
# ==========================
price = np.interp(S0, S, V)
print(f"American Put Price: {price:.6f}")

# ==========================
# Plots
# ==========================
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(S, V, label="American Put")
plt.plot(S, payoff, "--", label="Payoff")
plt.xlabel("Asset Price")
plt.ylabel("Option Value")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(np.linspace(0, T, N), exercise_boundary)
plt.xlabel("Time")
plt.ylabel("Exercise Boundary")

plt.tight_layout()
plt.savefig("results/american_put_psor.png")
