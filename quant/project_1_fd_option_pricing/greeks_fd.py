import numpy as np
from european_option_fd import S, V, dS

Delta = np.gradient(V, dS)
Gamma = np.gradient(Delta, dS)

print("Delta at S0:", Delta[len(S)//2])
print("Gamma at S0:", Gamma[len(S)//2])
