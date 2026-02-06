import numpy as np
import time
from utils import PARAMS, black_scholes_call

S0, K, r, sigma, T = PARAMS.values()

def mc_price(n=200000):
    Z = np.random.randn(n)
    ST = S0 * np.exp((r - 0.5*sigma**2)*T + sigma*np.sqrt(T)*Z)
    return np.exp(-r*T) * np.mean(np.maximum(ST-K,0))

start = time.time()
mc = mc_price()
print("MC Price:", mc, "Time:", time.time()-start)

print("BS Price:", black_scholes_call(S0, K, T, r, sigma))
