import numpy as np

def effective_sample_size(weights):
    return 1.0 / np.sum(weights**2)

def particle_filter_sv(
    returns,
    N=1000,
    tau=0.15,
    sigma0=0.2,
    resample_threshold=0.5
):
    T = len(returns)

    particles = np.random.normal(
        np.log(sigma0**2), 0.5, N
    )
    weights = np.ones(N) / N

    filtered_sigma = np.zeros(T)
    ess_history = np.zeros(T)
    loglikelihood = np.zeros(T)

    for t in range(T):

        # Prediction
        particles += np.random.normal(0, tau, N)

        # Likelihood
        var = np.exp(particles)
        likelihoods = (
            1.0 / np.sqrt(2 * np.pi * var)
            * np.exp(-returns[t]**2 / (2 * var))
        )

        weights *= likelihoods
        weights += 1e-300
        weights /= np.sum(weights)

        # Estimates
        x_hat = np.sum(weights * particles)
        filtered_sigma[t] = np.exp(x_hat / 2)

        # Diagnostics
        ess = effective_sample_size(weights)
        ess_history[t] = ess
        loglikelihood[t] = np.log(np.mean(likelihoods))

        # Resampling
        if ess < resample_threshold * N:
            indices = np.random.choice(N, size=N, p=weights)
            particles = particles[indices]
            weights.fill(1.0 / N)

    return filtered_sigma, ess_history, loglikelihood
