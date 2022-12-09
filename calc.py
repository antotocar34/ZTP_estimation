import numpy as np
from scipy.optimize import root_scalar

def sample_ztp(number_of_samples, lam):
    assert lam > 0
    result = np.zeros(number_of_samples)
    i = 0
    while i <= result.size - 1:
        poisson_sample = np.random.poisson(lam=lam, size=1)[0]
        if poisson_sample > 0:
            result[i] = poisson_sample
            i += 1
        else:
            continue
    return result

# Root function corresponding to expression
# (lam / (1 - exp(-lam)) = x_bar
def mm_estimator_root(x_bar: int):
    return lambda lam: lam - x_bar * (1 - np.exp(-1*lam))

# Takes gradient of root function
def grad_mm_estimator_root(x_bar: int):
    return lambda lam: 1 - x_bar * (np.exp(-lam))

def estimate(mean: int):
    f = mm_estimator_root(mean)
    f_prim = grad_mm_estimator_root(mean)
    estimate = root_scalar(f, 
                       # bracket=[0,100]
                       x0=100, # Setting this too low causes problems
                       method='newton',
                       fprime=grad_mm_estimator_root(mean),
                       ).root
    assert estimate > 0, mean
    return estimate


## Test
lam = 0.1 # Lambda we are trying to estimate
mean = np.mean(sample_ztp(10**5, lam)) # sample bunch from Zero Truncated Poisson
estimated_lam = estimate(mean)
error = np.abs(lam - estimated_lam)
print(error)
# assert np.abs(lam - estimated_lam) < 0.1

# Caching the outputs
# All possible means from 1 to 200 
np.array([ estimate(mean) for mean in  np.arange(1, 200, 0.1) ])




