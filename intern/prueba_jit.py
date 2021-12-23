import numpy as np
from matplotlib import pyplot as plt
from class_and_func.multivariate_exponential_process import multivariate_exponential_hawkes
from numba import njit
from class_and_func.likelihood_functions import multivariate_loglikelihood_simplified, multivariate_loglikelihood_jit
import time


if __name__ == "__main__":
    # Set seed
    np.random.seed(1)

    dim = 2  # 2, 3 ou 4

    jitted_like = njit(multivariate_loglikelihood_jit)

    mu = np.array([[0.5], [1.0]])
    alpha = np.array([[-1.9, 3], [1.2, 1.5]])
    beta = np.array([[5], [8]])

    hawkes = multivariate_exponential_hawkes(mu=mu, alpha=alpha, beta=beta, max_jumps=5000)

    # Create a process with given parameters and maximal number of jumps.

    hawkes.simulate()

    theta = np.concatenate((mu.squeeze(), np.ravel(alpha), beta.squeeze()))

    print(theta)

    # tList0 = [i for (i, j) in hawkes.timestamps]
    # tList1 = [j for (i, j) in hawkes.timestamps]

    start_time = time.time()
    for k in range(100*10):
        jitted_time = jitted_like(theta, hawkes.timestamps)
    end_time = time.time()

    print("Jitted parameters: \n", end_time-start_time)

    start_time = time.time()
    for k in range(100*10):
        jitted_time = multivariate_loglikelihood_simplified((mu, alpha, beta), hawkes.timestamps)
    end_time = time.time()

    print("Original parameters: \n", end_time - start_time)