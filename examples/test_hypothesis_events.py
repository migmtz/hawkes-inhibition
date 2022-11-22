import numpy as np
from matplotlib import pyplot as plt
from class_and_func.multivariate_exponential_process import multivariate_exponential_hawkes
import seaborn as sns

from matplotlib import rcParams

rcParams['font.family'] = 'serif'
rcParams["mathtext.fontset"] = "dejavuserif"

if __name__ == "__main__":
    # Set seed

    dim = 2  # 2, 3 ou 4

    if dim == 2:

        mu = np.array([1.0, 100.0])
        alpha = np.array([[0.0, -0.1], [0.0, 0.0]])
        beta = np.array([[1, 1], [1, 1]])

    elif dim == 3:

        mu = np.array([0.5, 1.0, 1.0])
        alpha = np.array([[-1.9, 3, 0], [0, 0, 0], [1.0, 0, -0.5]])
        beta = np.array([[3, 20, 0], [0, 0, 0], [3, 0, 2]])

    elif dim == 4:

        mu = np.array([0.5, 1.0, 0.7, 0.4])
        alpha = 0.5*np.array([[-1.9, 3, -1.1, 0.5], [0.1, 0.6, 0, -1.3], [1.0, 0, -0.5, 1.7], [0.4, 0.8, 0.5, -1.0]])
        beta = 2.5*np.array([[3, 10, 2.5, 1.2], [1.7, 1.3, 0, 0.9], [3, 0, 1.4, 3.2],[1.2, 1.5, 0.8, 0.8]])

    elif dim == 5:

        mu = np.ones((5,1))
        alpha = np.zeros((5,5))
        beta = np.zeros((5,5))

    else:
        raise ValueError("Nein")

    avg = np.array([0.0])
    for i in range(1000):
        np.random.seed(i)
        hawkes = multivariate_exponential_hawkes(mu=mu, alpha=alpha, beta=beta, max_jumps=10000)
        hawkes.simulate()
        avg += hawkes.count[0]
    #avg /= 100
    print(avg)
    # Create a process with given parameters and maximal number of jumps.

