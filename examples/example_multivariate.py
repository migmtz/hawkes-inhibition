import numpy as np
from matplotlib import pyplot as plt
from code.multivariate_exponential_process import multivariate_exponential_hawkes

if __name__ == "__main__":
    # Set seed
    np.random.seed(1)

    dim = 3 #2 ou 3

    if dim == 2:

        mu = np.array([0.5, 1.0])
        alpha = np.array([[-1.9, 3], [0, 0]])
        beta = np.array([[2, 20], [0, 0]])

    elif dim == 3:

        np.random.seed(1)

        mu = np.array([0.5, 1.0, 1.0])
        alpha = np.array([[-1.9, 3, 0], [0, 0, 0], [1.0, 0, -0.5]])
        beta = np.array([[3, 20, 0], [0, 0, 0], [3, 0, 2]])

    else:
        raise ValueError("Nein")

    hawkes = multivariate_exponential_hawkes(mu=mu, alpha=alpha, beta=beta, max_jumps=15*(dim-1))

    # Create a process with given parameters and maximal number of jumps.

    hawkes.simulate()

    print("here")

    hawkes.plot_intensity(plot_N=True)

    plt.show()
    #
    # s = [0,0]
    #
    # for i in hawkes.timestamps:
    #     s[i[1]-1] += 1
    # print(s)
    #
    # print(hawkes.intensity_jumps)