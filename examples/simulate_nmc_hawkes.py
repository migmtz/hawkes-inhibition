from class_and_func.hawkes_process import multi_simple_hawkes
import numpy as np
from matplotlib import pyplot as plt


if __name__ == "__main__":
    baselines = 5 * (10 ** -4) * np.ones(2)
    kernels = [[np.array([[1000], [4.5 * (10 ** -4)]]), np.array([[1000], [9.0 * (10 ** -4)]])],
               [np.array([[0.0], [0.0]]), np.array([[1000], [4.5 * (10 ** -4)]])]]

    for num_first in range(1, 101):
        np.random.seed(num_first)

        hp = multi_simple_hawkes(baselines, kernels, max_jumps=500)
        hp.simulate()
        #hp.plot_intensity()

        hp.add_nmc(4000, write=True, num_first=num_first)
        plt.show()
