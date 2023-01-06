# Imports
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from simple_hawkes_process.hawkes_process import multi_simple_hawkes


if __name__ == "__main__":
    baselines = np.array([1e-4, 1e-4])
    kernels = [[np.array([[1000], [4.5e-4]]), np.array([[1000], [9e-4]])],[np.zeros((2, 1)), np.array([[1000], [4.5e-4]])]]
    hawkes = multi_simple_hawkes(baselines=baselines, kernels=kernels, max_jumps=100)

    hawkes.simulate()

    hawkes.plot_intensity()

    plt.show()