from class_and_func.hawkes_process import multi_simple_hawkes
import numpy as np
from matplotlib import pyplot as plt


if __name__ == "__main__":
    np.random.seed(1)
    baselines = 5*(10**-4)*np.ones(2)
    kernels = [[np.array([[1000], [4.5*(10**-4)]]), np.array([[1000], [9.0*(10**-4)]])],
               [np.array([[0.0], [0.0]]), np.array([[1000], [4.5*(10**-4)]])]]

    print(baselines, kernels)
    hp = multi_simple_hawkes(baselines, kernels, max_jumps=500)
    hp.simulate()
    print(hp.timestamps_type[0][:3], hp.timestamps_type[1][-5:])
    #hp.plot_intensity()

    hp.add_nmc_check(1000)
    plt.show()