import numpy as np
from matplotlib import pyplot as plt
from class_and_func.multivariate_exponential_process import multivariate_powerlaw_hawkes
import seaborn as sns


from matplotlib import rcParams

rcParams['font.family'] = 'serif'
rcParams["mathtext.fontset"] = "dejavuserif"

if __name__ == "__main__":
    # Set seed
    np.random.seed(0)

    dim = 2  # 2, 3 ou 4

    # mu = np.array([1.0, 1.1]).reshape((2,1))
    # alpha = np.array([[0.1, 1.5], [1.0, -0.5]])
    # beta = np.array([[1.0, 2.0], [0.1, 1.0]])
    # gamma = np.array([[4.0, 4.0], [4.0, 4.0]])

    mu = np.array([1.0, 1.0]).reshape((2,1))
    alpha = np.array([[0.1, 1.5], [1.0, -0.5]])
    beta = np.array([[1.0, 1.1], [1.2, 1.0]])
    gamma = np.array([[4.0, 4.0], [4.0, 4.0]])

    gamma_list = [0.5, 0.7, 1.0, 1.5, 2.0]

    sns.set_theme()
    fig, ax = plt.subplots(2, len(gamma_list), sharey=True, sharex=True)

    for n, gamma_c in enumerate(gamma_list):

        gamma_corr = gamma_c * gamma

        hawkes = multivariate_powerlaw_hawkes(mu=mu, alpha=alpha, beta=beta, gamma=gamma_corr, max_jumps=10)

        # Create a process with given parameters and maximal number of jumps.

        hawkes.simulate()

        #print(hawkes.timestamps[0], hawkes.timestamps[-1])
        print(hawkes.timestamps)
        hawkes.plot_intensity(ax=ax[:, n].T, plot_N=False)

    ax[0, 0].set_ylabel("$\lambda^1$")
    ax[1, 0].set_ylabel("$\lambda^2$")

    ax[1, 0].set_xlabel("$t$")

    plt.show()
