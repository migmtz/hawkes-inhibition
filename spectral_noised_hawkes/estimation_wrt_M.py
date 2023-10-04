from class_and_func.hawkes_process import exp_thinning_hawkes
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from functions_and_classes import *
from scipy.optimize import minimize
import time

if __name__ == "__main__":
    sns.set_theme()

    ##############
    # Simulation #
    ##############

    np.random.seed(2)
    mu = 1
    alpha = 0.45
    beta = 0.5

    noise = 1.0

    max_time = 1000.0
    burn_in = -1000

    bounds = [(1e-12, None)] + [(1e-12, 1 - 1e-12)] + [(1e-12, None)]

    M_grid = [5*(k%2 + 1)*10**(k//2) for k in range(1,9)] # [10, 50, 100, 500, 1000, 5000, 10000, 50000]
    alpha_list = np.linspace(0, 1, len(M_grid))

    hp = exp_thinning_hawkes(mu, alpha, beta, t=burn_in, max_time=max_time)
    hp.simulate()

    ppp = exp_thinning_hawkes(noise, 0, beta, t=burn_in, max_time=max_time)
    ppp.simulate()

    times_hp = [0.0] + [t for t in hp.timestamps if t > 0] + [max_time]
    times_pp = [t for t in ppp.timestamps if t > 0]

    parasited_times = [0.0] + np.sort(times_hp[1:-1] + times_pp).tolist() + [max_time]

    print("      Parameters:")
    print(mu, alpha/beta, beta)

    print("      Number of events:")
    print(len(times_hp) - 2, len(times_pp))
    print(len(parasited_times) - 2)
    print("")

    ########
    # Plot #
    ########

    fig, ax = plt.subplots(3, 1)
    x = np.linspace(0, 2, 1000)
    estimations = np.zeros((len(M_grid), 3))
    loglike = np.zeros(len(M_grid))
    estimation_time = np.zeros(len(M_grid))
    mean_nb_points = np.zeros(len(M_grid))

    #################################################################
    # Estimation with no noise with noised parametrisation and grad #
    #################################################################

    for i, M in enumerate(M_grid):

        start_time = time.time()
        res = minimize(spectral_log_likelihood_grad, (1.0, 0.5, 1.0), method="L-BFGS-B", jac=True,
                       args=(spectral_f_exp_grad, M, times_hp), bounds=bounds,
                       options=None)
        end_time = time.time()

        estimations[i, :] = res.x
        loglike[i] = res.fun
        estimation_time[i] = end_time - start_time
        mean_nb_points[i] = max_time * res.x[0] / (1 - res.x[1])

        f_estim = np.array([spectral_f_exp(x_0, res.x) for x_0 in x])
        ax[0].plot(x, f_estim, label="M = {}".format(M), alpha=0.6)

    #######################################
    # Plot Spectral exp against estimated #
    #######################################

    f_x = np.array([spectral_f_exp(x_0, (mu, alpha / beta, beta)) for x_0 in x])
    ax[0].plot(x, f_x, label="True")

    ax[1].plot(M_grid, estimation_time)

    ax[0].legend()
    plt.show()
