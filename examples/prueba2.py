import numpy as np
from class_and_func.multivariate_exponential_process import multivariate_exponential_hawkes
from class_and_func.streamline_tick import four_estimation, plot_four
from class_and_func.estimator_class import multivariate_estimator_jit
from class_and_func.likelihood_functions import multivariate_loglikelihood_jit
from numba import njit

from matplotlib import pyplot as plt

if __name__ == "__main__":
    ### Simulation of event times
    np.random.seed(0)

    dim = 2  # 2, 3 ou 4

    mu = 10*np.random.uniform(0,5,(dim,1))
    alpha = 10*np.random.normal(0, 1, (dim,dim))
    beta = 10*np.random.uniform(0,5, (dim,1))
    print(mu,"\n", alpha, "\n", beta)
    max_jumps = 5000

    ################# SIMULATION
    hawkes = multivariate_exponential_hawkes(mu=mu, alpha=alpha, beta=beta, max_jumps=max_jumps)

    print("Starting simulation...")
    hawkes.simulate()
    print("Finished simulation")

    ################# ESTIMATION LOG
    jitted_loss = njit(multivariate_loglikelihood_jit)
    loglikelihood_estimation = multivariate_estimator_jit(loss=jitted_loss, dimension=dim, options={"disp": False})
    mu_est, alpha_est, beta_est = loglikelihood_estimation.fit(hawkes.timestamps)
    print(mu_est, "\n", alpha_est,"\n", beta_est)

    ################# ESTIMATION TICK
    print("Tick")
    params_tick = four_estimation(beta, hawkes.timestamps, penalty="l2")

    # for mtick, atick in params_tick:
    #     print(mtick, "\n", atick, "\n\n")

    # for i in range(2):
    #     for j in range(2):
    #         for u in range(2):
    #             print(i,j,u, params_tick[2][1][i][j][u])

    ################# PLOT

    fig, ax = plt.subplots(dim, dim)

    lim_x = np.max((1/beta)*(np.log(np.abs(alpha)) - np.log(0.01)))

    x = np.linspace(0, lim_x, 100)

    for i in range(dim):
        for j in range(dim):
            ax[i, j].plot(x, alpha[i, j] * np.exp(-beta[i] * x), c="r", label="Real kernel")
            ax[i, j].plot(x, alpha_est[i, j] * np.exp(-beta_est[i] * x), c="m", label="Estimated kernel")

    plot_four(params_tick, beta, ax=ax, x=x)

    plt.legend()
    plt.show()