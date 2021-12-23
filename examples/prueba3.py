import numpy as np
from class_and_func.multivariate_exponential_process import multivariate_exponential_hawkes
from class_and_func.streamline_tick import four_estimation, four_estimation_with_grid, plot_four
from class_and_func.estimator_class import multivariate_estimator_bfgs

from matplotlib import pyplot as plt

if __name__ == "__main__":
    # With False, the grid of beta for sumExpKern in Tick will contain only the real parameters beta
    # With True, a random grid must be provided
    with_grid = True
    beta_grid = np.array(range(1,8))
    ### Simulation of event times
    np.random.seed(0)

    dim = 2  # 2, 3 ou 4

    mu = np.array([[1.5], [2.5]])
    alpha = np.array([[0.0, 0.6], [-1.2, -1.5]])
    beta = np.array([[1.], [2.]])
    max_jumps = 5000

    ################# SIMULATION
    hawkes = multivariate_exponential_hawkes(mu=mu, alpha=alpha, beta=beta, max_jumps=max_jumps)

    print("Starting simulation...")
    hawkes.simulate()
    print("Finished simulation")

    fig, ax = plt.subplots(dim, dim)

    lim_x = np.max((1 / beta) * (np.log(np.abs(alpha)+1e-10) - np.log(0.01)))

    x = np.linspace(0, lim_x, 100)

    for C in [10**i for i in range(-3, 4)]:
        ################# ESTIMATION LOG
        loglikelihood_estimation_pen = multivariate_estimator_bfgs(dimension=dim, penalty="l2", C=C,
                                                                   options={"disp": False})
        mu_pen, alpha_pen, beta_pen = loglikelihood_estimation_pen.fit(hawkes.timestamps)
        print(mu_pen, "\n", alpha_pen, "\n", beta_pen, "")
        #
        # ################# ESTIMATION TICK
        # print("Tick", C)
        # if with_grid:
        #     params_tick = four_estimation_with_grid(beta, beta_grid, hawkes.timestamps, C=C)
        # else:
        #     params_tick = four_estimation(beta, hawkes.timestamps, penalty="l2")
        # print(params_tick[2][1], "\n" * 3, params_tick[3][1])
        ################# PLOT

        for i in range(dim):
            for j in range(dim):
                ax[i, j].plot(x, alpha_pen[i, j] * np.exp(-beta_pen[i] * x), label="Pen C={}".format(C))

    for i in range(dim):
        for j in range(dim):
            ax[i, j].plot(x, alpha[i, j] * np.exp(-beta[i] * x), c="k", label="Real kernel")
    plt.legend()
    plt.show()