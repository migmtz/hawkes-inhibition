import numpy as np
from class_and_func.multivariate_exponential_process import multivariate_exponential_hawkes
from class_and_func.estimator_class import multivariate_estimator_bfgs, multivariate_estimator_bfgs_grad
from matplotlib import pyplot as plt

if __name__ == "__main__":
    ### Simulation of event times
    np.random.seed(0)

    dim = 2  # 2, 3 ou 4

    mu = np.array([[1.5], [2.5]])
    alpha = np.array([[0.0, 0.0], [-1.2, -1.5]])
    beta = np.array([[1.], [2.]])
    max_jumps = 5000
    C = 10

    ################# SIMULATION
    hawkes = multivariate_exponential_hawkes(mu=mu, alpha=alpha, beta=beta, max_jumps=max_jumps)

    print("Starting simulation...")
    hawkes.simulate()
    print("Finished simulation")

    ################# NON PEN LOG
    loglikelihood_estimation = multivariate_estimator_bfgs(dimension=dim, options={"disp": False})
    # loglikelihood_estimation = multivariate_estimator_bfgs_grad(grad=True, dimension=dim, options={"disp": False})
    mu_est, alpha_est, beta_est = loglikelihood_estimation.fit(hawkes.timestamps)
    print(mu_est, "\n", alpha_est, "\n", beta_est)

    ################# ESTIMATION LOG
    loglikelihood__pen = multivariate_estimator_bfgs(dimension=dim, options={"disp": False}, penalty="rlsquares", C=C, eps=1e-3)
    # loglikelihood__pen = multivariate_estimator_bfgs_grad(dimension=dim, options={"disp": False}, penalty="rlsquares", C=C, eps=1e-3, grad=True)
    mu_pen, alpha_pen, beta_pen = loglikelihood__pen.fit(hawkes.timestamps, limit=15)
    print("\n", mu_pen, "\n", alpha_pen, "\n", beta_pen)

    fig, ax = plt.subplots(dim, dim)

    lim_x = np.max((1 / beta) * (np.log(np.abs(alpha) + 1e-10) - np.log(0.01)))

    x = np.linspace(0, lim_x, 100)

    for i in range(dim):
        for j in range(dim):
            ax[i, j].plot(x, alpha[i, j] * np.exp(-beta[i] * x), c="r", label="Real kernel")
            ax[i, j].plot(x, alpha_est[i, j] * np.exp(-beta_est[i] * x), c="m", label="Estimated kernel")
            ax[i, j].plot(x, alpha_pen[i, j] * np.exp(-beta_pen[i] * x), c="k", label="Penalized kernel")

    plt.legend()
    plt.show()
