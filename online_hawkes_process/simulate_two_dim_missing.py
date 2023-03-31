import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from class_and_func.multivariate_exponential_process import multivariate_exponential_hawkes
from class_and_func.estimator_class import multivariate_estimator_bfgs_grad


if __name__ == "__main__":
    sns.set_theme()

    mu = np.array([1.0, 0.5])
    alpha = np.array([[0.5, -0.25],
                     [0.25, 0.5]])
    beta = np.array([1.0, 1.0])

    max_jumps = 1500
    dim = int(mu.shape[0])

    repet = 500
    box_full = np.zeros((repet, dim*(2+dim)))
    box_missing = np.zeros((repet, (dim - 1) * (1 + dim)))

    for i in range(repet):
        np.random.seed(i)
        hawkes = multivariate_exponential_hawkes(mu=mu, alpha=alpha, beta=beta, max_jumps=max_jumps)
        hawkes.simulate()

        ## Estimation with full information
        estimation_full = multivariate_estimator_bfgs_grad(dimension=dim, options={"disp": False})
        estimation_full.fit(hawkes.timestamps)
        box_full[i, :] += estimation_full.res.x

        ## Estimation with a missing process
        missing_times = [(t, m) for t, m in hawkes.timestamps if m < 2]
        estimation_missing = multivariate_estimator_bfgs_grad(dimension=dim-1, options={"disp":False})
        estimation_missing.fit(missing_times)
        box_missing[i, :] += estimation_missing.res.x

    ## Boxplots

    fig, ax = plt.subplots(dim, dim + 2)

    for i in range(dim):
        ax[i, 0].boxplot(box_full[:, i], meanline=True, showmeans=True)
        ax[i, 3].boxplot(box_full[:, dim*(1+dim) + i], meanline=True, showmeans=True)
        if i < dim - 1:
            ax[i, 0].boxplot(box_missing[:, i], positions=[1.5], meanline=True, showmeans=True)
            ax[i, 3].boxplot(box_missing[:, (dim - 1) * (dim) + i], positions=[1.5], meanline=True, showmeans=True)
        for j in range(dim):
            ax[i, j+1].boxplot(box_full[:, dim * (1 + i) + j], meanline=True, showmeans=True)
            if i < dim - 1 and j < dim - 1:
                ax[i, j + 1].boxplot(box_full[:, (dim-1) * (1 + i) + j], positions=[1.5], meanline=True, showmeans=True)

    ## Check averages

    l11 = 1 - alpha[0, 0]/beta[0]
    l22 = 1 - alpha[1, 1]/beta[1]

    l12 = alpha[0, 1]/beta[0]
    l21 = alpha[1, 0] / beta[1]

    avg1 = (mu[0] + l22 * mu[1] * l12)/(1/l11 - l12 * l21 * l22)
    avg2 = (mu[1] + l11 * mu[0] * l21) / (1 / l22 - l12 * l21 * l11)

    print("Real averages: ", round(avg1, 4), round(avg2, 4))

    ## Estimated averages

    avg_estim = np.mean(box_full, axis=0)
    mu_estim = avg_estim[0:dim]
    alpha_estim = avg_estim[dim:-dim].reshape((dim, dim))
    beta_estim = avg_estim[-dim:]

    l11 = 1 - alpha_estim[0, 0] / beta_estim[0]
    l22 = 1 - alpha_estim[1, 1] / beta_estim[1]

    l12 = alpha_estim[0, 1] / beta_estim[0]
    l21 = alpha_estim[1, 0] / beta_estim[1]

    avg1 = (mu_estim[0] + l22 * mu_estim[1] * l12) / (1 / l11 - l12 * l21 * l22)
    avg2 = (mu_estim[1] + l11 * mu_estim[0] * l21) / (1 / l22 - l12 * l21 * l11)

    print("Full estimated averages: ", round(avg1, 4), round(avg2, 4))
    print("Full estimated average process 1: ", mu_estim[0] / (1 - alpha_estim[0, 0]/beta_estim[0]))

    ## Estimated missing averages

    avg_estim = np.mean(box_missing, axis=0)
    print(avg_estim, avg_estim[0], (1 - avg_estim[1]/avg_estim[2]))
    print("Missing estimated average: ", avg_estim[0]/(1 - avg_estim[1]/avg_estim[2]))

    plt.show()