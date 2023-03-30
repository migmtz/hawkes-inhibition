import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from class_and_func.multivariate_exponential_process import multivariate_exponential_hawkes
from class_and_func.estimator_class import multivariate_estimator_bfgs_grad


if __name__ == "__main__":
    sns.set_theme()

    mu = np.array([1.0, 0.5])
    alpha = np.array([[0.5, 0.25],
                     [0.25, 0.5]])
    beta = np.array([1.0, 1.0])

    max_jumps = 3000
    dim = int(mu.shape[0])

    repet = 50
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
        ax[i, 0].boxplot(box_full[:, i])
        ax[i, 3].boxplot(box_full[:, dim*(1+dim) + i])
        if i < dim - 1:
            ax[i, 0].boxplot(box_missing[:, i], positions=[1.5])
            ax[i, 3].boxplot(box_missing[:, (dim - 1) * (dim) + i], positions=[1.5])
        for j in range(dim):
            ax[i, j+1].boxplot(box_full[:, dim * (1 + i) + j])
            if i < dim - 1 and j < dim - 1:
                ax[i, j + 1].boxplot(box_full[:, (dim-1) * (1 + i) + j], positions=[1.5])

    plt.show()