import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from class_and_func.multivariate_exponential_process import multivariate_exponential_hawkes
from class_and_func.estimator_class import multivariate_estimator_bfgs_grad, multivariate_estimator_bfgs_conf
from simulated_data.time_change import time_change
from scipy.stats import kstest, t


if __name__ == "__main__":
    sns.set_theme()

    mu = np.array([1.0, 0.5, 0.8])
    alpha = np.array([[0.5, -0.4, 0.4],
                      [0.0, 0.2, 0.3],
                      [0.0, 0.0, 0.2]])
    beta = np.array([1.0, 1.5, 1.0])

    max_time = 100
    dim = int(mu.shape[0])

    repet = 50
    estim_full_mean = np.zeros((repet, dim * (2 + dim)))
    estim_full_std = np.zeros((repet, dim * (2 + dim)))
    estim_missing_mean = np.zeros((repet, (dim - 1) * (1 + dim)))
    estim_missing_std = np.zeros((repet, (dim - 1) * (1 + dim)))

    tot_points = np.zeros((repet, dim))

    for i in range(repet):
        np.random.seed(i)
        hawkes = multivariate_exponential_hawkes(mu=mu, alpha=alpha, beta=beta, max_time=max_time)
        hawkes.simulate()

        ## Estimation with full information
        estimation_full = multivariate_estimator_bfgs_grad(dimension=dim, options={"disp": False})
        estimation_full.fit(hawkes.timestamps)
        estim_full_mean[i, :] += estimation_full.res.x
        estim_full_std[i, :] += estimation_full.res.x ** 2

        ## Estimation with a missing process
        missing_times = [(t, m) for t, m in hawkes.timestamps if m < dim]
        estimation_missing = multivariate_estimator_bfgs_grad(dimension=dim-1, options={"disp":False})
        estimation_missing.fit(missing_times)
        estim_missing_mean[i, :] += estimation_missing.res.x
        estim_missing_std[i, :] += estimation_missing.res.x ** 2

        tot_points[i, :] += np.array([np.sum([1 for t, m in hawkes.timestamps if m == k]) for k in range(1, dim + 1)])

    print("End first estimation")
    level_conf = 0.95
    quantile = -t.ppf((1 - level_conf) / 2, repet - 1)

    full_mean = np.mean(estim_full_mean, axis=0)
    full_std = np.sqrt((np.sum(estim_full_std, axis=0) - repet * (full_mean ** 2)) / (repet - 1))

    missing_mean = np.mean(estim_missing_mean, axis=0)
    missing_std = np.sqrt((np.sum(estim_missing_std, axis=0) - repet * (missing_mean ** 2)) / (repet - 1))

    alpha_full = full_mean[dim:-dim].reshape((dim, dim))
    alpha_dev_full = (full_std[dim:-dim].reshape((dim, dim))) / (np.sqrt(repet))
    alpha_missing = missing_mean[dim-1:-(dim-1)].reshape((dim-1, dim-1))
    alpha_dev_missing = (missing_std[dim-1:-(dim-1)].reshape((dim-1, dim-1))) / (np.sqrt(repet))

    T_full = np.abs(alpha_full / alpha_dev_full).ravel()
    T_missing = np.abs(alpha_missing / alpha_dev_missing).ravel()

    p_full = 1 - (t.cdf(T_full, repet - 1) - t.cdf(-T_full, repet - 1))
    p_missing = 1 - (t.cdf(T_missing, repet - 1) - t.cdf(-T_missing, repet - 1))

    x_full = np.arange(1, len(p_full) + 1)
    x_missing = np.arange(1, len(p_missing) + 1)
    # fig, ax = plt.subplots()
    # ax.scatter(np.arange(1, len(p_values) + 1), np.sort(p_values))
    # ax.plot(x, x * (1 - level_conf) / len(p_values))

    ord_p_full = np.argsort(p_full)
    reord_p_full = np.argsort(ord_p_full)
    support_full = (p_full[ord_p_full] < x_full * (1 - level_conf) / len(p_full))
    ord_p_missing = np.argsort(p_missing)
    reord_p_missing = np.argsort(ord_p_missing)
    support_missing = (p_full[ord_p_missing] < x_missing * (1 - level_conf) / len(p_missing))

    support_full = support_full[reord_p_full].reshape((dim, dim))
    support_missing = support_missing[reord_p_missing].reshape((dim-1, dim-1))

    box_full = np.zeros((repet, dim * (2 + dim)))
    box_missing = np.zeros((repet, (dim - 1) * (1 + dim)))

    for i in range(repet):
        np.random.seed(i)
        hawkes = multivariate_exponential_hawkes(mu=mu, alpha=alpha, beta=beta, max_time=max_time)
        hawkes.simulate()

        ## Estimation with full information
        estimation_full = multivariate_estimator_bfgs_conf(dimension=dim, options={"disp": False})
        estimation_full.fit(hawkes.timestamps, support_full)
        box_full[i, :] += estimation_full.res.x

        ## Estimation with a missing process
        missing_times = [(t, m) for t, m in hawkes.timestamps if m < dim]
        estimation_missing = multivariate_estimator_bfgs_conf(dimension=dim-1, options={"disp":False})
        estimation_missing.fit(missing_times, support_missing)
        box_missing[i, :] += estimation_missing.res.x

    ## Boxplots

    print("Avg nb of points : ", np.mean(tot_points, axis=0))

    fig, ax = plt.subplots(dim, dim + 2)

    for i in range(dim):
        ax[i, 0].boxplot(box_full[:, i], meanline=True, showmeans=True)
        ax[i, -1].boxplot(box_full[:, dim*(1+dim) + i], meanline=True, showmeans=True)
        if i < dim - 1:
            ax[i, 0].boxplot(box_missing[:, i], positions=[1.5], meanline=True, showmeans=True)
            ax[i, -1].boxplot(box_missing[:, (dim - 1) * (dim) + i], positions=[1.5], meanline=True, showmeans=True)
        for j in range(dim):
            ax[i, j+1].boxplot(box_full[:, dim * (1 + i) + j], meanline=True, showmeans=True)
            if i < dim - 1 and j < dim - 1:
                ax[i, j + 1].boxplot(box_full[:, (dim-1) * (1 + i) + j], positions=[1.5], meanline=True, showmeans=True)

    ## Estimated averages

    avg_estim = np.mean(box_full, axis=0)
    mu_estim = avg_estim[0:dim]
    alpha_estim = avg_estim[dim:-dim].reshape((dim, dim))
    beta_estim = avg_estim[-dim:]

    # print(avg_estim)


    ## Estimated missing averages

    avg_estim = np.mean(box_missing, axis=0)
    # print(avg_estim)
    # print("Missing estimated average: ", max_time * avg_estim[0]/(1 - avg_estim[1]/avg_estim[2]))

    ## Goodness of fit

    p_values = np.zeros((repet, dim + 2))  # 1 p_tot, 2-3 p_dim, 4 p_missing
    theta_full = np.concatenate((mu_estim, alpha_estim.ravel(), beta_estim))
    # print(theta_full)
    theta_missing = avg_estim

    for i in range(repet):
        np.random.seed(i+repet)
        hawkes = multivariate_exponential_hawkes(mu=mu, alpha=alpha, beta=beta, max_time=max_time)
        hawkes.simulate()

        test_times = hawkes.timestamps
        missing_times = [(t, m) for t, m in hawkes.timestamps if m < dim]

        test_transformed, transformed_dimensional = time_change(theta_full, test_times)
        p_values[i, 0] += kstest(test_transformed, cdf="expon", mode="exact").pvalue
        for ref, lis in enumerate(transformed_dimensional):
            p_values[i, ref+1] += kstest(lis, cdf="expon", mode="exact").pvalue

        test_missing, transformed_missing = time_change(theta_missing, missing_times)
        p_values[i, -1] += kstest(test_missing, cdf="expon", mode="exact").pvalue

    fig_gof, ax_gof = plt.subplots()
    ax_gof.boxplot(p_values)

    plt.show()
