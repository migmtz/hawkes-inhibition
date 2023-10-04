import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from class_and_func.multivariate_exponential_process import multivariate_exponential_hawkes
from class_and_func.estimator_class import multivariate_estimator_bfgs_grad
from simulated_data.time_change import time_change
from scipy.stats import kstest


if __name__ == "__main__":
    sns.set_theme()

    mu = np.array([1.0, 0.5])
    alpha = np.array([[0.5, -0.7],
                     [0.25, 0.5]])
    beta = np.array([1.0, 1.0])

    max_time = 500
    dim = int(mu.shape[0])

    repet = 50
    box_full = np.zeros((repet, dim*(2+dim)))
    box_missing = np.zeros((repet, (dim - 1) * (1 + dim)))

    tot_points = np.zeros((repet, 2))

    for i in range(repet):
        np.random.seed(i)
        hawkes = multivariate_exponential_hawkes(mu=mu, alpha=alpha, beta=beta, max_time=max_time)
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

        tot_points[i, :] += np.array([np.sum([1 for t, m in hawkes.timestamps if m == k]) for k in range(1, dim + 1)])

    ## Boxplots

    print("Avg nb of points : ", np.mean(tot_points, axis=0))

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

    l11 = 1 / (1 - alpha[0, 0]/beta[0])
    l22 = 1 / (1 - alpha[1, 1]/beta[1])

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

    l11 = 1 / (1 - alpha_estim[0, 0] / beta_estim[0])
    l22 = 1 / (1 - alpha_estim[1, 1] / beta_estim[1])

    l12 = alpha_estim[0, 1] / beta_estim[0]
    l21 = alpha_estim[1, 0] / beta_estim[1]

    avg1 = (mu_estim[0] + l22 * mu_estim[1] * l12) / ((1 / l11) - l12 * l21 * l22)
    avg2 = (mu_estim[1] + l11 * mu_estim[0] * l21) / (1 / l22 - l12 * l21 * l11)

    print("Full estimated averages: ", round(avg1, 4), round(avg2, 4), avg1 * max_time)
    print("Full estimated average process 1: ", mu_estim[0] / (1 - alpha_estim[0, 0]/beta_estim[0]))

    ## Estimated missing averages

    avg_estim = np.mean(box_missing, axis=0)
    print("Missing estimated average: ", max_time * avg_estim[0]/(1 - avg_estim[1]/avg_estim[2]))

    ## Goodness of fit

    p_values = np.zeros((repet, 4))  # 1 p_tot, 2-3 p_dim, 4 p_missing
    theta_full = np.concatenate((mu_estim, alpha_estim.ravel(), beta_estim))
    print(theta_full)
    theta_missing = avg_estim

    for i in range(repet):
        np.random.seed(i+repet)
        hawkes = multivariate_exponential_hawkes(mu=mu, alpha=alpha, beta=beta, max_time=max_time)
        hawkes.simulate()

        test_times = hawkes.timestamps
        missing_times = [(t, m) for t, m in hawkes.timestamps if m < 2]

        test_transformed, transformed_dimensional = time_change(theta_full, test_times)
        p_values[i, 0] += kstest(test_transformed, cdf="expon", mode="exact").pvalue
        for ref, lis in enumerate(transformed_dimensional):
            p_values[i, ref+1] += kstest(lis, cdf="expon", mode="exact").pvalue

        test_missing, transformed_missing = time_change(theta_missing, missing_times)
        p_values[i, -1] += kstest(test_missing, cdf="expon", mode="exact").pvalue

    fig_gof, ax_gof = plt.subplots()
    ax_gof.boxplot(p_values)

    ## Estimated nb of points

    mu_estim_miss = box_full[:, 0:dim]
    alpha_estim_miss = box_full[:, dim:-dim].reshape((repet, dim, dim))
    beta_estim_miss = box_full[:, -dim:]

    l11 = np.array([1 / (1 - alpha_estim_miss[i, 0, 0] / beta_estim_miss[i, 0]) for i in range(repet)])
    l22 = np.array([1 / (1 - alpha_estim_miss[i, 1, 1] / beta_estim_miss[i, 1]) for i in range(repet)])

    l12 = np.array([alpha_estim_miss[i, 0, 1] / beta_estim_miss[i, 0] for i in range(repet)])
    l21 = np.array([alpha_estim_miss[i, 1, 0] / beta_estim_miss[i, 1] for i in range(repet)])

    avg1 = np.array([(mu_estim_miss[i, 0] + l22[i] * mu_estim_miss[i, 1] * l12[i]) / ((1 / l11[i]) - l12[i] * l21[i] * l22[i]) for i in range(repet)])
    avg2 = np.array([(mu_estim_miss[i, 1] + l11[i] * mu_estim_miss[i, 0] * l21[i]) / (1 / l22[i]- l12[i] * l21[i] * l11[i]) for i in range(repet)])


    fig_pt, ax_pt = plt.subplots()

    ax_pt.boxplot([max_time * avg1, max_time * avg2])

    avg1 = np.array([box_missing[i, 0]/(1 - box_missing[i, 1]/box_missing[i, 2]) for i in range(repet)])

    ax_pt.boxplot(max_time * avg1, positions=[1.2])
    #ax_pt.legend()

    plt.show()
