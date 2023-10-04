import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import csv
from class_and_func.multivariate_exponential_process import multivariate_powerlaw_hawkes
from simulated_data.time_change import time_change
from scipy.stats import kstest
from ast import literal_eval as make_tuple
from class_and_func.colormaps import get_continuous_cmap


def obtain_average_estimation(number, dim, number_estimations):

    n = 0
    result = np.zeros((2 * dim + dim * dim,))

    with open("estimation_"+str(number)+'_file/_estimation_grad', 'r') as read_obj:
        csv_reader = csv.reader(read_obj)
        for row in csv_reader:
            if n < number_estimations:
                result += np.array([float(i) for i in row])
                n += 1
    result /= n

    return result


if __name__ == "__main__":

    np.random.seed(0)

    dim = 2  # 2, 3 ou 4

    mu = np.array([1.0, 1.0]).reshape((2,1))
    alpha = np.array([[0.1, 1.5], [1.0, -0.5]])
    beta = np.array([[1.0, 1.1], [1.2, 1.0]])
    gamma = np.array([[4.0, 4.0], [4.0, 4.0]])

    gamma_list = [0.5, 1.0, 1.5, 2.0]

    repet = 25

    estimation_list = [obtain_average_estimation(gamma_c, dim, repet) for gamma_c in gamma_list]

    fig, ax = plt.subplots(3, 5, figsize=(14, 6))
    fig_rel, ax_rel = plt.subplots(2, 5, figsize=(14, 4))

    sns.set_theme()
    hex_list = ['#FF3333', '#FFFFFF', '#33FF49']
    hex_err = ['#33FF49', '#FFFFFF', '#FF3333']
    blah = get_continuous_cmap(hex_list)
    blah2 = get_continuous_cmap(hex_err)

    for i, estimation in enumerate(estimation_list):
        mu_estim, alpha_estim, beta_estim = estimation[:dim], estimation[dim:-dim].reshape((dim, dim)), estimation[-dim:]

        g = sns.heatmap(alpha / (gamma_list[i] * gamma), ax=ax[0, i], cmap=blah, annot=True, vmin=-0.25, vmax=0.75, center=0, xticklabels=range(1, dim + 1),
                    yticklabels=range(1, dim + 1))
        g.set_xticklabels(g.get_xticklabels(), rotation=0)
        g.set_xticklabels([])
        if i > 0:
            g.set_yticklabels([])
        g = sns.heatmap(alpha_estim/beta_estim, ax=ax[1, i], cmap=blah, annot=True, vmin=-0.25, vmax=0.75, center=0, xticklabels=range(1, dim + 1),
                    yticklabels=range(1, dim + 1))

        g.set_xticklabels(g.get_xticklabels(), rotation=0)
        g.set_xticklabels([])
        if i > 0:
            g.set_yticklabels([])
        print(alpha / (gamma_list[i] * gamma), alpha_estim/beta_estim)

        aux = np.abs((alpha / (gamma_list[i] * gamma) - alpha_estim/beta_estim)/(alpha / (gamma_list[i] * gamma)))

        g = sns.heatmap(aux, ax=ax_rel[0, i], cmap=blah2, annot=True, center=0, xticklabels=range(1, dim + 1),
                    yticklabels=range(1, dim + 1))
        ax[0, i].set_title("Scenario $\gamma$ = %1.1f"%(gamma_list[i]*gamma[0,0]))

        g.set_xticklabels(g.get_xticklabels(), rotation=0)
        g.set_xticklabels([])
        if i > 0:
            g.set_yticklabels([])

        aux = np.zeros((dim, dim))
        aux[np.abs(np.sign(alpha_estim) - np.sign(alpha)) == 2] = -2
        aux[(alpha_estim != 0.0) * (alpha == 0.0)] = 1
        aux[(alpha_estim == 0.0) * (alpha != 0.0)] = -1
        g = sns.heatmap(aux, ax=ax[2, i], cmap=get_continuous_cmap(['#000000', '#9B59B6', '#FFFFFF', '#E67E22']),
                        annot=False, linewidths=.5, vmin=-2, vmax=1, xticklabels=range(1, dim + 1),
                        yticklabels=range(1, dim + 1))
        g = sns.heatmap(aux, ax=ax_rel[1, i], cmap=get_continuous_cmap(['#000000', '#9B59B6', '#FFFFFF', '#E67E22']),
                        annot=False, linewidths=.5, vmin=-2, vmax=1, xticklabels=range(1, dim + 1),
                        yticklabels=range(1, dim + 1))


        if i > 0:
            g.set_yticklabels([])

        print(i, mu_estim)

    estimation_extreme = obtain_average_estimation("extreme", dim, repet)
    mu_estim, alpha_estim, beta_estim = estimation_extreme[:dim], estimation_extreme[dim:-dim].reshape((dim, dim)), estimation_extreme[-dim:]

    g = sns.heatmap(alpha / (1.0 * gamma), ax=ax[0, 4], cmap=blah, annot=True, vmin=-0.25, vmax=0.75, center=0, xticklabels=range(1, dim + 1),
                    yticklabels=range(1, dim + 1))
    g.set_xticklabels([])
    g.set_yticklabels([])
    g = sns.heatmap(alpha_estim / beta_estim, ax=ax[1, 4], cmap=blah, annot=True, vmin=-0.25, vmax=0.75, center=0, xticklabels=range(1, dim + 1),
                    yticklabels=range(1, dim + 1))
    g.set_xticklabels([])
    g.set_yticklabels([])
    aux = np.abs((alpha / (gamma_list[i] * gamma) - alpha_estim / beta_estim) / (alpha / (gamma_list[i] * gamma)))

    g = sns.heatmap(aux, ax=ax_rel[0, 4], cmap=blah2, annot=True, center=0, xticklabels=range(1, dim + 1),
                yticklabels=range(1, dim + 1))
    g.set_xticklabels([])
    g.set_yticklabels([])
    ax[0, 4].set_title("Scenario $\\beta$")

    aux = np.zeros((dim, dim))
    aux[np.abs(np.sign(alpha_estim) - np.sign(alpha)) == 2] = -2
    aux[(alpha_estim != 0.0) * (alpha == 0.0)] = 1
    aux[(alpha_estim == 0.0) * (alpha != 0.0)] = -1
    g = sns.heatmap(aux, ax=ax[2, 4], cmap=get_continuous_cmap(['#000000', '#9B59B6', '#FFFFFF', '#E67E22']),
                    annot=False, linewidths=.5, vmin=-2, vmax=1, xticklabels=range(1, dim + 1),
                    yticklabels=range(1, dim + 1))
    g.set_yticklabels([])
    g = sns.heatmap(aux, ax=ax_rel[1, 4], cmap=get_continuous_cmap(['#000000', '#9B59B6', '#FFFFFF', '#E67E22']),
                    annot=False, linewidths=.5, vmin=-2, vmax=1, xticklabels=range(1, dim + 1),
                    yticklabels=range(1, dim + 1))
    g.set_yticklabels([])
    print(5, mu_estim)
    #fig.suptitle("Vrais alpha/gamma contre Estimations alpha/beta")
    #
    # ##################
    # #### p-values ####
    # ##################
    #
    # p_values_orig = np.zeros((6, repet, 1 + dim))
    #
    # # First with original simulation
    # for i, gamma_c in enumerate(gamma_list):
    #     with open("estimation_" + str(gamma_c) + "_file/_simulation", 'r') as read_obj:
    #         csv_reader = csv.reader(read_obj)
    #         for j, row in enumerate(csv_reader):
    #             tList = [make_tuple(k) for k in row]
    #
    #             theta = estimation_list[i].copy()
    #             theta[-dim:] *= gamma_c
    #
    #             test_transformed, transformed_dimensional = time_change(theta, tList)
    #             p_values_orig[i, j, dim] += kstest(test_transformed, cdf="expon", mode="exact").pvalue
    #             for ref, trdim in enumerate(transformed_dimensional):
    #                 p_values_orig[i, j, ref] += kstest(trdim, cdf="expon", mode="exact").pvalue
    #
    # print(np.mean(p_values_orig, axis=1))
    #
    # p_values_new = np.zeros((6, repet, 1 + dim))
    #
    # for i, gamma_c in enumerate(gamma_list):
    #     for j in range(repet):
    #         np.random.seed(1000*j)
    #         gamma_corr = gamma_c * gamma
    #
    #         hawkes = multivariate_powerlaw_hawkes(mu=mu, alpha=alpha, beta=beta, gamma=gamma_corr, max_jumps=200)
    #         hawkes.simulate()
    #         tList = hawkes.timestamps
    #
    #         theta = estimation_list[i].copy()
    #         theta[-dim:] *= gamma_c
    #
    #         test_transformed, transformed_dimensional = time_change(theta, tList)
    #         p_values_new[i, j, dim] += kstest(test_transformed, cdf="expon", mode="exact").pvalue
    #         for ref, trdim in enumerate(transformed_dimensional):
    #             p_values_new[i, j, ref] += kstest(trdim, cdf="expon", mode="exact").pvalue
    #
    # print("")
    # print(np.mean(p_values_new, axis=1))
    fig.savefig('eps_images/heatmap_powerlaw_both.eps', bbox_inches='tight', format="eps")
    #plt.show()
