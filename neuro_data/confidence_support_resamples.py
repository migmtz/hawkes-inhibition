import numpy as np
import csv
import seaborn as sns
from matplotlib import pyplot as plt
from class_and_func.colormaps import get_continuous_cmap
from scipy.stats import t
from scipy.optimize import minimize
from class_and_func.likelihood_functions import *
import pickle
from matplotlib.colors import LinearSegmentedColormap
from class_and_func.colormaps import get_continuous_cmap


def obtain_average_estimation(file_name, number, dim, number_estimations):
    n = 0
    if file_name[0:4] == "tick":
        if file_name[5:9] == "beta":
            result = np.zeros((dim + dim * dim * dim,))
        else:
            result = np.zeros((dim + dim * dim,))
    else:
        result = np.zeros((2 * dim + dim * dim,))
    with open("estimation_resamples/_resamples_" + str(number) + file_name, 'r') as read_obj:
        csv_reader = csv.reader(read_obj)
        for row in csv_reader:
            if n < number_estimations:
                result += np.array([float(i) for i in row])
                n += 1
    result /= n

    return result


def obtain_confidence_intervals(file_name, number, dim, number_estimations):
    n = 0
    if file_name[0:4] == "tick":
        if file_name[5:9] == "beta":
            average = np.zeros((dim + dim * dim * dim,))
            st_dev = np.zeros((dim + dim * dim * dim,))
        else:
            average = np.zeros((dim + dim * dim,))
            st_dev = np.zeros((dim + dim * dim,))
    else:
        average = np.zeros((2 * dim + dim * dim,))
        st_dev = np.zeros((2 * dim + dim * dim,))
    with open("estimation_resamples/_resamples_" + str(number) + file_name, 'r') as read_obj:
        csv_reader = csv.reader(read_obj)
        for row in csv_reader:
            if n < number_estimations:
                average += np.array([float(i) for i in row])
                st_dev += np.array([float(i) for i in row])**2
                n += 1
    average /= n
    st_dev = np.sqrt((st_dev - n*(average**2))/(n-1))

    return average, st_dev, n


if __name__ == "__main__":
    mu = np.zeros((250, 1))
    alpha = np.zeros((250, 250))
    beta = np.zeros((250, 1))

    mu_dev = np.zeros((250, 1))
    alpha_dev = np.zeros((250, 250))
    beta_dev = np.zeros((250, 1))

    min_alpha, max_alpha = np.zeros((250, 250)), np.zeros((250, 250))

    number_estimations = np.zeros((250, 250))
    for number in range(1, 21):
        a_file = open("resamples/resample" + str(number) + "", "rb")
        tList, filtre_dict_orig, orig_dict_filtre = pickle.load(a_file)
        dim = len(filtre_dict_orig)
        a_file.close()

        aux = [[1 if j in orig_dict_filtre.keys() else 0 for j in range(1, 251)] if i in orig_dict_filtre.keys() else [0 for j in range(1, 251)] for i in range(1, 251)]

        number_estimations += np.array(aux)

        # for i in orig_dict_filtre.keys():
        #     number_estimations[i-1] += 1

        estimation = obtain_average_estimation("grad", number, dim, 1)
        mu_est = estimation[:dim]
        alpha_est = estimation[dim:-dim].reshape((dim, dim))
        alpha_est[np.abs(alpha_est) <= 1e-16] = 0
        beta_est = estimation[-dim:]

        # print(filtre_dict_orig)

        for i in range(1, dim + 1):
            mu[int(filtre_dict_orig[i]) - 1] += mu_est[i - 1]
            mu_dev[int(filtre_dict_orig[i]) - 1] += mu_est[i - 1]**2
            aux = []
            for j in range(250):
                if j + 1 in filtre_dict_orig.values():
                    aux += [alpha_est[i - 1, orig_dict_filtre[j + 1] - 1]]
                else:
                    aux += [0]

            alpha[int(filtre_dict_orig[i]) - 1, :] += np.array(aux)
            alpha_dev[int(filtre_dict_orig[i]) - 1, :] += np.array(aux)**2
            beta[int(filtre_dict_orig[i]) - 1] += beta_est[i - 1]
            beta_dev[int(filtre_dict_orig[i]) - 1] += beta_est[i - 1]**2

            if number == 1:
                min_alpha[int(filtre_dict_orig[i]) - 1, :] = np.array(aux)
                max_alpha[int(filtre_dict_orig[i]) - 1, :] = np.array(aux)
            else:
                min_alpha[int(filtre_dict_orig[i]) - 1, :] = np.minimum(aux, min_alpha[int(filtre_dict_orig[i]) - 1, :])
                max_alpha[int(filtre_dict_orig[i]) - 1, :] = np.maximum(aux, max_alpha[int(filtre_dict_orig[i]) - 1, :])

    number_estimations[number_estimations == 0] = 1
    mu /= np.amax(number_estimations, axis=1).reshape((250, 1))
    alpha /= number_estimations
    beta /= np.amax(number_estimations, axis=1).reshape((250, 1))

    mu_dev = np.sqrt((mu_dev - number_estimations*(mu**2))/(np.maximum(number_estimations-1, 0)))
    alpha_dev = np.sqrt((alpha_dev - number_estimations * (alpha ** 2)) / (np.maximum(number_estimations - 1, 0)))
    beta_dev = np.sqrt((beta_dev - number_estimations * (beta ** 2)) / (np.maximum(number_estimations - 1, 0)))

    a_file = open("traitements2/kept_dimensions.pkl", "rb")
    estimated_mask = pickle.load(a_file)
    print(np.sum(estimated_mask))
    a_file.close()

    mu = mu[estimated_mask[0]]
    alpha = alpha[estimated_mask[0], :][:, estimated_mask[0]]
    beta = beta[estimated_mask[0]]

    mu_dev = mu_dev[estimated_mask[0], :]
    alpha_dev = alpha_dev[estimated_mask[0], :][:, estimated_mask[0]]
    beta_dev = beta_dev[estimated_mask[0], :]

    min_alpha = min_alpha[estimated_mask[0], :][:, estimated_mask[0]]
    max_alpha = max_alpha[estimated_mask[0], :][:, estimated_mask[0]]

    number_estimations = number_estimations[estimated_mask[0], :][:, estimated_mask[0]]

    level_conf = 0.95
    print("Number of estimations: ", number_estimations)
    quantile = -t.ppf((1 - level_conf) / 2, np.maximum(number_estimations - 1, 0))
    print(quantile)

    alpha_dev = (alpha_dev) / np.sqrt(np.maximum(number_estimations, 0))

    #support = np.invert((alpha - alpha_dev < 0) & (0 < alpha + alpha_dev))
    support = np.invert((min_alpha < 0) & (0 < max_alpha))

    hex_list = ['#FF3333', '#FFFFFF', '#33FF49']
    blah = get_continuous_cmap(hex_list)

    fig, ax = plt.subplots(1, 2)
    sns.heatmap(support, ax=ax[0])
    print(np.sum(alpha>0), np.sum(alpha<0))
    sns.heatmap((support*alpha), ax=ax[1], cmap=blah, center=0)

    print("Supportminmax", np.sum(support)/(alpha.shape[0]**2))

    support = np.invert((alpha - quantile*alpha_dev < 0) & (0 < alpha + quantile*alpha_dev))

    hex_list = ['#FF3333', '#FFFFFF', '#33FF49']
    blah = get_continuous_cmap(hex_list)

    fig2, ax2 = plt.subplots(1, 2)
    sns.heatmap(support, ax=ax2[0])
    sns.heatmap((support * alpha), ax=ax2[1], cmap=blah, center=0)

    print("Supportintervals", np.sum(support) / (alpha.shape[0] ** 2))

    T_statistic = np.abs(alpha / alpha_dev).ravel()
    n_est = np.maximum(number_estimations, 0).ravel()

    p_values = np.array([1 - (t.cdf(T, n - 1) - t.cdf(-T, n - 1)) for T, n in zip(T_statistic, n_est)])
    x = np.arange(1, len(p_values) + 1)

    ord_p_values = np.argsort(p_values)
    reord_p_values = np.argsort(ord_p_values)
    support = (p_values[ord_p_values] < x * (1 - level_conf) / len(p_values))

    dim = mu.shape[0]
    support = support[reord_p_values].reshape((dim, dim))

    fig3, ax3 = plt.subplots(1, 2)
    sns.heatmap(support, ax=ax3[0])
    sns.heatmap((support * alpha), ax=ax3[1], cmap=blah, center=0)

    print("SupportMTest", np.sum(support) / (alpha.shape[0] ** 2))

    plt.show()

