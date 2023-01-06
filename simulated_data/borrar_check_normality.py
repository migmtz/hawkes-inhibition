import csv
from dictionary_parameters import dictionary as param_dict
from ast import literal_eval as make_tuple
from scipy.optimize import minimize
from class_and_func.likelihood_functions import *
from matplotlib import pyplot as plt
import scipy.stats
from scipy.stats import kstest


def obtain_confidence_intervals(file_name, number, dim, number_estimations):
    n = 0
    if file_name[0:4] == "tick":
        if file_name[5:9] == "beta":
            average = np.zeros((number_estimations, dim + dim * dim * dim))
        else:
            average = np.zeros((number_estimations, dim + dim * dim))
    else:
        average = np.zeros((number_estimations, 2 * dim + dim * dim))

    if file_name[0:4] == "conf":
        with open("sample_" + str(number_estimations) + "/estimation_" + str(number) + '_file/_estimation' + str(number) + file_name, 'r') as read_obj:
            print(file_name)
            csv_reader = csv.reader(read_obj)
            for row in csv_reader:
                if n < number_estimations:
                    average[n, :] += np.array([float(i) for i in row])
                    n += 1
    else:
        with open("estimation_" + str(number) + '_file/_estimation' + str(number) + file_name, 'r') as read_obj:
            csv_reader = csv.reader(read_obj)
            for row in csv_reader:
                if n < number_estimations:
                    average[n, :] += np.array([float(i) for i in row])
                    n += 1
    if n != number_estimations:
        print("Wrong number of estimations")
    full_matrix = average
    average = np.mean(average, axis=0)

    return average, full_matrix, n


if __name__ == "__main__":
    number = 7
    theta = param_dict[number]
    dim = int(np.sqrt(1 + theta.shape[0]) - 1)
    mu = theta[:dim]
    alpha = theta[dim:-dim].reshape((dim, dim))
    beta = theta[-dim:]
    number_estimations = 25
    annot = False

    avg, full_matrix, n = obtain_confidence_intervals("grad", number, dim, number_estimations)

    print("Number of estimations: ", n)

    mu_avg = avg[:dim]
    alpha_avg = avg[dim:-dim].reshape((dim, dim))
    beta_avg = avg[-dim:]

    aux = full_matrix[:, dim:-dim]
    print(aux.shape)

    fig, ax = plt.subplots(dim, dim)
    for i in range(dim*dim):
        ax[i//dim, i%dim].hist(aux[:, i])
        ax[i // dim, i % dim].plot([0, 0], [0, 25])

    fig2, ax2 = plt.subplots(dim, dim)
    aux_list = []
    for i in range(dim * dim):
        scipy.stats.probplot(aux[:, i], dist="norm", plot=ax2[i // dim, i % dim])
        _, aux_val = kstest((aux[:, i] - np.mean(aux[:, i]))/np.std(aux[:, i]), "norm")
        aux_list += [aux_val]
        print(aux_val)
        # blah = np.random.poisson(1, 100)
        # print(kstest((blah - np.mean(blah)) / np.std(blah), "norm"))

    print(np.min(aux_list))
    fig3, ax3 = plt.subplots()
    ax3.scatter(np.arange(len(aux_list)), np.sort(aux_list))
    plt.show()