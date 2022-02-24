import numpy as np
import csv
from ast import literal_eval as make_tuple
from class_and_func.multivariate_exponential_process import multivariate_exponential_hawkes
from matplotlib import pyplot as plt
import seaborn as sns
from simulated_data.time_change import time_change
from scipy.stats import kstest
import pickle


def obtain_average_estimation(file_name, number, dim, number_estimations):
    n = 0
    if file_name[0:4] == "tick":
        if file_name[5:9] == "beta":
            result = np.zeros((dim + dim * dim * dim,))
        else:
            result = np.zeros((dim + dim * dim,))
    else:
        result = np.zeros((2 * dim + dim * dim,))
    with open("estimation/_traitements1_"+str(number)+file_name, 'r') as read_obj:
        csv_reader = csv.reader(read_obj)
        for row in csv_reader:
            if n < number_estimations:
                result += np.array([float(i) for i in row])
                n += 1
    result /= n

    return result


if __name__ == "__main__":
    number = 9

    a_file = open("traitements1/neuro_data" + str(number) + ".pkl", "rb")
    tList, filtre_dict_orig, orig_dict_filtre = pickle.load(a_file)
    dim = len(filtre_dict_orig)
    a_file.close()
    number_simulations = 5

    plot_names = ["grad", "threshgrad90.0"]
    labels = ["MLE", "MLE-thresh75"]
    estimations = [obtain_average_estimation(file_name, number, dim, 1) for file_name in plot_names]

    p_values = np.zeros((len(plot_names), dim+1)) # As in the table

    test_times = tList

    for ref, file_name in enumerate(plot_names):
        if file_name[0:4] == "tick":
            test_transformed, transformed_dimensional = time_change(np.concatenate((estimations[ref],beta.squeeze())), test_times)
        else:
            test_transformed, transformed_dimensional = time_change(estimations[ref], test_times)

        p_values[ref, dim] += kstest(test_transformed, cdf="expon", mode="exact").pvalue
        for ref_dim, i in enumerate(transformed_dimensional):
            p_values[ref, ref_dim] += kstest(i, cdf="expon", mode="exact").pvalue

    p_values = np.round(p_values, 3)

    for ref, file_name in enumerate(plot_names):
        print(labels[ref] + " estimated values p-value: ", p_values[ref])

    a = p_values
    print(" \\\\\n".join([" & ".join(map(str, line)) for line in a]))

    fig,ax = plt.subplots()
    for ref, j in enumerate(p_values):
        ax.scatter([i for i in range(dim+1)], np.sort(j), label=labels[ref])
    ax.plot([i for i in range(dim+1)], [(i*0.05)/(dim+1) for i in range(dim+1)])

    #         if file_name[0:4] == "" or file_name[0:3] == "pen":
    #             ax.scatter([i for i in range(dim)], np.sort(plist), label=file_name)
    #
    #     ax.plot([i for i in range(dim)], [0.05 for i in range(dim)], c="g", linestyle="dashed")
    #     ax.plot([i for i in range(dim)], [0.05/(dim-i) for i in range(dim)], c="b", linestyle="dashed")
    #
    # if dim > 5:
    plt.legend()
    plt.show()
