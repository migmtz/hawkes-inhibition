import numpy as np
import csv
from ast import literal_eval as make_tuple
from class_and_func.multivariate_exponential_process import multivariate_exponential_hawkes
from matplotlib import pyplot as plt
import seaborn as sns
from simulated_data.time_change import time_change
from scipy.stats import kstest
import pickle


def obtain_average_estimation(file_name, numbers):

    mu = np.zeros((250, 1))
    alpha = np.zeros((250, 250))
    beta = np.zeros((250, 1))

    number_estimations = np.zeros((250, 250))
    for number in numbers:
        a_file = open("traitements2/train" + str(number) + ".pkl", "rb")
        tList, filtre_dict_orig, orig_dict_filtre = pickle.load(a_file)
        dim = len(filtre_dict_orig)
        a_file.close()

        aux = [[1 if j in orig_dict_filtre.keys() else 0 for j in range(1, 251)] if i in orig_dict_filtre.keys() else [0 for j in range(1, 251)] for i in range(1, 251)]

        number_estimations += np.array(aux)

        # for i in orig_dict_filtre.keys():
        #     number_estimations[i-1] += 1

        with open("estimation/_traitements2_" + str(number) + file_name, 'r') as read_obj:
            csv_reader = csv.reader(read_obj)
            for row in csv_reader:
                result = np.array([float(i) for i in row])

        mu_est = result[:dim]
        if file_name == "tick":
            alpha_est = result[dim:].reshape((dim, dim))
            beta_est = 4.5*np.ones((dim,))
        else:
            alpha_est = result[dim:-dim].reshape((dim, dim))
            alpha_est[np.abs(alpha_est) <= 1e-16] = 0
            beta_est = result[-dim:]

        for i in range(1, dim + 1):
            mu[filtre_dict_orig[i] - 1] += mu_est[i - 1]
            aux = []
            for j in range(250):
                if j + 1 in filtre_dict_orig.values():
                    aux += [alpha_est[i - 1, orig_dict_filtre[j + 1] - 1]]
                else:
                    aux += [0]

            alpha[filtre_dict_orig[i] - 1, :] += np.array(aux)
            beta[filtre_dict_orig[i] - 1] += beta_est[i - 1]

    number_estimations[number_estimations == 0] = 1
    mu /= np.amax(number_estimations, axis=1).reshape((250, 1))
    alpha /= number_estimations
    beta /= np.amax(number_estimations, axis=1).reshape((250, 1))

    # result = np.concatenate((mu.squeeze(), np.concatenate((alpha.ravel(), beta.squeeze()))))

    return mu, alpha, beta


if __name__ == "__main__":
    np.random.seed(1)
    plot_names = ["grad", "threshgrad20.0", "threshgrad40.0", "threshgrad50.0", "threshgrad60.0", "threshgrad75.0", "threshgrad90.0", "threshgrad95.0", "diag"]
    labels = ["MLE", "thresh20", "thresh40", "thresh50", "thresh60", "thresh75", "thresh90", "thresh95", "diag"]

    tot_repetition = 100
    subsampling_size = 3

    a_file = open("traitements2/kept_dimensions.pkl", "rb")
    estimated_mask = np.concatenate((pickle.load(a_file)[0], np.array([True])))
    a_file.close()

    fig, ax = plt.subplots()

    for label, file_name in zip(labels, plot_names):
        p_values = np.zeros((250 + 1,))
        counting = np.zeros((250 + 1,))

        transformation_list = []
        mu, alpha, beta = obtain_average_estimation(file_name, range(1, 11))
        kep = np.sum(alpha[estimated_mask[:-1],:][:, estimated_mask[:-1]] != 0)/(np.sum(estimated_mask) - 1)**2
        print(file_name, " Kept proportion of coeff = ", kep)
        for number in range(1,11):
            a_file = open("traitements2/test" + str(number) + ".pkl", "rb")
            tList, filtre_dict_orig, orig_dict_filtre = pickle.load(a_file)
            dim = len(filtre_dict_orig)
            a_file.close()

            aux = [i - 1 for i in filtre_dict_orig.values()]

            mu_aux = mu[aux, :]
            alpha_aux = alpha[aux, :][:, aux]
            beta_aux = beta[aux, :]

            estimation = np.concatenate((mu_aux.squeeze(), np.concatenate((alpha_aux.ravel(), beta_aux.squeeze()))))

            help = np.array([t for (t, m) in tList if m != 0])
            help = np.argwhere((help[1:] - help[:-1]) == 0.0)

            definitive_list = []
            j = 1
            for i in range(len(tList)):
                if i - 2 in help:
                    definitive_list += [(tList[i][0] + j * (1e-10), tList[i][1])]
                    j += 1
                else:
                    definitive_list += [(tList[i][0], tList[i][1])]
            transformation_list += [time_change(estimation, definitive_list)]

        for repet in range(tot_repetition):
            np.random.seed(repet)
            subsample = np.random.choice(10, subsampling_size, replace=False)
            #print(subsample)
            total_set = set()
            times_list = [[] for i in range(250 + 1)]
            for sub in subsample:
                a_file = open("traitements2/test" + str(sub + 1) + ".pkl", "rb")
                tList, filtre_dict_orig, orig_dict_filtre = pickle.load(a_file)
                dim = len(filtre_dict_orig)
                a_file.close()

                total_set = total_set.union(set(filtre_dict_orig.values()))
                test_transformed, transformed_dimensional = transformation_list[sub]

                for ref, j in enumerate(transformed_dimensional):
                    times_list[filtre_dict_orig[ref + 1] - 1] += j
                times_list[-1] += test_transformed

            counting[np.array(list(total_set)) - 1] += 1
            counting[-1] += 1
            for ref, j in enumerate(times_list[:-1]):
                if len(j) != 0:
                    if 0.0 in j:
                        p_values[ref] += 0
                    else:
                        p_values[ref] += kstest(j, cdf="expon", mode="exact").pvalue
            if 0.0 in times_list[-1]:
                p_values[-1] += 0
            else:
                p_values[-1] += kstest(times_list[-1], cdf="expon", mode="exact").pvalue
        p_values /= np.maximum(counting, 1)
        #print(counting)

        p_values = p_values[estimated_mask]
        p_values = np.round(p_values, 5)

        a = p_values.reshape((1, len(p_values)))
        print(" \\\\\n".join([" & ".join(map(str, line)) for line in a]))

        sc = ax.scatter([i for i in range(len(p_values))], np.sort(p_values), label=label, s=16)
        argsort = np.argsort(p_values)
        ax.scatter([[np.argmax(argsort)]], [p_values[-1]], c=sc.get_edgecolor(), marker="X", s=64)

    ax.plot([i for i in range(len(p_values))], [(i * 0.05) / len(p_values) for i in range(len(p_values))])
    plt.legend()
    plt.title("Patricia")
    plt.show()
