import numpy as np
from class_and_func.likelihood_functions import multivariate_loglikelihood_simplified
import pickle
import csv


def obtain_average_estimation(file_name, number, dim, number_estimations):
    n = 0
    if file_name[0:4] == "tick":
        if file_name[5:9] == "beta":
            result = np.zeros((dim + dim * dim * dim,))
        else:
            result = np.zeros((dim + dim * dim,))
    else:
        result = np.zeros((2 * dim + dim * dim,))
    with open("estimation/_traitements2_" + str(number) + file_name, 'r') as read_obj:
        csv_reader = csv.reader(read_obj)
        for row in csv_reader:
            if n < number_estimations:
                print(np.array([float(i) for i in row]).shape)
                result += np.array([float(i) for i in row])
                n += 1
    result /= n

    return result


if __name__ == "__main__":

    for number in range(1, 11):
        a_file = open("traitements2/train" + str(number) + ".pkl", "rb")
        tList, filtre_dict_orig, orig_dict_filtre = pickle.load(a_file)
        dim = len(filtre_dict_orig)
        print(dim)
        a_file.close()

        estimation = obtain_average_estimation("grad", number, dim, 1)
        mu_est = estimation[:dim].reshape((dim,1))
        alpha_est = estimation[dim:-dim].reshape((dim, dim))
        alpha_est[np.abs(alpha_est) <= 1e-16] = 0
        beta_est = estimation[-dim:].reshape((dim,1))

        mine = multivariate_loglikelihood_simplified((mu_est, alpha_est, beta_est), tList)
        print("Estimation's loglikelihood", mine)