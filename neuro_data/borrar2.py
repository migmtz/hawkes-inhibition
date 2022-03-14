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
        first_set = set(filtre_dict_orig.values())
        a_file.close()
        print(filtre_dict_orig[tList[1][1]])
        print(tList[0:2], tList[-2:])
        list_tot = [(t, filtre_dict_orig[m]) for (t, m) in tList[1:-1]]

        a_file = open("traitements2/test" + str(number) + ".pkl", "rb")
        tList, filtre_dict_orig, orig_dict_filtre = pickle.load(a_file)
        dim = len(filtre_dict_orig)
        second_set = set(filtre_dict_orig.values())
        a_file.close()
        print(filtre_dict_orig[tList[-2][1]])
        print(tList[0:2], tList[-2:])

        total_dims = first_set.union(second_set)

        list_tot += [(t+6.5, filtre_dict_orig[m]) for (t, m) in tList[1:-1]]

        filtre_new_orig = {}
        orig_new_filtre = {}
        definitive_list = []
        i = 1
        print(total_dims)
        for m in total_dims:
            orig_new_filtre[m] = i
            filtre_new_orig[i] = m

            i += 1
        definitive_list = [(t, orig_new_filtre[m]) for (t, m) in list_tot]
        definitive_list = [(0.0, 0)] + definitive_list + [(13.0, 0)]

        print(definitive_list[:2], definitive_list[-2:])
        print(filtre_new_orig[definitive_list[1][1]], filtre_new_orig[definitive_list[-2][1]])

        a_file = open("traitements2/testcomplete" + str(number) + ".pkl", "wb")
        pickle.dump([definitive_list, filtre_new_orig, orig_new_filtre], a_file)
        a_file.close()