import numpy as np
import csv
from matplotlib import pyplot as plt
import seaborn as sns
from simulated_data.time_change import time_change
from scipy.stats import kstest
import pickle


if __name__ == "__main__":
    np.random.seed(11)
    sample_size = 3
    number_of_resamples = 20

    count_of_index = np.zeros(10)

    for i in range(number_of_resamples):
        sample = np.random.choice(10, sample_size, replace=False) + 1
        count_of_index[sample - 1] += 1
        final_list = []
        count = 0
        for j in sample:
            a_file = open("traitements2/train" + str(j) + ".pkl", "rb")
            tList, filtre_dict_orig, orig_dict_filtre = pickle.load(a_file)
            a_file.close()

            final_list += [(t + count*6.5, filtre_dict_orig[m]) for t,m in tList[1:-1]]
            count += 1

        aux = np.unique(np.array(final_list)[:, 1])
        o_dict_f = dict(zip(aux, range(1, len(aux)+1)))
        f_dict_o = dict(zip(o_dict_f.values(), o_dict_f.keys()))
        final_list = [(t, o_dict_f[m]) for t, m in final_list]
        final_list = [(0.0, 0)] + final_list + [(6.5*sample_size, 0)]
        file = open('resamples/resample' + str(i + 1), 'wb')
        pickle.dump((final_list, f_dict_o, o_dict_f), file)
        file.close()

    print(count_of_index)
