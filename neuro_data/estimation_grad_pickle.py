import numpy as np
import pickle
from class_and_func.estimator_class import multivariate_estimator_bfgs, multivariate_estimator_bfgs_grad
import csv
import time


if __name__ == "__main__":
    numbers = [2,4,5]

    for number in numbers:
        np.random.seed(0)

        a_file = open("traitements1/neuro_data" + str(number) + ".pkl", "rb")
        tList, filtre_dict_orig, orig_dict_filtre = pickle.load(a_file)
        # tList = [(t,m) for t,m in tList]
        dim = len(filtre_dict_orig)
        # print(filtre_dict_orig)
        # print(np.sum([m==212 for i,m in tList]))
        a_file.close()
        with open("estimation/_traitements1_"+str(number) + 'grad', 'w', newline='') as myfile:
            wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)

            loglikelihood_estimation = multivariate_estimator_bfgs_grad(dimension=dim, options={"disp": False})

            print("Starting estimation...")
            start_time = time.time()
            res = loglikelihood_estimation.fit(tList)
            end_time = time.time()
            # print(loglikelihood_estimation.res.x)

            wr.writerow(loglikelihood_estimation.res.x.tolist())

        print("Time it took: ", (end_time - start_time) // 60, " minutes.")
