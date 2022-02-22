import numpy as np
import csv
from ast import literal_eval as make_tuple
from class_and_func.estimator_class import multivariate_estimator_bfgs_grad

import time


if __name__ == "__main__":
    np.random.seed(0)

    number = 1
    dim = 250

    with open("data/_neuro_data"+str(number), 'r') as read_obj:
        csv_reader = csv.reader(read_obj)
        with open("estimation/_estimation"+str(number)+'grad', 'w', newline='') as myfile:
            wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
            for row in csv_reader:
                tList = [make_tuple(i) for i in row]

                loglikelihood_estimation = multivariate_estimator_bfgs_grad(dimension=dim, options={"disp": False})
                print("Starting estimation...")
                start_time = time.time()
                res = loglikelihood_estimation.fit(tList)
                end_time = time.time()
                # print(loglikelihood_estimation.res.x)
                wr.writerow(loglikelihood_estimation.res.x.tolist())
    print("Time it took: ", (end_time - start_time) // 60, " minutes.")