import numpy as np
import csv
from ast import literal_eval as make_tuple
from class_and_func.estimator_class import multivariate_estimator_bfgs
from class_and_func.likelihood_functions import multivariate_loglikelihood_approximated
from dictionary_parameters import dictionary as param_dict


if __name__ == "__main__":
    np.random.seed(0)

    number = 7
    print("Estimation number ", str(number))
    theta = param_dict[number]
    print(theta)
    dim = int(np.sqrt(1 + theta.shape[0]) - 1)

    first = False
    before = 5
    until = 25

    if first:
        with open("estimation_"+str(number)+'_file/_simulation'+str(number), 'r') as read_obj:
            csv_reader = csv.reader(read_obj)
            with open("estimation_"+str(number)+'_file/_estimation'+str(number)+'approx', 'w', newline='') as myfile:
                i = 1
                wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
                for row in csv_reader:
                  while i <= 1:
                    print("# ", i)
                    tList = [make_tuple(i) for i in row]

                    loglikelihood_estimation = multivariate_estimator_bfgs(loss=multivariate_loglikelihood_approximated, dimension=dim, options={"disp": False})
                    res = loglikelihood_estimation.fit(tList)
                    # print(loglikelihood_estimation.res.x)
                    wr.writerow(loglikelihood_estimation.res.x.tolist())
                    i += 1

    with open("estimation_"+str(number)+'_file/_simulation'+str(number), 'r') as read_obj:
        csv_reader = csv.reader(read_obj)
        with open("estimation_"+str(number)+'_file/_estimation'+str(number)+'approx', 'a', newline='') as myfile:
            i = 1
            wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
            for row in csv_reader:
                if i <= before:
                    i += 1
                elif i > before and i <= until:
                    print("# ", i)
                    tList = [make_tuple(i) for i in row]

                    loglikelihood_estimation = multivariate_estimator_bfgs(loss=multivariate_loglikelihood_approximated, dimension=dim, options={"disp": False})
                    res = loglikelihood_estimation.fit(tList)
                    # print(loglikelihood_estimation.res.x)
                    wr.writerow(loglikelihood_estimation.res.x.tolist())
                    i += 1