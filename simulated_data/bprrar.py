import numpy as np
import csv
from ast import literal_eval as make_tuple
from class_and_func.estimator_class import multivariate_estimator_bfgs


if __name__ == "__main__":
    np.random.seed(0)
    dim = 10
    number = 3
    first = 1
    C_grid = [15]#, 50]
    stop_criteria = 1e-4
    for C in C_grid:
        with open("estimation_"+str(number)+'_file/_simulation'+str(number), 'r') as read_obj:
            csv_reader = csv.reader(read_obj)
            with open("estimation_"+str(number)+'_file/_estimation'+str(number)+'penC'+str(C), 'w', newline='') as myfile:
                i = 1
                wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
                for row in csv_reader:
                  if i <= first:
                    print("# ", i)
                    tList = [make_tuple(i) for i in row]

                    loglikelihood_estimation_pen = multivariate_estimator_bfgs(dimension=dim, penalty="rlsquares", C=C, eps=stop_criteria, options={"disp": False})
                    res = loglikelihood_estimation_pen.fit(tList)
                    print(loglikelihood_estimation_pen.res.x)
                    wr.writerow(loglikelihood_estimation_pen.res.x.tolist())
                    i += 1
                  else:
                    break
        before = 1
        until = 1
        with open("estimation_"+str(number)+'_file/_simulation'+str(number), 'r') as read_obj:
            csv_reader = csv.reader(read_obj)
            with open("estimation_"+str(number)+'_file/_estimation'+str(number)+'penC'+str(C), 'a', newline='') as myfile:
                i = 1
                wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
                for row in csv_reader:
                    if i <= before:
                        i += 1
                    elif i > until:
                        break
                    else:
                        print("# ", i)
                        tList = [make_tuple(i) for i in row]
                        # print(stop_criteria)
                        loglikelihood_estimation_pen = multivariate_estimator_bfgs(dimension=dim, penalty="rlsquares",
                                                                                   eps=stop_criteria, C=C, options={"disp": False})
                        res = loglikelihood_estimation_pen.fit(tList)
                        # print(loglikelihood_estimation_pen.res.x)
                        wr.writerow(loglikelihood_estimation_pen.res.x.tolist())
                        i += 1