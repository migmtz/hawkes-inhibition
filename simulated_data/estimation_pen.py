import numpy as np
import csv
from ast import literal_eval as make_tuple
from class_and_func.estimator_class import multivariate_estimator_bfgs


if __name__ == "__main__":
    dim = 2
    number = 2
    first = 1
    C = 20
    stop_criteria = 1e-6
    with open('_simulation'+str(number), 'r') as read_obj:
        csv_reader = csv.reader(read_obj)
        with open('_estimation'+str(number)+'pen', 'w', newline='') as myfile:
            i = 1
            wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
            for row in csv_reader:
              if i <= first:
                print("# ", i)
                tList = [make_tuple(i) for i in row]

                loglikelihood_estimation_pen = multivariate_estimator_bfgs(dimension=dim, penalty="rlsquares", C=C, eps=stop_criteria, options={"disp": False})
                res = loglikelihood_estimation_pen.fit(tList, limit=30)
                print(loglikelihood_estimation_pen.res.x)
                wr.writerow(loglikelihood_estimation_pen.res.x.tolist())
                i += 1
              else:
                break
    dim = 2
    before = 1
    until = 1
    C = 1
    with open('_simulation'+str(number), 'r') as read_obj:
        csv_reader = csv.reader(read_obj)
        with open('_estimation'+str(number)+'pen', 'a', newline='') as myfile:
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
                    print(stop_criteria)
                    loglikelihood_estimation_pen = multivariate_estimator_bfgs(dimension=dim, penalty="rlsquares",
                                                                               eps=stop_criteria, C=C, options={"disp": False})
                    res = loglikelihood_estimation_pen.fit(tList, limit=30)
                    # print(loglikelihood_estimation_pen.res.x)
                    wr.writerow(loglikelihood_estimation_pen.res.x.tolist())
                    i += 1