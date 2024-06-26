import numpy as np
from scipy import stats
from scipy.optimize import minimize
import csv
from ast import literal_eval as make_tuple
import seaborn as sns
from dictionary_parameters import dictionary as param_dict
from matplotlib import pyplot as plt
from class_and_func.colormaps import get_continuous_cmap
from metrics import relative_squared_loss
from class_and_func.likelihood_functions import *
import time


def obtain_average_estimation(file_name, number, dim, number_estimations):
    n = 0
    if file_name[0:4] == "tick":
        if file_name[5:9] == "beta":
            result = np.zeros((dim + dim * dim * dim,))
        else:
            result = np.zeros((dim + dim * dim,))
    else:
        result = np.zeros((2 * dim + dim * dim,))
    with open("estimation_"+str(number)+'_file/_estimation'+str(number)+file_name, 'r') as read_obj:
        csv_reader = csv.reader(read_obj)
        for row in csv_reader:
            if n < number_estimations:
                result += np.array([float(i) for i in row])
                n += 1
    result /= n
    return result


class multivariate_estimator_bfgs_non_penalized(object):
    """
    Estimator class for Exponential Hawkes process obtained through minimizaton of a loss using the L-BFGS-B algorithm.

    Attributes
    ----------
    res : OptimizeResult
        Result from minimization.
    """

    def __init__(self, loss=multivariate_loglikelihood_simplified, grad=True, dimension=None, initial_guess="random",
                 options=None):
        """
        Parameters
        ----------
        loss : {loglikelihood, likelihood_approximated} or callable.
            Function to minimize. Default is loglikelihood.
        dimension : int
            Dimension of problem to optimize. Default is None.
        initial_guess : str or ndarray.
            Initial guess for estimated vector. Either random initialization, or given vector of dimension (2*dimension + dimension**2,). Default is "random".
        options : dict
            Options to pass to the minimization method. Default is {'disp': False}.

        Attributes
        ----------
        bounds :
        """
        if dimension is None:
            raise ValueError("Dimension is necessary for initialization.")
        self.dim = dimension

        if isinstance(grad, bool) and grad:
            self.loss = multivariate_loglikelihood_with_grad
        else:
            self.loss = loss
        self.grad = grad

        self.bounds = [(1e-12, None) for i in range(self.dim)] + [(None, None) for i in range(self.dim * self.dim)] + [
            (1e-12, None) for i in range(self.dim)]
        if isinstance(initial_guess, str) and initial_guess == "random":
            self.initial_guess = np.concatenate(
                (np.concatenate((np.ones(self.dim), np.ones(self.dim * self.dim))), np.ones(self.dim)))
        if options is None:
            self.options = {'disp': False}
        else:
            self.options = options

    def fit(self, timestamps, initial, threshold=0.01):
        """
        Parameters
        ----------
        timestamps : list of tuple.
            Ordered list containing event times and marks.
        """

        self.options['iprint'] = 0
        #print("loss", self.loss)
        #print("first")

        self.initial_guess = initial

        alpha = np.abs(self.initial_guess[self.dim:-self.dim])
        ordered_alpha = np.sort(alpha)
        norm = np.sum(ordered_alpha)
        aux, i = 0, 0
        while aux <= threshold:
            aux += ordered_alpha[i] / norm
            i += 1
        i -= 1
        thresh = ordered_alpha[i]  # We erase those STRICTLY lower
        self.bounds = [(1e-12, None) for i in range(self.dim)] + [(None, None) if i >= thresh else (0, 1e-16)
                                                                  for i in alpha] + [
                          (1e-12, None) for i in range(self.dim)]
        #print("second")
        self.res = minimize(self.loss, self.initial_guess, method="L-BFGS-B",jac=self.grad,
                            args=(timestamps, self.dim), bounds=self.bounds,
                            options=self.options)

        #print(self.res.x)

        self.mu_estim = np.array(self.res.x[0: self.dim])
        self.alpha_estim = np.array(self.res.x[self.dim: -self.dim]).reshape((self.dim, self.dim))
        self.beta_estim = np.array(self.res.x[-self.dim:])

        return self.mu_estim, self.alpha_estim, self.beta_estim


if __name__ == "__main__":
    np.random.seed(0)
    number = 7
    theta = param_dict[number]
    dim = int(np.sqrt(1 + theta.shape[0]) - 1)

    file_name = "grad"

    until = 25

    label = "grad"
    method = "grad"

    for threshold in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35]:
        print("Threshold", threshold)
        np.random.seed(0)
        computation_time = 0

        with open("estimation_" + str(number) + '_file/_estimation' + str(number) + file_name, 'r') as read_obj:
            with open("estimation_" + str(number) + '_file/_simulation' + str(number), 'r') as sim_obj:
                csv_reader = csv.reader(read_obj)
                csv_sim_obj = csv.reader(sim_obj)
                for i, (row, row2) in enumerate(zip(csv_reader, csv_sim_obj)):
                    if i <= until:
                        result = np.array([float(i) for i in row])
                        tList = [make_tuple(i) for i in row2]

                        loglikelihood_estimation_pen = multivariate_estimator_bfgs_non_penalized(dimension=dim,
                                                                                                 options={"disp": False})
                        start_time = time.time()
                        res = loglikelihood_estimation_pen.fit(tList, initial=result, threshold=threshold)
                        end_time = time.time()

                        computation_time += end_time - start_time
                        print(end_time - start_time)
                        # print(loglikelihood_estimation.res.x)
        computation_time /= until
        with open("revision_jcgs/computation_times.txt", "a") as write_obj:
            wr = csv.writer(write_obj, quoting=csv.QUOTE_ALL)
            wr.writerow(["threshgrad " + str(threshold) + " until " + str(until) + " : " + str(computation_time)])
        print(computation_time)
        print("")



    #
    # for threshold in [0.15, 0.20, 0.25, 0.30, 0.35]:
    #     np.random.seed(0)
    #     #threshold = 0.10
    #
    #     with open("estimation_"+str(number)+'_file/_estimation'+str(number)+file_name, 'r') as read_obj:
    #         csv_reader = csv.reader(read_obj)
    #         for z, row in enumerate(csv_reader):
    #             result = np.array([float(i) for i in row])
    #             print("z", z)
    #             # if z==0:
    #             #     with open("estimation_" + str(number) + '_file/_simulation' + str(number), 'r') as read_obj:
    #             #         csv_reader = csv.reader(read_obj)
    #             #         with open("estimation_" + str(number) + '_file/_estimation' + str(number) + 'threshgrad'+str(threshold*100), 'w', newline='') as myfile:
    #             #             i = 1
    #             #             wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    #             #             for row in csv_reader:
    #             #                 if i <= 1:
    #             #                     print("# ", i)
    #             #                     tList = [make_tuple(i) for i in row]
    #             #
    #             #                     loglikelihood_estimation_pen = multivariate_estimator_bfgs_non_penalized(dimension=dim,
    #             #                                                                                options={"disp": False})
    #             #                     res = loglikelihood_estimation_pen.fit(tList, initial=result, threshold=threshold)
    #             #                     print(loglikelihood_estimation_pen.res.x)
    #             #                     wr.writerow(loglikelihood_estimation_pen.res.x.tolist())
    #             #                     i += 1
    #             #                 else:
    #             #                     break
    #             if z >= 5:
    #                 with open("estimation_"+str(number)+'_file/_simulation'+str(number), 'r') as read_obj:
    #                     csv_reader = csv.reader(read_obj)
    #                     with open("estimation_"+str(number)+'_file/_estimation'+str(number)+'threshgrad'+str(threshold*100), 'a', newline='') as myfile:
    #                         wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    #                         for i, row in enumerate(csv_reader):
    #                             if z == i:
    #                                 print("# ", i+1)
    #                                 tList = [make_tuple(i) for i in row]
    #                                 # print(stop_criteria)
    #                                 loglikelihood_estimation_pen = multivariate_estimator_bfgs_non_penalized(dimension=dim,
    #                                                                                                          options={"disp": False})
    #                                 res = loglikelihood_estimation_pen.fit(tList, initial=result, threshold=threshold)
    #                                 print(loglikelihood_estimation_pen.res.x)
    #                                 wr.writerow(loglikelihood_estimation_pen.res.x.tolist())
    #                                 #i += 1
    #
