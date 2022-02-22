import numpy as np
from scipy import stats
from scipy.optimize import minimize
import csv
from ast import literal_eval as make_tuple
import seaborn as sns
from matplotlib import pyplot as plt
from class_and_func.colormaps import get_continuous_cmap
from class_and_func.likelihood_functions import *
import pickle
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
    with open("estimation/_traitements1_" + str(number) + file_name, 'r') as read_obj:
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

    def fit(self, timestamps, threshold=0.01):
        """
        Parameters
        ----------
        timestamps : list of tuple.
            Ordered list containing event times and marks.
        """

        self.options['iprint'] = 0
        print("loss", self.loss)
        print("first")
        self.res = minimize(self.loss, self.initial_guess, method="L-BFGS-B", jac=self.grad,
                            args=(timestamps, self.dim), bounds=self.bounds,
                            options=self.options)
        self.initial_guess = self.res.x

        alpha = np.abs(self.res.x[self.dim:-self.dim])
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
        print("second")
        self.res = minimize(self.loss, self.initial_guess, method="L-BFGS-B",jac=self.grad,
                            args=(timestamps, self.dim), bounds=self.bounds,
                            options=self.options)

        print(self.res.x)

        self.mu_estim = np.array(self.res.x[0: self.dim])
        self.alpha_estim = np.array(self.res.x[self.dim: -self.dim]).reshape((self.dim, self.dim))
        self.beta_estim = np.array(self.res.x[-self.dim:])

        return self.mu_estim, self.alpha_estim, self.beta_estim


class multivariate_estimator_bfgs_non_penalized2(object):
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

    def fit(self, timestamps, threshold=0.01, initial=None):
        """
        Parameters
        ----------
        timestamps : list of tuple.
            Ordered list containing event times and marks.
        """

        self.options['iprint'] = 0
        print("loss", self.loss)
        print("first")
        self.res = initial

        alpha = np.abs(self.res[self.dim:-self.dim])
        ordered_alpha = np.sort(alpha)
        norm = np.sum(ordered_alpha)
        aux, i = 0, 0
        while aux <= threshold:
            aux += ordered_alpha[i] / norm
            i += 1
        i -= 1
        thresh = ordered_alpha[i]  # We erase those STRICTLY lower
        print(self.options)
        self.bounds = [(1e-12, None) for i in range(self.dim)] + [(None, None) if i >= thresh else (0, 1e-16)
                                                                  for i in alpha] + [
                          (1e-12, None) for i in range(self.dim)]
        print("second")
        self.initial_guess = np.concatenate(
                (np.concatenate((np.ones(self.dim), np.zeros(self.dim * self.dim))), np.ones(self.dim)))
        self.res = minimize(self.loss, self.initial_guess, method="L-BFGS-B",jac=self.grad,
                            args=(timestamps, self.dim), bounds=self.bounds,
                            options=self.options)

        print(self.res.x)

        self.mu_estim = np.array(self.res.x[0: self.dim])
        self.alpha_estim = np.array(self.res.x[self.dim: -self.dim]).reshape((self.dim, self.dim))
        self.beta_estim = np.array(self.res.x[-self.dim:])

        return self.mu_estim, self.alpha_estim, self.beta_estim


if __name__ == "__main__":
    np.random.seed(0)

    number = 9
    a_file = open("traitements1/neuro_data" + str(number) + ".pkl", "rb")
    tList, filtre_dict_orig, orig_dict_filtre = pickle.load(a_file)
    print(tList[0], tList[-1])
    dim = len(filtre_dict_orig)
    a_file.close()

    threshold = 0.9

    file_name = "grad"
    initial = obtain_average_estimation(file_name, number, dim, 1)

    with open("estimation/_traitements1_" + str(number) + 'threshgrad'+str(round(threshold*100, 1)), 'w', newline='') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)

        loglikelihood_estimation = multivariate_estimator_bfgs_non_penalized2(dimension=dim, options={"disp": False})
        #"ftol":1e-16, "gtol":1e-16,

        print("Starting estimation...")
        start_time = time.time()
        res = loglikelihood_estimation.fit(tList, threshold=threshold, initial=initial)
        end_time = time.time()
        # print(loglikelihood_estimation.res.x)

        wr.writerow(loglikelihood_estimation.res.x.tolist())

    print("Time it took: ", (end_time - start_time) // 60, " minutes.")
