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
from scipy.stats import t
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
    with open("estimation/_traitements2_" + str(number) + file_name, 'r') as read_obj:
        csv_reader = csv.reader(read_obj)
        for row in csv_reader:
            if n < number_estimations:
                result += np.array([float(i) for i in row])
                n += 1
    result /= n

    return result


def obtain_confidence_intervals(file_name, number, dim, number_estimations):
    n = 0
    if file_name[0:4] == "tick":
        if file_name[5:9] == "beta":
            average = np.zeros((dim + dim * dim * dim,))
            st_dev = np.zeros((dim + dim * dim * dim,))
        else:
            average = np.zeros((dim + dim * dim,))
            st_dev = np.zeros((dim + dim * dim,))
    else:
        average = np.zeros((2 * dim + dim * dim,))
        st_dev = np.zeros((2 * dim + dim * dim,))
    with open("estimation_" + str(number) + '_file/_estimation' + str(number) + file_name, 'r') as read_obj:
        csv_reader = csv.reader(read_obj)
        for row in csv_reader:
            if n < number_estimations:
                average += np.array([float(i) for i in row])
                st_dev += np.array([float(i) for i in row])**2
                n += 1
    average /= n
    st_dev = np.sqrt((st_dev - n*(average**2))/(n-1))

    return average, st_dev, n


class multivariate_estimator_bfgs_conf(object):
    """
    Estimator class for Exponential Hawkes process obtained through minimizaton of a loss using the L-BFGS-B algorithm choosing the support through confidence level.

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

        if isinstance(initial_guess, str) and initial_guess == "random":
            self.initial_guess = np.concatenate(
                (np.concatenate((np.ones(self.dim), np.ones(self.dim * self.dim))), np.ones(self.dim)))
        if options is None:
            self.options = {'disp': False}
        else:
            self.options = options

    def fit(self, timestamps, initial, support):
        """
        Parameters
        ----------
        timestamps : list of tuple.
            Ordered list containing event times and marks.
        """

        self.options['iprint'] = 0
        #print("loss", self.loss)

        self.initial_guess = initial

        support_flat = support.ravel()

        self.bounds = [(1e-12, None) for i in range(self.dim)] + [(None, None) if i else (0, 1e-16)
                                                                  for i in support_flat] + [
                          (1e-12, None) for i in range(self.dim)]

        self.res = minimize(self.loss, self.initial_guess, method="L-BFGS-B", jac=self.grad,
                            args=(timestamps, self.dim), bounds=self.bounds,
                            options=self.options)

        print(self.res.x)

        self.mu_estim = np.array(self.res.x[0: self.dim])
        self.alpha_estim = np.array(self.res.x[self.dim: -self.dim]).reshape((self.dim, self.dim))
        self.alpha_estim[np.abs(self.alpha_estim) <= 1e-16] = 0.0
        self.beta_estim = np.array(self.res.x[-self.dim:])

        return self.mu_estim, self.alpha_estim, self.beta_estim


if __name__ == "__main__":
    np.random.seed(0)

    mu = np.zeros((250, 1))
    alpha = np.zeros((250, 250))
    beta = np.zeros((250, 1))

    mu_dev = np.zeros((250, 1))
    alpha_dev = np.zeros((250, 250))
    beta_dev = np.zeros((250, 1))

    number_estimations = np.zeros((250, 250))
    for number in range(1, 11):
        a_file = open("traitements2/train" + str(number) + ".pkl", "rb")
        tList, filtre_dict_orig, orig_dict_filtre = pickle.load(a_file)
        dim = len(filtre_dict_orig)
        a_file.close()

        aux = [[1 if j in orig_dict_filtre.keys() else 0 for j in range(1, 251)] if i in orig_dict_filtre.keys() else [0
                                                                                                                       for
                                                                                                                       j
                                                                                                                       in
                                                                                                                       range(
                                                                                                                           1,
                                                                                                                           251)]
               for i in range(1, 251)]

        number_estimations += np.array(aux)

        # for i in orig_dict_filtre.keys():
        #     number_estimations[i-1] += 1

        estimation = obtain_average_estimation("grad", number, dim, 1)
        mu_est = estimation[:dim]
        alpha_est = estimation[dim:-dim].reshape((dim, dim))
        alpha_est[np.abs(alpha_est) <= 1e-16] = 0
        beta_est = estimation[-dim:]

        # print(filtre_dict_orig)

        for i in range(1, dim + 1):
            mu[filtre_dict_orig[i] - 1] += mu_est[i - 1]
            mu_dev[filtre_dict_orig[i] - 1] += mu_est[i - 1] ** 2
            aux = []
            for j in range(250):
                if j + 1 in filtre_dict_orig.values():
                    aux += [alpha_est[i - 1, orig_dict_filtre[j + 1] - 1]]
                else:
                    aux += [0]

            alpha[filtre_dict_orig[i] - 1, :] += np.array(aux)
            alpha_dev[filtre_dict_orig[i] - 1, :] += np.array(aux) ** 2
            beta[filtre_dict_orig[i] - 1] += beta_est[i - 1]
            beta_dev[filtre_dict_orig[i] - 1] += beta_est[i - 1] ** 2

    number_estimations[number_estimations == 0] = 1
    mu /= np.amax(number_estimations, axis=1).reshape((250, 1))
    alpha /= number_estimations
    beta /= np.amax(number_estimations, axis=1).reshape((250, 1))

    mu_dev = np.sqrt((mu_dev - number_estimations * (mu ** 2)) / (np.maximum(number_estimations - 1, 0)))
    alpha_dev = np.sqrt((alpha_dev - number_estimations * (alpha ** 2)) / (np.maximum(number_estimations - 1, 0)))
    beta_dev = np.sqrt((beta_dev - number_estimations * (beta ** 2)) / (np.maximum(number_estimations - 1, 0)))

    a_file = open("traitements2/kept_dimensions.pkl", "rb")
    estimated_mask = pickle.load(a_file)
    print(np.sum(estimated_mask))
    a_file.close()

    level_conf = 0.95
    print("Number of estimations: ", number_estimations)
    quantile = -t.ppf((1 - level_conf) / 2, np.maximum(number_estimations - 1, 0))
    print(quantile)

    # mu_dev = quantile*(mu_dev)/(np.maximum(number_estimations - 1, 0))
    alpha_dev = quantile * (alpha_dev) / (np.maximum(number_estimations - 1, 0))
    # beta_dev = quantile * (beta_dev) / (np.maximum(number_estimations - 1, 0))

    support = np.invert((alpha - alpha_dev < 0) & (0 < alpha + alpha_dev))
    print(support.shape)

    for number in range(1, 11):
        a_file = open("traitements2/train" + str(number) + ".pkl", "rb")
        tList, filtre_dict_orig, orig_dict_filtre = pickle.load(a_file)
        real_support = support[np.array(list(orig_dict_filtre.keys()))-1, :][:, np.array(list(orig_dict_filtre.keys()))-1]
        print(tList[0], tList[-1])
        print(real_support)
        dim = len(filtre_dict_orig)
        a_file.close()


        file_name = "grad"
        initial = obtain_average_estimation(file_name, number, dim, 1)
        #initial = np.array([1]*dim + [0]*dim*dim + [1]*dim)

        with open("estimation/_traitements2_" + str(number) + 'confidencegrad', 'w', newline='') as myfile:
            wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)

            loglikelihood_estimation = multivariate_estimator_bfgs_conf(dimension=dim, options={"disp": False})

            print("Starting estimation...")
            start_time = time.time()
            res = loglikelihood_estimation.fit(tList, initial=initial, support=real_support)
            end_time = time.time()
            # print(loglikelihood_estimation.res.x)

            wr.writerow(loglikelihood_estimation.res.x.tolist())

        print("Time it took: ", (end_time - start_time) // 60, " minutes.")
        print("")
