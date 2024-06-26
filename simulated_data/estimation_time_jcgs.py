import csv
from dictionary_parameters import dictionary as param_dict
from ast import literal_eval as make_tuple
from scipy.stats import t
from scipy.optimize import minimize
from class_and_func.likelihood_functions import *
from matplotlib import pyplot as plt
import seaborn as sns
import time


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

    def fit(self, timestamps, support):
        """
        Parameters
        ----------
        timestamps : list of tuple.
            Ordered list containing event times and marks.
        """

        self.options['iprint'] = 0
        #print("loss", self.loss)

        support_flat = support.ravel()

        self.bounds = [(1e-12, None) for i in range(self.dim)] + [(None, None) if i else (0, 1e-16)
                                                                  for i in support_flat] + [
                          (1e-12, None) for i in range(self.dim)]
        np.random.seed(0)
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
    number = 7
    theta = param_dict[number]
    dim = int(np.sqrt(1 + theta.shape[0]) - 1)
    mu = theta[:dim]
    alpha = theta[dim:-dim].reshape((dim, dim))
    beta = theta[-dim:]
    number_estimations = 25
    level_conf = 0.95
    annot = False

    label = "grad"
    method = "CfE"

    avg, st_dev, n = obtain_confidence_intervals(label, number, dim, number_estimations)

    print("Number of estimations: ", n)
    quantile = -t.ppf((1 - level_conf) / 2, n - 1)
    print(quantile)

    mu_avg = avg[:dim]
    alpha_avg = avg[dim:-dim].reshape((dim, dim))
    beta_avg = avg[-dim:]

    mu_dev = (st_dev[:dim]) / (np.sqrt(n))
    alpha_dev = (st_dev[dim:-dim].reshape((dim, dim))) / (np.sqrt(n))
    beta_dev = (st_dev[-dim:]) / (np.sqrt(n))

    T_statistic = np.abs(alpha_avg/alpha_dev).ravel()

    p_values = 1 - (t.cdf(T_statistic, n-1) - t.cdf(-T_statistic, n-1))

    x = np.arange(1, len(p_values)+1)
    sns.set_theme()
    fig, ax = plt.subplots(figsize=(12,8))
    ax.scatter(np.arange(1, len(p_values)+1), np.sort(p_values), label="Ordered p-values")
    ax.plot(x, x*(1-level_conf)/len(p_values), label="B-H threshold")

    ord_p_values = np.argsort(p_values)
    reord_p_values = np.argsort(ord_p_values)

    if method == "CfSt":
        support = (p_values[ord_p_values] < x*(1-level_conf)/len(p_values))
        support = support[reord_p_values].reshape((dim, dim))
    else:
        support = np.invert((alpha_avg - alpha_dev < 0) & (0 < alpha_avg + alpha_dev))

    ax.legend()

    fig2, ax2 = plt.subplots()
    sns.heatmap(support, ax=ax2)

    print("Support matrix: ", support)

    until = 25

    computation_time = 0

    with open("estimation_" + str(number) + '_file/_simulation' + str(number), 'r') as read_obj:
        csv_reader = csv.reader(read_obj)
        for i, row in enumerate(csv_reader):
            if i <= until:
                tList = [make_tuple(i) for i in row]

                loglikelihood_estimation = multivariate_estimator_bfgs_conf(dimension=dim, options={"disp": False})
                start_time = time.time()
                res = loglikelihood_estimation.fit(tList, support=support)
                end_time = time.time()

                computation_time += end_time - start_time
            # print(loglikelihood_estimation.res.x)
    computation_time /= until
    with open("revision_jcgs/computation_times.txt", "a") as write_obj:
        wr = csv.writer(write_obj, quoting=csv.QUOTE_ALL)
        wr.writerow([method + " until " + str(until) + " : " + str(computation_time)])
    print(computation_time)

    plt.show()
