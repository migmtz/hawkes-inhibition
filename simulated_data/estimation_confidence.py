import csv
from dictionary_parameters import dictionary as param_dict
from ast import literal_eval as make_tuple
from scipy.optimize import minimize
from class_and_func.likelihood_functions import *


def obtain_confidence_intervals(file_name, number, dim, number_estimations):
    n = 0
    if file_name[0:4] == "tick":
        if file_name[5:9] == "beta":
            average = np.zeros((number_estimations, dim + dim * dim * dim))
        else:
            average = np.zeros((number_estimations, dim + dim * dim))
    else:
        average = np.zeros((number_estimations, 2 * dim + dim * dim))
    with open("estimation_" + str(number) + '_file/_estimation' + str(number) + file_name, 'r') as read_obj:
        csv_reader = csv.reader(read_obj)
        for row in csv_reader:
            if n < number_estimations:
                average[n, :] += np.array([float(i) for i in row])
                n += 1
    if n != number_estimations:
        print("Wrong number of estimations")
    minimum = np.min(average, axis=0)
    maximum = np.max(average, axis=0)
    average = np.mean(average, axis=0)

    return average, minimum, maximum, n


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
    number_estimations = 5
    annot = False

    avg, minimum, maximum, n = obtain_confidence_intervals("grad", number, dim, number_estimations)

    print("Number of estimations: ", n)

    mu_avg = avg[:dim]
    alpha_avg = avg[dim:-dim].reshape((dim, dim))
    beta_avg = avg[-dim:]

    alpha_min = minimum[dim:-dim].reshape((dim, dim))
    alpha_max = maximum[dim:-dim].reshape((dim, dim))

    support = (alpha_min > 0) | (alpha_max < 0)

    print("Support matrix: ", support)

    first = 1
    before = 1
    until = 5

    with open("estimation_" + str(number) + '_file/_simulation' + str(number), 'r') as read_obj:
        csv_reader = csv.reader(read_obj)
        with open("sample_" + str(number_estimations) + "/estimation_" + str(number) + '_file/_estimation' + str(number) + 'confminmax', 'w', newline='') as myfile:
            i = 1
            wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
            for row in csv_reader:
                while i <= first:
                    print("# ", i)
                    tList = [make_tuple(i) for i in row]

                    loglikelihood_estimation = multivariate_estimator_bfgs_conf(dimension=dim, initial_guess="random", options={"disp": False})
                    res = loglikelihood_estimation.fit(tList, support=support)
                    # print(loglikelihood_estimation.res.x)
                    wr.writerow(loglikelihood_estimation.res.x.tolist())
                    i += 1

    with open("estimation_" + str(number) + '_file/_simulation' + str(number), 'r') as read_obj:
        csv_reader = csv.reader(read_obj)
        with open("sample_" + str(number_estimations) + "/estimation_" + str(number) + '_file/_estimation' + str(number) + 'confminmax', 'a', newline='') as myfile:
            i = 1
            wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
            for row in csv_reader:
                if i <= before:
                    i += 1
                elif before < i <= until:
                    print("# ", i)
                    tList = [make_tuple(i) for i in row]

                    loglikelihood_estimation = multivariate_estimator_bfgs_conf(dimension=dim, initial_guess="random", options={"disp": False})
                    res = loglikelihood_estimation.fit(tList, support=support)
                    # print(loglikelihood_estimation.res.x)
                    wr.writerow(loglikelihood_estimation.res.x.tolist())
                    i += 1
