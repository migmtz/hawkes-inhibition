import numpy as np
import scipy.integrate as integrate
from scipy.optimize import minimize
from ast import literal_eval as make_tuple
import time
import csv


def multivariate_loglikelihood_approx_integral(theta, tList, precision, dim=None, dimensional=False):
    """

    Parameters
    ----------
    theta : tuple of array
        Tuple containing 3 arrays. First corresponds to vector of baseline intensities mu. Second is a square matrix
        corresponding to interaction matrix alpha. Last is vector of recovery rates beta.

    tList : list of tuple
        List containing tuples (t, m) where t is the time of event and m is the mark (dimension). The marks must go from
        1 to nb_of_dimensions.
        Important to note that this algorithm expects the first and last time to mark the beginning and
        the horizon of the observed process. As such, first and last marks must be equal to 0, signifying that they are
        not real event times.
        The algorithm checks by itself if this condition is respected, otherwise it sets the beginning at 0 and the end
        equal to the last time event.
    dim : int
        Number of processes only necessary if providing 1-dimensional theta. Default is None
    dimensional : bool
        Whether to return the sum of loglikelihood or decomposed in each dimension. Default is False.
Returns
    -------
    likelihood : array of float
        Value of likelihood at each process.
        The value returned is the opposite of the mathematical likelihood in order to use minimization packages.
    """

    if isinstance(theta, np.ndarray):
        if dim is None:
            raise ValueError("Must provide dimension to unpack correctly")
        else:
            mu = np.array(theta[:dim]).reshape((dim, 1))
            alpha = np.array(theta[dim:dim * (dim + 1)]).reshape((dim, dim))
            beta = np.array(theta[dim * (dim + 1):]).reshape((dim, 1))
    else:
        mu, alpha, beta = (i.copy() for i in theta)
    beta = beta + 1e-10

    beta_1 = 1/beta

    timestamps = tList.copy()

    # We first check if we have the correct beginning and ending.
    if timestamps[0][1] > 0:
        timestamps = [(0, 0)] + timestamps
    if timestamps[-1][1] > 0:
        timestamps += [(timestamps[-1][0], 0)]

    # Initialise values
    tb, mb = timestamps[1]
    # Compensator between beginning and first event time
    compensator = mu*(tb - timestamps[0][0])
    # Intensity before first jump
    log_i = np.zeros((alpha.shape[0],1))
    log_i[mb-1] = np.log(mu[mb-1])

    ic = mu + alpha[:, [mb - 1]]
    # j=1

    for tc, mc in timestamps[2:]:
        # First we estimate the compensator
        inside_log = (mu - np.minimum(ic, 0))/mu
        # Restart time
        t_star = tb + np.multiply(beta_1, np.log(inside_log))
        exp_tpu = np.exp(-beta * (tc - tb))
        aux = 1/inside_log  # inside_log can't be equal to zero (coordinate-wise)

        #compensator += (t_star < tc)*(np.multiply(mu, tc-t_star) + np.multiply(beta_1, ic-mu)*(aux - exp_tpu))
        aux = [integrate.quad(lambda x: np.maximum(0, mu[k] + (ic[k] - mu[k]) * np.exp(-beta[k] * (x - tb))), tb, tc, epsabs=precision)[0] for k in range(dim)]
        compensator += np.array(aux).reshape(compensator.shape)

        if mc > 0:
            old_ic = ic
            ic = mu + np.multiply((ic - mu), np.exp(-beta*(tc-tb)))

            if ic[mc - 1] <= 0.0:
                # print("oh no")
                # res = 1e8

                # rayon_spec = np.max(np.abs(np.linalg.eig(np.abs(alpha) / beta)[0]))
                # rayon_spec = min(rayon_spec, 0.999)
                # if 2*(tList[-1][0])/(1-rayon_spec) < 0:
                #     print("wut")
                # res = 2*(tList[-1][0])/(1-rayon_spec)

                res = 1e8 #*(np.sum(mu**2) + np.sum(alpha**2) + np.sum(beta**2))
                return res
            else:
                log_i[mc-1] += np.log(ic[mc - 1])

            ic += alpha[:, [mc - 1]]

        tb = tc
    likelihood = log_i - compensator
    if not(dimensional):
        likelihood = np.sum(likelihood)
    return -likelihood


class multivariate_estimator_bfgs_integrate_approx(object):
    """
    Estimator class for Exponential Hawkes process obtained through minimizaton of a loss using the L-BFGS-B algorithm.

    Attributes
    ----------
    res : OptimizeResult
        Result from minimization.
    """

    def __init__(self, loss=multivariate_loglikelihood_approx_integral, dimension=None, initial_guess="random",
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
        self.loss = multivariate_loglikelihood_approx_integral

        self.bounds = [(1e-12, None) for i in range(self.dim)] + [(None, None) for i in range(self.dim * self.dim)] + [
            (1e-12, None) for i in range(self.dim)]
        if isinstance(initial_guess, str) and initial_guess == "random":
            self.initial_guess = np.concatenate(
                (np.concatenate((np.ones(self.dim), np.ones(self.dim * self.dim))), np.ones(self.dim)))
        if options is None:
            self.options = {'disp': False}
        else:
            self.options = options

    def fit(self, timestamps, precision):
        """
        Parameters
        ----------
        timestamps : list of tuple.
            Ordered list containing event times and marks.
        """

        np.random.seed(0)
        self.res = minimize(self.loss, self.initial_guess, method="L-BFGS-B",
                            args=(timestamps, precision, self.dim), bounds=self.bounds,
                            options=self.options)

        self.mu_estim = np.array(self.res.x[0: self.dim])
        self.alpha_estim = np.array(self.res.x[self.dim: self.dim + self.dim ** 2]).reshape((self.dim, self.dim))
        self.beta_estim = np.array(self.res.x[-self.dim:])

        return self.mu_estim, self.alpha_estim, self.beta_estim


if __name__ == "__main__":

    dim = 2
    theta = np.array([0.5, 1.0, -1.9, 3.0, 1.2, 1.5, 5.0, 8.0])
    mu, alpha, beta = theta[:dim], theta[dim:-dim].reshape((dim, dim)), theta[-dim:]

    precision_list = [10**i for i in range(-8, 2)]
    precision_list = [10 ** i for i in range(-6, -2)]

    number = 0
    until = 5

    method = "approx"

    for precision in precision_list:

        computation_time = 0

        with open("estimation_" + str(number) + '_file/_simulation' + str(number), 'r') as read_obj:
            csv_reader = csv.reader(read_obj)
            with open("revision_jcgs/estimation_extreme_file/_estimation_approx_random", 'a', newline='') as estim_file:
                er = csv.writer(estim_file, quoting=csv.QUOTE_ALL)
                for i, row in enumerate(csv_reader):
                    if i < until:
                        #np.random.seed(i)

                        tList = [make_tuple(i) for i in row]

                        loglikelihood_estimation = multivariate_estimator_bfgs_integrate_approx(dimension=dim, options={"disp": False})
                        start_time = time.time()
                        res = loglikelihood_estimation.fit(tList, precision)
                        end_time = time.time()

                        computation_time += end_time - start_time

                        er.writerow(loglikelihood_estimation.res.x.tolist())
                    # print(loglikelihood_estimation.res.x)
                er.writerow("")
        computation_time /= until
        with open("revision_jcgs/computation_times.txt", "a") as write_obj:
            wr = csv.writer(write_obj, quoting=csv.QUOTE_ALL)
            wr.writerow([str(precision) + method + " with precision of " + str(precision) + " until " + str(until) + " : " + str(computation_time)])
        print(computation_time)
