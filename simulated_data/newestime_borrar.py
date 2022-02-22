import numpy as np
import csv
from ast import literal_eval as make_tuple
# from class_and_func.estimator_class import multivariate_estimator_bfgs
from scipy import stats
from scipy.optimize import minimize
from class_and_func.likelihood_functions import *
import time


class multivariate_estimator_bfgs(object):
    """
    Estimator class for Exponential Hawkes process obtained through minimizaton of a loss using the L-BFGS-B algorithm.

    Attributes
    ----------
    res : OptimizeResult
        Result from minimization.
    """

    def __init__(self, loss=multivariate_loglikelihood_simplified, dimension=None, initial_guess="random", options=None,
                 penalty=False, C=1, eps=1e-6):
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
        self.penalty = penalty
        if penalty == "l2":
            self.loss = lambda x, y, z: loss(x, y, z) + C * np.linalg.norm(x[-dimension:])
        elif penalty == "rlsquares":
            self.eps = eps
            self.loss = lambda x, y, z, eta, eps: loss(x, y, z) + 0.5 * C * np.sum(
                (x[self.dim: self.dim + self.dim ** 2] ** 2 + eps) / eta) + 0.5 * C * np.sum(eta)
        else:
            self.loss = loss

        self.bounds = [(1e-12, None) for i in range(self.dim)] + [(None, None) for i in range(self.dim * self.dim)] + [
            (1e-12, None) for i in range(self.dim)]
        if isinstance(initial_guess, str) and initial_guess == "random":
            self.initial_guess = np.concatenate(
                (np.concatenate((np.ones(self.dim), np.ones(self.dim * self.dim))), np.ones(self.dim)))
        if options is None:
            self.options = {'disp': False}
        else:
            self.options = options

    def fit(self, timestamps, threshold=0.01, limit=1000, maxiter=15):
        """
        Parameters
        ----------
        timestamps : list of tuple.
            Ordered list containing event times and marks.
        """

        if self.penalty != "rlsquares":
            self.res = minimize(self.loss, self.initial_guess, method="L-BFGS-B",
                                args=(timestamps, self.dim), bounds=self.bounds,
                                options=self.options)
        else:
            self.options['iprint'] = 0
            eps = 1
            self.et = np.abs(self.initial_guess[self.dim: self.dim + self.dim ** 2])
            start_time = time.time()
            self.res = minimize(self.loss, self.initial_guess, method="L-BFGS-B",
                                args=(timestamps, self.dim, self.et, eps), bounds=self.bounds,
                                options=self.options)
            print(time.time()-start_time)
            self.old_et =self.et
            self.et = np.sqrt(np.array(self.res.x[self.dim: self.dim + self.dim ** 2]) ** 2 + eps)
            self.initial_guess = self.res.x
            acc = 1
            eps *= 1/2
            # self.options['maxiter'] = maxiter
            while acc < limit and np.linalg.norm(self.et - self.old_et) > self.eps:
                alpha = np.abs(self.res.x[dim:-dim])
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

                start_time = time.time()
                self.res = minimize(self.loss, self.initial_guess, method="L-BFGS-B",
                                    args=(timestamps, self.dim, self.et, eps), bounds=self.bounds,
                                    options=self.options)
                print(time.time() - start_time, end=" ")
                self.old_et = self.et
                self.et = np.sqrt(np.array(self.res.x[self.dim: -self.dim]) ** 2 + eps)

                print(self.res.x, np.linalg.norm(self.et - self.old_et))
                acc += 1
                eps *= 1 / 2
                self.initial_guess = self.res.x

            print(acc, "   ", np.linalg.norm(self.et - self.old_et))

            alpha = np.abs(self.res.x[dim:-dim])
            ordered_alpha = np.sort(alpha)
            norm = np.sum(ordered_alpha)
            aux, i = 0, 0
            while aux <= threshold:
                aux += ordered_alpha[i]/norm
                i += 1
            i -= 1
            thresh = ordered_alpha[i] # We erase those STRICTLY lower
            self.bounds = [(1e-12, None) for i in range(self.dim)] + [(None, None) if i >= thresh else (0, 1e-16) for i in alpha] + [
                              (1e-12, None) for i in range(self.dim)]
            self.res = minimize(multivariate_loglikelihood_simplified, self.initial_guess, method="L-BFGS-B",
                                args=(timestamps, self.dim), bounds=self.bounds,
                                options=self.options)

        self.mu_estim = np.array(self.res.x[0: self.dim])
        self.alpha_estim = np.array(self.res.x[self.dim: -self.dim]).reshape((self.dim, self.dim))
        self.beta_estim = np.array(self.res.x[-self.dim:])

        return self.mu_estim, self.alpha_estim, self.beta_estim



if __name__ == "__main__":
    np.random.seed(0)
    dim = 2
    number = 2
    first = 1
    C = 10
    threshold = 0.01
    stop_criteria = 1e-4
    with open("estimation_"+str(number)+'_file/_simulation'+str(number), 'r') as read_obj:
        csv_reader = csv.reader(read_obj)
        with open("estimation_"+str(number)+'_file/_estimation'+str(number)+'seuil2', 'w', newline='') as myfile:
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
    until = 2
    with open("estimation_"+str(number)+'_file/_simulation'+str(number), 'r') as read_obj:
        csv_reader = csv.reader(read_obj)
        with open("estimation_"+str(number)+'_file/_estimation'+str(number)+'seuil2', 'a', newline='') as myfile:
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