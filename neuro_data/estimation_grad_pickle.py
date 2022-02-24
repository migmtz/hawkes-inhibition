import numpy as np
import pickle
from class_and_func.estimator_class import multivariate_estimator_bfgs
from class_and_func.likelihood_functions import multivariate_loglikelihood_simplified, multivariate_loglikelihood_with_grad
from scipy.optimize import minimize
import csv
import time


class multivariate_estimator_bfgs_grad(object):
    """
    Estimator class for Exponential Hawkes process obtained through minimizaton of a loss using the L-BFGS-B algorithm.

    Attributes
    ----------
    res : OptimizeResult
        Result from minimization.
    """

    def __init__(self, loss=multivariate_loglikelihood_simplified, grad=True, dimension=None, initial_guess="random",
                 options=None, penalty=False, C=1, eps=1e-6):
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
            if isinstance(grad, bool) and grad:
                self.loss = multivariate_loglikelihood_with_grad_pen
                self.grad = True
            else:
                self.loss = lambda x, y, z, eta, eps: loss(x, y, z) + 0.5 * C * np.sum(
                    (x[self.dim: self.dim + self.dim ** 2] ** 2 + eps) / eta) + 0.5 * C * np.sum(eta)
                self.grad = False

        else:
            if isinstance(grad, bool) and grad:
                self.loss = multivariate_loglikelihood_with_grad
            else:
                self.loss = loss
            self.grad = grad

        self.bounds = [(1e-12, None) for i in range(self.dim)] + [(None, None) for i in range(self.dim * self.dim)] + [
            (1e-12, None) for i in range(self.dim)]
        if isinstance(initial_guess, str) and initial_guess == "random":
            self.initial_guess = np.concatenate(
                (np.concatenate((np.ones(self.dim), np.zeros(self.dim * self.dim))), np.ones(self.dim)))
        if options is None:
            self.options = {'disp': False}
        else:
            self.options = options

    def fit(self, timestamps, limit=1000):
        """
        Parameters
        ----------
        timestamps : list of tuple.
            Ordered list containing event times and marks.
        """

        if self.penalty != "rlsquares":
            self.res = minimize(self.loss, self.initial_guess, method="L-BFGS-B", jac=self.grad,
                                args=(timestamps, self.dim), bounds=self.bounds,
                                options=self.options)
        else:
            if self.grad:
                self.et = np.ones((self.dim*self.dim))
                self.old_et = self.et + 2 * self.eps
                acc = 1
                eps = 1
                while acc < limit and np.linalg.norm(self.et - self.old_et) > self.eps:
                    print(acc, "   ", self.initial_guess)
                    self.res = minimize(self.loss, self.initial_guess, method="L-BFGS-B", jac=self.grad,
                                        args=(timestamps, eps, self.dim, self.et, eps), bounds=self.bounds,
                                        options=self.options)
                    self.old_et = self.et
                    self.et = np.sqrt(np.array(self.res.x[self.dim: self.dim + self.dim ** 2]) ** 2 + eps)
                    print(self.old_et, self.et)
                    acc += 1
                    eps *= 1 / 2
                    self.initial_guess = self.res.x
            else:
                self.et = np.abs(self.initial_guess[self.dim: self.dim + self.dim ** 2])
                self.old_et = self.et + 2 * self.eps
                acc = 1
                eps = 1
                while acc < limit and np.linalg.norm(self.et - self.old_et) > self.eps:
                    print(acc, "   ", np.linalg.norm(self.et - self.old_et))
                    self.res = minimize(self.loss, self.initial_guess, method="L-BFGS-B",
                                        args=(timestamps, self.dim, self.et, eps), bounds=self.bounds,
                                        options=self.options)
                    self.old_et = self.et
                    self.et = np.sqrt(np.array(self.res.x[self.dim: self.dim + self.dim ** 2]) ** 2 + eps)
                    print(self.old_et, self.et)
                    acc += 1
                    eps *= 1 / 2
                    self.initial_guess = self.res.x

        self.mu_estim = np.array(self.res.x[0: self.dim])
        self.alpha_estim = np.array(self.res.x[self.dim: self.dim + self.dim ** 2]).reshape((self.dim, self.dim))
        self.beta_estim = np.array(self.res.x[-self.dim:])

        return self.mu_estim, self.alpha_estim, self.beta_estim


if __name__ == "__main__":
    numbers = [9]

    for number in numbers:
        np.random.seed(0)

        a_file = open("traitements2/train" + str(number) + ".pkl", "rb")
        tList, filtre_dict_orig, orig_dict_filtre = pickle.load(a_file)
        # tList = [(t,m) for t,m in tList]
        dim = len(filtre_dict_orig)
        # print(filtre_dict_orig)
        # print(np.sum([m==212 for i,m in tList]))
        a_file.close()
        with open("estimation/_traitements2_"+str(number) + 'grad', 'w', newline='') as myfile:
            wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)

            loglikelihood_estimation = multivariate_estimator_bfgs_grad(dimension=dim, options={"disp": False})

            print("Starting estimation...")
            start_time = time.time()
            res = loglikelihood_estimation.fit(tList)
            end_time = time.time()
            # print(loglikelihood_estimation.res.x)

            wr.writerow(loglikelihood_estimation.res.x.tolist())

        print("Time it took: ", (end_time - start_time) // 60, " minutes.")
