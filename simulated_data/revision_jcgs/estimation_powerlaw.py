import numpy as np
from matplotlib import pyplot as plt
from class_and_func.multivariate_exponential_process import multivariate_powerlaw_hawkes
import seaborn as sns
import csv
from class_and_func.estimator_class import multivariate_estimator_bfgs_grad


if __name__ == "__main__":
    # Set seed
    np.random.seed(0)

    dim = 2  # 2, 3 ou 4

    mu = np.array([1.0, 1.1]).reshape((2,1))
    alpha = np.array([[0.1, 1.5], [1.0, -0.5]])
    beta = np.array([[1.0, 2.0], [0.1, 1.0]])
    gamma = np.array([[4.0, 4.0], [4.0, 4.0]])

    # mu = np.array([1.0, 1.0]).reshape((2,1))
    # alpha = np.array([[0.1, 1.5], [1.0, -0.5]])
    # beta = np.array([[1.0, 1.1], [1.2, 1.0]])
    # gamma = np.array([[4.0, 4.0], [4.0, 4.0]])

    gamma_list = [1.0]
    max_jumps = 200
    repet = 25

    for gamma_c in gamma_list:
        with open("estimation_extreme_file/_simulation", 'a', newline='') as myfile:
            wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
            with open("estimation_extreme_file/_estimation_grad", 'a',
                      newline='') as estim_file:
                er = csv.writer(estim_file, quoting=csv.QUOTE_ALL)

                for i in range(repet):
                    np.random.seed(i)
                    gamma_corr = gamma_c * gamma

                    hawkes = multivariate_powerlaw_hawkes(mu=mu, alpha=alpha, beta=beta, gamma=gamma_corr, max_jumps=max_jumps)

                    # Create a process with given parameters and maximal number of jumps.

                    hawkes.simulate()
                    tList = hawkes.timestamps

                    wr.writerow(tList)

                    np.random.seed(i)
                    loglikelihood_estimation = multivariate_estimator_bfgs_grad(dimension=dim, options={"disp": False})
                    res = loglikelihood_estimation.fit(tList)
                    er.writerow(loglikelihood_estimation.res.x.tolist())
                    print(i, end=" ")
        print("")
