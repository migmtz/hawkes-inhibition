import numpy as np
import time
from matplotlib import pyplot as plt
from class_and_func.multivariate_exponential_process import multivariate_exponential_hawkes
from class_and_func.likelihood_functions import multivariate_lstsquares_simplified
from class_and_func.estimator_class import multivariate_estimator_bfgs
from tick.hawkes import HawkesExpKern, HawkesSumExpKern

if __name__ == "__main__":
    ### Simulation of event times
    np.random.seed(2)

    dim = 2  # 2, 3 ou 4

    mu = np.array([[0.5], [1.0]])
    alpha = np.array([[-0.9, 3], [1.2, 1.5]])
    beta = np.array([[4], [5]])
    max_jumps = 200

    hawkes = multivariate_exponential_hawkes(mu=mu, alpha=alpha, beta=beta, max_jumps=max_jumps)

    print("Starting simulation...")
    hawkes.simulate()
    print("Finished simulation")

    loglikelihood_estimation = multivariate_estimator_bfgs(dimension=dim, options={"disp":False})
    print("Starting loglikelihood...")
    start_time = time.time()
    loglikelihood_estimation.fit(hawkes.timestamps)
    end_time = time.time() - start_time

    print("Estimation through loglikelihood: ", np.round(loglikelihood_estimation.res.x, 3), "\nIn: ", end_time)

    least_sq_estimation = multivariate_estimator_bfgs(loss=multivariate_lstsquares_simplified, dimension=dim, options={"disp": False})
    print("Starting least-squares...")
    start_time = time.time()
    least_sq_estimation.fit(hawkes.timestamps)
    end_time = time.time() - start_time

    print("Estimation through least-squares:", np.round(least_sq_estimation.res.x, 3), "\nIn: ", end_time)

    ################################## TICK

    list_tick = [[np.array([t for t, m in hawkes.timestamps if (m - 1) == i]) for i in range(dim)]]

    # With 'likelihood' goodness of fit, you must provide a constant decay for all kernels
    learnersq = HawkesExpKern(decays=np.double(np.c_[beta,beta]), solver="bfgs")
    # learnerlog = HawkesExpKern(decays=np.mean(beta),gofit="likelihood")

    # learnerlog.fit(list_tick)
    learnersq.fit(list_tick)

    # print("log_tick", learnerlog.adjacency*np.c_[beta,beta])
    print("log_sq", learnersq.baseline, "\n", learnersq.adjacency*np.c_[beta,beta])
