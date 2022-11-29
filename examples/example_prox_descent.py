import numpy as np
import time
from matplotlib import pyplot as plt
from class_and_func.multivariate_exponential_process import multivariate_exponential_hawkes
from class_and_func.likelihood_functions import multivariate_loglikelihood_simplified
from class_and_func.estimator_class import multivariate_estimator_proximal

if __name__ == "__main__":
    ### Simulation of event times
    np.random.seed(0)

    dim = 3

    mu = np.array([1.5, 2, 1.0]).reshape((dim, 1))
    alpha = np.array([[0.2, 0.5, -0.8],
                     [0.0, -0.5, 0.4],
                     [-0.3, 0.0, 0.9]])

    beta = np.array([0.8, 1.1, 1.5]).reshape((dim, 1))

    max_jumps = 500

    hawkes = multivariate_exponential_hawkes(mu=mu, alpha=alpha, beta=beta, max_jumps=max_jumps)

    print("Starting simulation...")
    hawkes.simulate()
    hawkes.plot_intensity(plot_N=True)
    print("Finished simulation")
    print("#"*30)
    print("My real loglikelihood", multivariate_loglikelihood_simplified((mu, alpha, beta), hawkes.timestamps))
    print("#" * 300)

    learner = multivariate_estimator_proximal(dimension=dim, lr=lambda x : 0.01, C=0.1)
    mu_estim, alpha_estim, beta_estim, control_list = learner.fit(hawkes.timestamps)

    print(alpha_estim)
    print("Estimated loglikelihood", multivariate_loglikelihood_simplified((mu_estim.reshape(dim, 1), alpha_estim, beta_estim.reshape(dim, 1)), hawkes.timestamps))

    #plt.show()
