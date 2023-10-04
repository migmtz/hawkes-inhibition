import numpy as np
from functions_fixed import *
from class_and_func.hawkes_process import exp_thinning_hawkes
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import cm
from scipy import stats
from collections import deque
from scipy.optimize import minimize
import time
import seaborn as sns
import pickle


def f_1(n):
    return int(n)


def f_2(n):
    return int(n * np.log(n))


def f_3(n):
    return int(n * np.sqrt(n))


def f_4(n):
    return int(n ** 2)

if __name__ == "__main__":

    mu = 1.0
    alpha = 0.2
    beta = 1.0

    burn_in = -100

    horizons = [1000 + 500 * k for k in range(0, 9)]
    max_time = horizons[-1]
    K_functions = [f_2]
    repetitions = 50
    noise_levels = mu/(1 - alpha) * np.array([0.2 * k for k in range(1, 11)])
    bounds = np.array([(1e-12, None), (1e-12, 1 - 1e-12), (1e-12, None), (1e-12, None)])

    estimations = np.zeros((3, 4, len(horizons), len(K_functions), len(noise_levels), repetitions))
    loglike_real = np.zeros((4, len(horizons), len(K_functions), len(noise_levels), repetitions))
    loglike_estim = np.zeros((4, len(horizons), len(K_functions), len(noise_levels), repetitions))
    estimation_time = np.zeros((4, len(horizons), len(K_functions), len(noise_levels), repetitions))
    simulated_nb_points = np.zeros((4, len(horizons), len(K_functions), len(noise_levels), repetitions))

    ll_functions = [spectral_ll_mu_grad_precomputed, spectral_ll_alpha_grad_precomputed,
                    spectral_ll_beta_grad_precomputed, spectral_ll_noise_grad_precomputed]

    for idx_noise, noise in enumerate(noise_levels):
        parameters = np.array([mu, alpha, beta, noise])
        for idx_repetitions in range(repetitions):
            np.random.seed(idx_repetitions)

            hp = exp_thinning_hawkes(mu, alpha, beta, t=burn_in, max_time=max_time)
            hp.simulate()

            ppp = exp_thinning_hawkes(noise, 0, beta, t=burn_in, max_time=max_time)
            ppp.simulate()

            for idx_horizon, horizon in enumerate(horizons):
                times_hp = [0.0] + [t for t in hp.timestamps if 0 < t < horizon] + [horizon]
                times_pp = [t for t in ppp.timestamps if 0 < t < horizon]

                parasited_times = [0.0] + np.sort(times_hp[1:-1] + times_pp).tolist() + [horizon]

                for idx_K, K_func in enumerate(K_functions):
                    K = K_func(len(parasited_times) - 2)
                    periodogram = np.array([bartlett_periodogram(2 * np.pi * j / horizon, parasited_times) for j in range(1, K + 1)])
                    for idx_parameter, fixed_parameter in enumerate(parameters):
                        theta = parameters[[k for k in range(4) if k != idx_parameter]]
                        start_time = time.time()
                        res = minimize(ll_functions[idx_parameter],
                                       (0.5, 0.5, 0.5),
                                       method="L-BFGS-B", jac=True,
                                       args=(K, parasited_times, periodogram, fixed_parameter),
                                       bounds=bounds[[k for k in range(4) if k != idx_parameter]], options=None)
                        end_time = time.time()

                        estimations[:, idx_parameter, idx_horizon, idx_K, idx_noise, idx_repetitions] = res.x
                        loglike_real[idx_parameter, idx_horizon, idx_K, idx_noise, idx_repetitions] = spectral_ll_noise_grad_precomputed((mu, alpha/beta, beta), K, parasited_times, periodogram, noise)[0]
                        loglike_estim[idx_parameter, idx_horizon, idx_K, idx_noise, idx_repetitions] = res.fun
                        estimation_time[idx_parameter, idx_horizon, idx_K, idx_noise, idx_repetitions] = end_time - start_time
                        simulated_nb_points[idx_parameter, idx_horizon, idx_K, idx_noise, idx_repetitions] = len(parasited_times) - 2

            #if (idx_repetitions+1)%10 == 0:
            print(idx_noise+1, idx_repetitions+1)
        print(np.sum(estimation_time))
        toSave = [idx_noise+1, estimations, loglike_real, loglike_estim, estimation_time, simulated_nb_points]

        with open("saved_estimations", 'wb') as f:
            pickle.dump(toSave, f)
