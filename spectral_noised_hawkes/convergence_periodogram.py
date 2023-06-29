from class_and_func.hawkes_process import exp_thinning_hawkes
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from functions_and_classes import *
from scipy.optimize import minimize
import time


if __name__ == "__main__":
    sns.set_theme()

    np.random.seed(2)
    mu = 1
    alpha = 0.0
    beta = 1

    noise = 0.75

    max_time = 400.0

    hp = exp_thinning_hawkes(mu, alpha, beta, max_time=max_time)
    hp.simulate()

    print(len(hp.timestamps))
    times = np.array(hp.timestamps + [hp.timestamps[-1]])[100:]

    x = np.linspace(0, 10, 1000)
    f_x = np.array([spectral_f_exp(x_0, (mu, alpha, beta)) for x_0 in x])
    IT_x = np.array([bartlett_periodogram(x_0, times) for x_0 in x])
    cum_IT_x = np.convolve((1/5)*np.ones(5), IT_x)[4:]

    fig, ax = plt.subplots()

    plt.plot(x, f_x, label="Spectral density")
    plt.plot(x, cum_IT_x, label="Periodogram")

    plt.legend()
    plt.show()
