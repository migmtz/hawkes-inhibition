from class_and_func.hawkes_process import exp_thinning_hawkes
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from functions_and_classes import *
from scipy.optimize import minimize
import time
from scipy.interpolate import CubicSpline


if __name__ == "__main__":
    sns.set_theme()

    np.random.seed(2)
    mu = 1
    alpha = -10.2
    beta = 1.5

    noise = 0.75

    avg_intensity = noise + mu / (1 - alpha / beta)

    max_time = 1000000.0
    burn_in = -100

    x = np.linspace(0.1, 10, 2000)
    f_x = np.array([spectral_f_exp_noised_grad(x_0, (mu, alpha/beta, beta, noise))[0] for x_0 in x])
    IT_x = np.zeros(x.shape)
    debiaised_IT_x = np.zeros(x.shape)
    debiaised_empirical_IT_x = np.zeros(x.shape)

    repet = 10
    for i in range(repet):
        np.random.seed(i)
        hp = exp_thinning_hawkes(mu, alpha, beta, t=burn_in, max_time=max_time)
        hp.simulate()

        print(len(hp.timestamps))
        #times = np.array(hp.timestamps + [hp.timestamps[-1]])
        times_intermediate = np.array([0.0] + [t for t in hp.timestamps if t > 0 and t < max_time/2] + [max_time/2
                                                                                                        ])
        times = np.array([0.0] + [t for t in hp.timestamps if t > 0] + [max_time])

        IT_x += np.array([bartlett_periodogram(x_0, times) for x_0 in x])
        debiaised_IT_x += np.array([debiaised_bartlett_periodogram(x_0, times, avg_intensity) for x_0 in x])
        empirical_avg = (len(times) - 2)/max_time
        debiaised_empirical_IT_x += np.array([debiaised_bartlett_periodogram(x_0, times, empirical_avg) for x_0 in x])
    # conv_IT_x = np.convolve((1 / 50) * np.ones(50), IT_x / repet)[:-49]
    #cum_IT_x = [np.mean((IT_x/repet)[:i+1]) for i in range(len(IT_x))]

    aux = []
    largo = 50

    IT_x /= repet
    debiaised_IT_x /= repet
    debiaised_empirical_IT_x /= repet

    for i in range(largo):
        aux += [np.mean(IT_x[:i+1])]
    for i in range(largo, len(IT_x)):
        aux += [np.mean(IT_x[i + 1 - largo:i + 1])]

    fig, ax = plt.subplots(2, 3, sharex=True)

    #plt.plot(x, conv_IT_x, label="Conv Periodogram")
    #plt.plot(x, conv_IT_x - np.abs(conv_IT_x[-55] - f_x[-55]), label="Conv Periodogram adjusted")
    #plt.plot(x, cum_IT_x, label="Cumul Periodogram")
    ax[0, 0].plot(x, f_x, label="Spectral density")
    ax[0, 0].plot(x, IT_x, c="r", alpha=0.5, label="Periodogram")
    ax[0, 0].plot(x, aux)
    ax[1, 0].plot(x, f_x / IT_x)
    print(np.mean(f_x / IT_x))

    ax[0, 1].plot(x, f_x, label="Spectral density")
    ax[0, 1].plot(x, debiaised_IT_x, c="r", alpha=0.5, label="Debiaised Periodogram")
    ax[1, 1].plot(x, IT_x / debiaised_IT_x)

    ax[0, 2].plot(x, f_x, label="Spectral density")
    ax[0, 2].plot(x, debiaised_empirical_IT_x, c="r", alpha=0.5, label="Debiaised Empirical Periodogram")
    ax[1, 2].plot(x, IT_x / debiaised_empirical_IT_x)

    ax[0, 0].set_title("Periodogram")
    ax[1, 0].set_title("Rapport")
    ax[0, 1].set_title("Debiaised Periodogram")
    ax[0, 2].set_title("Debiaised Empirical Periodogram")

    ax[0, 0].legend()
    ax[0, 1].legend()
    ax[0, 2].legend()
    plt.show()
