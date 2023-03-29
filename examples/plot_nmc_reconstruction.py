import numpy as np
from class_and_func.hawkes_process import multi_simple_hawkes
from matplotlib import pyplot as plt
from pathlib import Path
import seaborn as sns
import os


def f(base, kernels, K, delta):
    array_res = np.zeros(2 + 4 * K)
    array_res[0:2] = base
    for i in range(2):
        for j in range(2):
            k = 1
            for v in range(kernels[i][j].shape[1]):
                while k * delta <= kernels[i][j][0, v]:
                    array_res[2 + 2 * i * K + j * K + k - 1] = kernels[i][j][1, v]
                    k += 1
    return array_res


def compute_coef_over_one(estimation, delta, eta, K):
    result = [0]
    auxiliary_estimation = np.array(estimation.tolist() + [0])
    ratio = delta / eta

    result += [auxiliary_estimation[0] * ratio]

    for k in range(K):
        result += [(auxiliary_estimation[k + 1] - auxiliary_estimation[k]) * (1 - ratio) + auxiliary_estimation[k]]
        result += [(auxiliary_estimation[k + 1] - auxiliary_estimation[k]) * (ratio) + auxiliary_estimation[k]]

    result += [0]

    return result


def h1_easy(ak, ak1, delta, eta):  # Need the difference. \bar{a_k}
    return delta + eta * (ak1 / (ak - ak1))


def h1_hard(akm, ak, ak1, delta, eta):
    return (delta * (akm - ak) - eta * akm) / (ak1 - akm)


def l1_error_over_one(estimation, real, delta, eta):
    K = estimation.shape[0]
    aux_estimation = estimation.tolist() + [0]
    aux_real = real.tolist() + [0]
    ratio = delta / eta

    akm = 0
    ak = estimation[0] - real[0]
    error = np.abs(ak) * ((delta ** 2) / eta)
    right_sign = np.sign(ak)
    for k in range(1, K + 1):
        ak1 = aux_estimation[k] - aux_real[k]
        left_sign = np.sign((ak1 - ak) * (1 - ratio) + ak)

        if right_sign * left_sign >= 0.0:
            error += (eta - delta) * np.abs(ak1 + akm - ratio * (ak1 - 2 * ak + akm))
        else:
            h1 = h1_hard(akm, ak, ak1, delta, eta)
            error += np.abs(h1 * (ratio * (ak - akm) + akm) + (eta - delta - h1) * (ratio * (ak1 - ak) - ak1))

        right_sign = np.sign((ak1 - ak) * (ratio) + ak)

        if right_sign * left_sign >= 0.0:
            error += (2 * delta - eta) * np.abs(ak1 + ak)
        else:
            h1 = h1_easy(ak, ak1, delta, eta)
            error += np.abs((2 * delta - eta) * (ratio * (ak1 - ak) + ak) - h1 * (ak1 + ak))

        akm = ak
        ak = ak1

    error += (((eta - delta) ** 2) / eta) * np.abs(aux_estimation[K - 1] - aux_real[K - 1])
    if error < 0.0:
        print(delta, eta)
        print("negative shit")
    return 0.5 * error


def l1_error_under_one(estimation, real, delta, eta):
    K = estimation.shape[0]
    aux_estimation = estimation.tolist() + [0]
    aux_real = real.tolist() + [0]
    ratio = eta / delta
    coef = eta / 2

    ak = 0
    error = 0

    for i in range(K + 1):
        ak1 = aux_estimation[i] - aux_real[i]
        if np.sign(ak1 * ak) >= 0.0:
            error += coef * np.abs(ak1 + ak)
        else:
            error += coef * (ak1 ** 2 + ak ** 2) / (ak1 - ak)

        ak = ak1

        if ratio < 1.0:
            error += (delta - eta) * np.abs(ak)

    return error


def l1_error(estimation, real, delta, eta):
    if eta <= delta:
        return l1_error_under_one(estimation, real, delta, eta)
    else:
        return l1_error_over_one(estimation, real, delta, eta)


def step_error(estimation, real, delta):
    error = 0

    for i in range(estimation.shape[0]):
        error += np.abs(estimation[i] - real[i])

    return delta * error


if __name__ == "__main__":

    # np.random.seed(5)

    plt.rcParams['axes.grid'] = True
    # plt.rcParams['lines.linewidth'] = 0.1
    K = 5
    lamb = 1

    start = 1
    tot = 100  # 500

    delta = 1000
    coef = 1 / np.sqrt(delta)

    norm_lambda = 0.002
    coefficient = 0.1
    baselines = 0.5 * coefficient * np.array([norm_lambda, norm_lambda])
    kernels = [[np.array([[delta], [0.25 * (1 - coefficient) * norm_lambda]]),
                np.array([[delta], [0.5 * (1 - coefficient) * norm_lambda]])],
               [np.array([[0.0], [0.0]]), np.array([[delta], [0.25 * (1 - coefficient) * norm_lambda]])]]

    print(baselines)
    print(kernels)

    array_kernels = f(baselines, kernels, K, delta)

    num_baseline = 0.0005

    noises = np.array([0.5 * i * delta for i in range(1, 9)])

    estimated_no_noise = np.zeros((2 + 4 * K, len(noises), tot))
    estimated_noised = np.zeros((2 + 4 * K, len(noises), tot))
    estimated_interval = np.zeros((2 + 4 * K, len(noises), tot))

    for i, noise in enumerate(noises):
        print(i)
        path = "nmc_comparison/Eta" + str(int(noise)) + "/"
        for l in range(start, start + tot):

            no_noise_path = "hawkes_HawkesRealPoint" + str(100 * l) + "_HawkesRealPoint" + str(
                100 * l + 1) + "_forward_K_" + str(K) + "_delta_" + str(delta) + "_kernel_none_lambda_" + str(
                lamb) + ".txt"

            noised_path = "hawkes_HawkesNoisedPoint" + str(100 * l) + "_HawkesNoisedPoint" + str(
                100 * l + 1) + "_forward_K_" + str(K) + "_delta_" + str(delta) + "_kernel_none_lambda_" + str(
                lamb) + ".txt"

            interval_path = "hawkes_HawkesNoisedInterval" + str(100 * l) + "_HawkesNoisedInterval" + str(
                100 * l + 1) + "_forward_K_" + str(K) + "_delta_" + str(
                delta) + "_kernel_heterogeneous_interval_lambda_" + str(lamb) + ".txt"

            f_no_noise = open(Path("nmc_comparison/Eta500/" + no_noise_path), 'r')
            f_noised = open(Path(path + noised_path), 'r')
            f_interval = open(Path(path + interval_path), 'r')

            for r in range(1 + 2 * K):

                row_no_noise = f_no_noise.readline()
                for j, est in enumerate(row_no_noise.split()):
                    if r == 0:
                        estimated_no_noise[j, i, l - 1] = np.abs((float(est) - array_kernels[j]))
                    else:
                        estimated_no_noise[2 + 2 * K * j + r - 1, i, l - 1] = coef * float(est)

                row_noised = f_noised.readline()
                for j, est in enumerate(row_noised.split()):
                    if r == 0:
                        estimated_noised[j, i, l - 1] = np.abs((float(est) - array_kernels[j]))
                    else:
                        estimated_noised[2 + 2 * K * j + r - 1, i, l - 1] = coef * float(est)

                row_interval = f_interval.readline()
                for j, est in enumerate(row_interval.split()):
                    if r == 0:
                        estimated_interval[j, i, l - 1] = np.abs((float(est) - array_kernels[j]))
                    else:
                        estimated_interval[2 + 2 * K * j + r - 1, i, l - 1] = coef * float(est)

            f_no_noise.close()
            f_noised.close()
            f_interval.close()

    print("here")

    x = np.array([0.5 * i for i in range(1, 9)])

    norms = [np.sum(np.abs(baselines))]
    norms += [step_error(np.zeros(5), array_kernels[2:7], delta) + step_error(np.zeros(5), array_kernels[17:],
                                                                               delta) + step_error(np.zeros(5),
                                                                                                   array_kernels[7:12],
                                                                                                   delta) + step_error(
        np.zeros(5), array_kernels[12:17], delta)]

    fig, ax = plt.subplots(1, 1, sharex=True, sharey=True)

    kernels_noised = np.array([[(step_error(estimated_no_noise[2:7, i, j], array_kernels[2:7], delta) + step_error(
        estimated_no_noise[17:, i, j], array_kernels[17:], delta) + step_error(estimated_no_noise[7:12, i, j],
                                                                             array_kernels[7:12], delta) + step_error(
        estimated_no_noise[12:17, i, j], array_kernels[12:17], delta)) / norms[1] for j in range(tot)] for i in
                               range(len(noises))])
    ax.plot(x, np.mean(kernels_noised, axis=1), c="g", linestyle=":", label="Point")
    #
    # # Noised interval
    #
    # baseline_sum = np.mean(np.sum(np.abs(estimated_interval[0:2, :, :]), axis=0), axis=1) / norms[0]
    # ax.plot(x, baseline_sum, c="#0055b3", label="Baselines")
    #

    kernels_noised = np.array([[(step_error(estimated_noised[2:7, i, j], array_kernels[2:7], delta) + step_error(estimated_noised[17:, i, j], array_kernels[17:], delta) + step_error(estimated_noised[7:12, i, j], array_kernels[7:12], delta) + step_error(estimated_noised[12:17, i, j], array_kernels[12:17], delta)) / norms[1] for j in range(tot)] for i in range(len(noises))])
    ax.plot(x, np.mean(kernels_noised, axis=1), c="b", linestyle="--", label="Noised Point")

    # Noised interval

    # aux_noises_1 = [i / 2 for i in x]
    #
    # baseline_sum = np.mean(np.sum(np.abs(estimated_interval[0:2, :, :]), axis=0), axis=1) / norms[0]
    # ax.plot(x, baseline_sum, c="#0055b3", label="Baselines")

    kernels_interval = np.array([[(step_error(estimated_interval[2:7, i, j], array_kernels[2:7],
                                           delta) + step_error(estimated_interval[17:, i, j],
                                                               array_kernels[17:], delta) + step_error(
        estimated_interval[7:12, i, j], array_kernels[7:12], delta) + step_error(
        estimated_interval[12:17, i, j], array_kernels[12:17], delta)) / norms[1] for j in
                               range(tot)] for i in range(len(noises))])
    ax.plot(x, np.mean(kernels_interval, axis=1), c="#BF0A30", label="Noised Interval")

    # print(baseline_sum[0], np.mean(kernels_error, axis=1)[0], np.mean(kernels_error, axis=1)[0] + baseline_sum[0])

    # ax.plot(x, np.mean(kernels_error, axis=1) - baseline_sum, c="#059033", label="Total")

    ax.set_xlabel(r"$\eta_{max}/ \delta$")
    ax.set_ylabel("Relative error")
    plt.legend()

    ax.set_ylim(-0.01, ax.get_ylim()[1])

    fig1, axbis = plt.subplots(2, 2, sharey=True)
    fig2, axtris = plt.subplots(2, 2, sharey=True)

    axs = [axbis, axtris]

    estimates_no_noise_mean = np.mean(estimated_no_noise[:, :, :], axis=2)
    estimates_noised_mean = np.mean(estimated_noised[:, :, :], axis=2)
    estimates_interval_mean = np.mean(estimated_interval[:, :, :], axis=2)

    for i,s in enumerate([1, 5]):
        axchos = axs[i]
        print(noises[s])

        for i in range(2):
            for j in range(2):
                x = [0] + [k for k in kernels[i][j][0] for l in range(2)] + [delta * K]
                y = [k for k in kernels[i][j][1] for l in range(2)] + [0, 0]
                axchos[i, j].plot(x, y, c="m", label="Real Kernel")

                x_point = [0] + [delta * k for k in range(1, K) for l in range(2)] + [delta * K]
                y_point = [k for k in estimates_no_noise_mean[5 * j + 10 * i + 2: 5 * j + 10 * i + 7, s] for l in range(2)]
                axchos[i, j].plot(x_point, y_point, c="g", label="Point Estimation", linestyle="--", alpha=0.75)

                y_noised_point = [k for k in estimates_noised_mean[5 * j + 10 * i + 2:5 * j + 10 * i + 7, s] for l in
                                  range(2)]
                axchos[i, j].plot(x_point, y_noised_point, c="b", label="Noised Point Estimation", linestyle="--",
                                 alpha=0.75)

                y_inteval = [k for k in estimates_interval_mean[5 * j + 10 * i + 2:5 * j + 10 * i + 7, s] for l in
                             range(2)]
                axchos[i, j].plot(x_point, y_inteval, c="#BF0A30", label="Noised Interval Estimation", linestyle="--",
                                 alpha=0.75)

            axchos[1, i].set_xlabel("t")

        axchos[0, 0].legend()



    plt.show()
