import numpy as np
import scipy.integrate as integrate
from scipy.optimize import minimize
from ast import literal_eval as make_tuple
import time
import csv
from matplotlib import pyplot as plt
from matplotlib.legend_handler import HandlerTuple
import seaborn as sns


if __name__ == "__main__":

    dim = 2
    theta = np.array([0.5, 1.0, -1.9, 3.0, 1.2, 1.5, 5.0, 8.0])
    mu, alpha, beta = theta[:dim], theta[dim:-dim].reshape((dim, dim)), theta[-dim:]

    points_list = range(2,16)
    times_list = [266.796, 289.575, 292.7496, 295.3232, 297.9376, 298.8028, 307.5832, 321.3271, 358.848, 390.471, 421.831, 437.448, 449.896, 496.192]
    print(np.mean(times_list))

    number = 0
    until = 25
    total = len(points_list)

    method = "approx"

    result = np.zeros((2 * dim + dim * dim,))
    result_approx = np.zeros((2 * dim + dim * dim, total))
    for k, nb_points in enumerate(points_list):
        with open("revision_jcgs/estimation_approx_riemann/_estimation_approx_" + str(nb_points), 'r') as read_obj:
            csv_approx = csv.reader(read_obj)
            for i, row in enumerate(csv_approx):
                if i < until:
                    result_approx[:, k] += np.array([float(i) for i in row])
    result_approx /= until

    with open("estimation_0_file/_estimation0grad", 'r') as read_obj:
        csv_reader = csv.reader(read_obj)
        for i, row in enumerate(csv_reader):
            if i < until:
                result += np.array([float(i) for i in row])
    result /= until

    result_scipy = np.zeros((2 * dim + dim * dim))
    which = 0
    with open("revision_jcgs/estimation_extreme_file/_estimation_rmse_approx", 'r') as read_obj:
        csv_approx = csv.reader(read_obj)
        for i, row in enumerate(csv_approx):
            if which*until < i < (which+1)*until:# until * total:
                result_scipy += np.array([float(i) for i in row])
    result_scipy /= until

    approx_errors = np.sqrt(np.mean((result_approx - theta.reshape((theta.shape[0], 1))) ** 2, axis=0))
    rmse_error = np.sqrt(np.mean((result - theta) ** 2))
    scipy_error = np.sqrt(np.mean((result_scipy - theta) ** 2))
    print(rmse_error, scipy_error)
    #print(result_approx, result)

    sns.set_theme()
    # fig_points, ax_points = plt.subplots()
    # ax_points.plot(points_list, approx_errors, label="Integral Approximation")
    # ax_points.scatter(points_list, approx_errors)
    # #ax_times.plot(times_list[:total], [rmse_error for _ in times_list[:total]], label = "(MLE) error")
    # ax_points.scatter([58.47], [rmse_error], label="(MLE) error")
    # plt.legend()

    aux_sort = np.argsort(times_list)
    #print(aux_sort)
    fig_times, ax_times = plt.subplots(figsize=(14, 6))
    t1 = ax_times.scatter([58.47], [rmse_error], label="(MLE)", c="r")
    p1 = ax_times.plot(np.array(times_list)[aux_sort], approx_errors, "-o", label="(MLE) with Riemann integration")
    t2 = ax_times.scatter([625.975], [scipy_error], label="(MLE) with SciPy integration", c="g")
    ax_times.set_ylabel("RMSE")
    ax_times.set_xlabel("Computation time (s)")
    plt.legend()

    fig_times.savefig('rmse_approx.pdf', bbox_inches='tight', format="pdf", quality=90, pad_inches=0.1)

    #plt.show()

    # print(result, result_approx)
    # print("Errors", end="\n")
    # print("Error real: ", np.sqrt((((result - theta) / theta) ** 2).mean()))
    # print("Error approx: ", np.sqrt((np.abs((result_approx - theta) / theta) ** 2).mean()))

#     dim = 2
#     theta = np.array([0.5, 1.0, -1.9, 3.0, 1.2, 1.5, 5.0, 8.0])
#     mu, alpha, beta = theta[:dim], theta[dim:-dim].reshape((dim, dim)), theta[-dim:]
#
#     precision_list = [10 ** i for i in range(-8, 2)]
#     times_list = [1039.74, 900.00, 837.39, 798.88, 777.03, 700.85, 625.97, 581.37, 563.06, 563.12]
#
#     number = 0
#     total = 10
#     until = 25
#
#     method = "approx"
#
#     result = np.zeros((2 * dim + dim * dim,))
#     result_approx = np.zeros((2 * dim + dim * dim, total))
#     with open("revision_jcgs/estimation_extreme_file/_estimation_rmse_approx", 'r') as read_obj:
#         csv_approx = csv.reader(read_obj)
#         for i, row in enumerate(csv_approx):
#             if i < 250 and i%25 < until:# until * total:
#                 result_approx[:, i // 25] += np.array([float(i) for i in row])
#     result_approx /= until
#     with open("estimation_0_file/_estimation0grad", 'r') as read_obj:
#         csv_reader = csv.reader(read_obj)
#         for i, row in enumerate(csv_reader):
#             if i < until:
#                 result += np.array([float(i) for i in row])
#     result /= until
#
#     approx_errors = np.sqrt(np.mean((result_approx - theta.reshape((theta.shape[0], 1))) ** 2, axis=0))
#     rmse_error = np.sqrt(np.mean((result - theta) ** 2))
#     print(result_approx, result)
#
#     sns.set_theme()
#     fig_times, ax_times = plt.subplots()
#     ax_times.plot(times_list[:total], approx_errors, label="Integral Approximation")
#     ax_times.scatter(times_list[:total], approx_errors)
#     #ax_times.plot(times_list[:total], [rmse_error for _ in times_list[:total]], label = "(MLE) error")
#     ax_times.scatter([58.47], [rmse_error], label="(MLE) error")
#     plt.legend()
#
#     fig_precision, ax_precision = plt.subplots()
#     ax_precision.plot(precision_list[:total], approx_errors, label="Integral Approximation")
#     ax_precision.plot(precision_list[:total], [rmse_error for _ in precision_list[:total]], label="(MLE) error")
#     ax_precision.set_xscale("log")
#
#     plt.legend()
#     plt.show()
#
#     # print(result, result_approx)
#     # print("Errors", end="\n")
#     # print("Error real: ", np.sqrt((((result - theta) / theta) ** 2).mean()))
#     # print("Error approx: ", np.sqrt((np.abs((result_approx - theta) / theta) ** 2).mean()))
#
