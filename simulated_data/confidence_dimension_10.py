import numpy as np
import csv
import seaborn as sns
from dictionary_parameters import dictionary as param_dict
from matplotlib import pyplot as plt
from class_and_func.colormaps import get_continuous_cmap
from metrics import relative_squared_loss
from scipy.stats import t
from scipy.optimize import minimize
from class_and_func.likelihood_functions import *


def obtain_average_estimation(file_name, number, dim, number_estimations):
    n = 0
    if file_name[0:4] == "tick":
        if file_name[5:9] == "beta":
            result = np.zeros((dim + dim * dim * dim,))
        else:
            result = np.zeros((dim + dim * dim,))
    else:
        result = np.zeros((2 * dim + dim * dim,))
    with open("estimation_"+str(number)+'_file/_estimation'+str(number)+file_name, 'r') as read_obj:
        csv_reader = csv.reader(read_obj)
        for row in csv_reader:
            if n < number_estimations:
                result += np.array([float(i) for i in row])
                n += 1
    result /= n

    return result

def obtain_confidence_intervals(file_name, number, dim, number_estimations):
    n = 0
    if file_name[0:4] == "tick":
        if file_name[5:9] == "beta":
            average = np.zeros((dim + dim * dim * dim,))
            st_dev = np.zeros((dim + dim * dim * dim,))
        else:
            average = np.zeros((dim + dim * dim,))
            st_dev = np.zeros((dim + dim * dim,))
    else:
        average = np.zeros((2 * dim + dim * dim,))
        st_dev = np.zeros((2 * dim + dim * dim,))
    with open("estimation_" + str(number) + '_file/_estimation' + str(number) + file_name, 'r') as read_obj:
        csv_reader = csv.reader(read_obj)
        for row in csv_reader:
            if n < number_estimations:
                aux = np.array([float(i) for i in row])
                aux[np.abs(aux) < 1e-15] = 0
                print(aux)
                average += aux
                st_dev += aux**2
                n += 1
    average /= n
    st_dev = np.sqrt((st_dev - n*(average**2))/(n-1))

    return average, st_dev, n


if __name__ == "__main__":
    number = 7
    theta = param_dict[number]
    dim = int(np.sqrt(1 + theta.shape[0]) - 1)
    mu = theta[:dim]
    alpha = theta[dim:-dim].reshape((dim, dim))
    beta = theta[-dim:]
    number_estimations = 25
    level_conf = 0.95
    annot = False

    plot_names = ["grad"]
    labels = ["MLE"]
    estimations = [obtain_confidence_intervals(file_name, number, dim, number_estimations) for file_name in plot_names]
    n = estimations[0][2]
    print("Number of estimations: ", n)
    quantile = -t.ppf((1 - level_conf) / 2, n - 1)
    print(quantile)

    mu_avg = estimations[0][0][:dim]
    alpha_avg = estimations[0][0][dim:-dim].reshape((dim, dim))
    beta_avg = estimations[0][0][-dim:]

    print(np.sum((0 < alpha_avg) & (alpha_avg < 1e-14)))

    mu_dev = quantile*(estimations[0][1][:dim])/(np.sqrt(n-1))
    alpha_dev = quantile*(estimations[0][1][dim:-dim].reshape((dim, dim)))/(np.sqrt(n-1))
    beta_dev = quantile*(estimations[0][1][-dim:])/(np.sqrt(n-1))

    sns.set_theme()
    #fig, axr = plt.subplots(dim, dim, figsize=(8, 10))
    #fig2, ax2 = plt.subplots(dim, dim, figsize=(8, 10))

    fig, axr = plt.subplots(dim, 2*dim + 1, figsize=(17, 10))

    for i in range(dim):
        fig.delaxes(axr[i][dim])
        for j in range(dim):
            avg = alpha_avg[i, j]
            points = [avg, avg - alpha_dev[i, j], avg + alpha_dev[i, j]]
            original_sign = np.sign(alpha[i, j])

            #if points[1] <= alpha[i, j] <= points[2]:
            #    color = "blue"
            #else:
            #    color = "red"

            if np.sign(alpha[i, j]) != 0:
                if np.sign(points[1]) == original_sign and original_sign == np.sign(points[2]):
                    color = "blue"
                else:
                    color = "red"
                axr[i, j].scatter([0 for i in range(3)], points, s=5, color="k", marker="x")
                axr[i, j].plot([-1, 1], [0, 0], c="k", linewidth=1)
                axr[i, j].fill_between([-0.2, 0.2], [points[1],points[1]], [points[2], points[2]], alpha=0.5, color=color)
                fig.delaxes(axr[i][j+dim+1])
                axr[i, j].set_title(f"$\\alpha_{{i,j}} = {alpha[i, j]}$", fontdict={'fontsize': 7})
                axr[i, j].xaxis.set_visible(False)
                axr[i, j].yaxis.set_visible(False)
            else:
                if np.sign(points[1]) <= original_sign <= np.sign(points[2]):
                    color = "blue"
                else:
                    color = "red"
                axr[i, j+dim+1].scatter([0 for i in range(3)], points, s=5, color="k", marker="x")
                axr[i, j+dim+1].plot([-1, 1], [0, 0], c="k", linewidth=1)
                axr[i, j+dim+1].fill_between([-0.2, 0.2], [points[1],points[1]], [points[2], points[2]], alpha=0.5, color=color)
                fig.delaxes(axr[i][j])
                axr[i, j+dim+1].xaxis.set_visible(False)
                axr[i, j+dim+1].yaxis.set_visible(False)

    #fig.suptitle("Non-null interactions")
    #fig2.suptitle("Null interactions")

    fig.tight_layout()
    plt.savefig("pres.pdf", format="pdf")

    fig3, ax3 = plt.subplots(10, 2)

    for i in range(dim):
        avg = mu_avg[i]
        points = [avg, avg - mu_dev[i], avg + mu_dev[i]]

        if points[1] <= mu[i] <= points[2]:
            color = "blue"
        else:
            color = "red"

        ax3[i, 0].scatter([0 for i in range(3)], points, s=5, color="k", marker="x")
        ax3[i, 0].plot([-0.3, 0.3], [mu[i], mu[i]], color="k", linestyle="--")
        ax3[i, 0].fill_between([-0.2, 0.2], [points[1], points[1]], [points[2], points[2]], alpha=0.5, color=color)
        ax3[i, 0].set_title(f"$\\mu_{{i}} = {mu[i]}$", fontdict={'fontsize': 7})
        ax3[i, 0].xaxis.set_visible(False)
        ax3[i, 0].yaxis.set_visible(False)

        avg = beta_avg[i]
        points = [avg, avg - beta_dev[i], avg + beta_dev[i]]

        if points[1] <= beta[i] <= points[2]:
            color = "blue"
        else:
            color = "red"

        ax3[i, 1].scatter([0 for i in range(3)], points, s=5, color="k", marker="x")
        ax3[i, 1].plot([-0.3, 0.3], [beta[i], beta[i]], color="k", linestyle="--")
        ax3[i, 1].fill_between([-0.2, 0.2], [points[1], points[1]], [points[2], points[2]], alpha=0.5, color=color)
        ax3[i, 1].set_title(f"$\\beta_{{i}} = {beta[i]}$", fontdict={'fontsize': 7})
        ax3[i, 1].xaxis.set_visible(False)
        ax3[i, 1].yaxis.set_visible(False)

    plt.show()