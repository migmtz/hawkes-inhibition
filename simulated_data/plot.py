import numpy as np
import csv
from ast import literal_eval as make_tuple
from class_and_func.estimator_class import multivariate_estimator_bfgs
from matplotlib import pyplot as plt
# from class_and_func.streamline_tick import four_estimation
from metrics import relative_squared_loss
from dictionary_parameters import dictionary as param_dict
import seaborn as sns

if __name__ == "__main__":
    dim = 2
    number = 1
    theta = param_dict[number]
    theta_estimated = np.zeros((2*dim + dim*dim,))
    number_estimations = 25
    n = 0
    with open('_estimation'+str(number), 'r') as read_obj:
        csv_reader = csv.reader(read_obj)
        for row in csv_reader:
            if n < number_estimations:
                theta_estimated += np.array([float(i) for i in row])
                n += 1
    theta_estimated /= n
    # print("Estimation", theta_estimated)

    theta_pen = np.zeros((2 * dim + dim * dim,))
    n = 0
    with open('_estimation' + str(number) + 'pen', 'r') as read_obj:
        csv_reader = csv.reader(read_obj)
        for row in csv_reader:
            if n < number_estimations:
                theta_pen += np.array([float(i) for i in row])
                n += 1
    theta_pen /= n
    # print("Penalized", theta_pen)

    print("Error estimation", relative_squared_loss(theta, theta_estimated))
    print("Error penalized", relative_squared_loss(theta, theta_pen))

    ####### PLOT

    sns.set_theme()

    fig, ax = plt.subplots(dim, dim)

    x = np.linspace(0, 2, 100)

    for i in range(dim):
        for j in range(dim):
            ax[i, j].plot(x, theta[dim + dim*i + j] * np.exp(-theta[dim + dim*dim + i] * x), c="r", label="Real kernel")
            ax[i, j].plot(x, theta_estimated[dim + dim*i + j] * np.exp(-theta_estimated[dim + dim*dim + i] * x), c="m", label="Estimated kernel", alpha=0.75)
            ax[i, j].plot(x, theta_pen[dim + dim*i + j] * np.exp(-theta_pen[dim + dim*dim + i] * x), c="m", label="Penalized kernel", linestyle=":", alpha=0.75)

    plt.legend()
    plt.show()