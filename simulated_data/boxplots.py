import numpy as np
import csv
from ast import literal_eval as make_tuple
from class_and_func.estimator_class import multivariate_estimator_bfgs
from matplotlib import pyplot as plt
from metrics import relative_squared_loss
from dictionary_parameters import dictionary as param_dict
import seaborn as sns


if __name__ == "__main__":
    dim = 2
    number = 0
    theta = param_dict[number]
    number_estimations = 95
    errors = np.zeros((number_estimations, 4))
    errors_pen = np.zeros((number_estimations, 4))
    n = 0

    with open('_estimation'+str(number), 'r') as read_obj:
        csv_reader = csv.reader(read_obj)
        for row in csv_reader:
            if n < number_estimations:
                theta_estimated = np.array([float(i) for i in row])
                errors[n, :] = np.array(relative_squared_loss(theta, theta_estimated))
                n += 1
    n = 0
    with open('_estimation'+str(number)+'pen', 'r') as read_obj:
        csv_reader = csv.reader(read_obj)
        for row in csv_reader:
            if n < number_estimations:
                theta_estimated = np.array([float(i) for i in row])
                errors_pen[n, :] = np.array(relative_squared_loss(theta, theta_estimated))
                n += 1

    sns.set_theme()
    fig, ax = plt.subplots()
    ax.boxplot(errors)
    print(errors)

    fig2, ax2 = plt.subplots()
    ax2.boxplot(errors_pen)
    print("\n", errors_pen)

    plt.show()