import numpy as np
import csv
from class_and_func.likelihood_functions import multivariate_loglikelihood_simplified
from ast import literal_eval as make_tuple
from dictionary_parameters import dictionary as param_dict


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


dict_names = {"":0, "grad":1, "tick":2, "tick_bfgs":3, "tick_beta":4, "tick_beta_bfgs":5}
styles = ["solid", "dashdot", "dashed", "dashed", "dotted", "dotted"]
colors = ["orange", "orange", "g", "b", "g", "b"]


if __name__ == "__main__":
    number = 1
    theta = param_dict[number]
    dim = int(np.sqrt(1 + theta.shape[0]) - 1)
    number_estimations = 25

    plot_names = ["thresh10.0", "threshgrad10.0"]

    with open("estimation_" + str(number) + '_file/_simulation' + str(number), 'r') as sim_ob:
        with open("estimation_" + str(number) + '_file/_estimation' + str(number) + plot_names[0], 'r') as read_obj:
            with open("estimation_" + str(number) + '_file/_estimation' + str(number) + plot_names[1], 'r') as read_obj_grad:
                csv_sim = csv.reader(sim_ob)
                list_sim = [row for row in csv_sim]
                csv_reader = csv.reader(read_obj)
                list_thresh = [row for row in csv_reader]
                csv_grad = csv.reader(read_obj_grad)
                list_grad = [row for row in csv_grad]

                for i in range(len(list_thresh)):
                    tList = [make_tuple(j) for j in list_sim[i]]

                    theta_thresh = np.array([float(j) for j in list_thresh[i]])
                    uno = multivariate_loglikelihood_simplified(theta_thresh, tList, dim)
                    print("Normal thresh : ", uno)

                    theta_grad = np.array([float(j) for j in list_grad[i]])
                    dos = multivariate_loglikelihood_simplified(theta_grad, tList, dim)
                    print("Grad thresh : ", dos)

                    print("Meilleur grad ? : ", dos > uno, end="\n\n")