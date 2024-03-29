import numpy as np
import csv
from ast import literal_eval as make_tuple
from class_and_func.estimator_class import multivariate_estimator_bfgs
from matplotlib import pyplot as plt
from metrics import relative_squared_loss
from dictionary_parameters import dictionary as param_dict
import seaborn as sns
from matplotlib.patches import Patch


def obtain_average_error(file_name, number, dim, number_estimations, theta):
    n = 0
    result = np.zeros((number_estimations, 4))

    if file_name[0:4] == "conf":
        with open("sample_" + str(number_estimations) + "/estimation_" + str(number) + '_file/_estimation' + str(number) + file_name, 'r') as read_obj:
            print(file_name)
            csv_reader = csv.reader(read_obj)
            for row in csv_reader:
                if n < number_estimations:
                    theta_estimated = np.array([float(i) for i in row])
                    result[n, :] = np.array(relative_squared_loss(theta, theta_estimated))
                    n += 1

    else:

        with open("estimation_"+str(number)+'_file/_estimation'+str(number)+file_name, 'r') as read_obj:
            csv_reader = csv.reader(read_obj)
            for row in csv_reader:
                if n < number_estimations:
                    theta_estimated = np.array([float(i) for i in row])
                    if file_name[0:4] == "tick":
                        theta_estimated = np.concatenate((theta_estimated, theta[-dim:]))
                    result[n, :] = np.array(relative_squared_loss(theta, theta_estimated))
                    n += 1

    return result


colors = ["orange", "orange", "r", "r", "g", "b"]
text = ["$\mu_i$", "$\\alpha_{ij}$", "$\\beta_i$", "Total"]
hatches = ["", "///", "", "///", "", ""]


if __name__ == "__main__":
    dim = 2
    number_grid = [0,1,9]
    number_estimations = 25

    plot_names = ["grad", "threshgrad", "confminmax", "confinterval", "approx", "tick_bfgs"]#, "tick_beta", "tick_beta_bfgs"]
    numbers_thresh = [10.0, 5.0, 3.0]
    labels = [["MLE", "MLE-0.10", "CfE", "CfSt", "Approx", "Lst-sq"]]
    labels += [["MLE", "MLE-0.05", "CfE", "CfSt", "Approx", "Lst-sq"]]
    labels += [["MLE", "MLE-0.03", "CfE", "CfSt", "Approx", "Lst-sq"]]

    sns.set_theme()
    fig, ax = plt.subplots(3, 3, figsize=(20, 12))#, sharey="col")

    for count, number in enumerate(number_grid):
        theta = param_dict[number]
        errors = [np.zeros((number_estimations, len(plot_names))) for i in range(3)]
        for ref, file_name in enumerate(plot_names):
            if file_name == "threshgrad" or file_name == "thresh":
                file_name += str(numbers_thresh[count])
            err = obtain_average_error(file_name, number, dim, number_estimations, theta)
            # print(err.shape)
            for i in range(3):
                errors[i][:, ref] = err[:, i]

        for ref, i in enumerate(errors):
            if ref == 2:
                boxplot = ax[count, ref].boxplot(i[:, 0:5], patch_artist=True)
                ax[count, ref].set_xticklabels(labels[count][0:5], fontdict={"fontsize": 13})
            else:
                boxplot = ax[count, ref].boxplot(i, patch_artist=True)
                ax[count, ref].set_xticklabels(labels[count], fontdict={"fontsize": 13})
            if count == 0:
                ax[count, ref].set_title(text[ref], fontdict={"fontsize": 15})
            for j, patch in enumerate(boxplot['boxes']):
                if len(plot_names[j]) == 10:
                    alpha = 0.75
                    hatch = "///"
                elif len(plot_names[j]) > 10:
                    alpha = 1.0
                    hatch = "///"
                else:
                    alpha = 1.0
                    hatch = ""
                patch.set(alpha=alpha, hatch=hatch, facecolor=colors[j])
        ax[count, 0].set_ylabel("Scenario ("+str(count+1)+")", fontdict={"fontsize": 15})

        #legend_elements = [Patch(facecolor=colors[i], edgecolor='k', hatch=hatches[i], label=labels[i]) for i in range(4)]

    # ax[0, 0].legend(handles=legend_elements, loc='best')
    plt.savefig('revision_jcgs/eps_images/boxplots2O.eps', bbox_inches='tight', format="eps")
    plt.tight_layout()
    #plt.show()