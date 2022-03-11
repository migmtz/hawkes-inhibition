import numpy as np
import csv
from ast import literal_eval as make_tuple
import pickle
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import cm
from class_and_func.colormaps import get_continuous_cmap


def obtain_average_estimation(file_name, number, dim, number_estimations):
    n = 0
    if file_name[0:4] == "tick":
        if file_name[5:9] == "beta":
            result = np.zeros((dim + dim * dim * dim,))
        else:
            result = np.zeros((dim + dim * dim,))
    else:
        result = np.zeros((2 * dim + dim * dim,))
    with open("estimation/_traitements2_" + str(number) + file_name, 'r') as read_obj:
        csv_reader = csv.reader(read_obj)
        for row in csv_reader:
            if n < number_estimations:
                result += np.array([float(i) for i in row])
                n += 1
    result /= n

    return result


if __name__ == "__main__":
    for number in range(4,5):

        a_file = open("traitements2/train" + str(number) + ".pkl", "rb")
        tList, filtre_dict_orig, orig_dict_filtre = pickle.load(a_file)
        dim = len(filtre_dict_orig)
        a_file.close()

        number_estimations = 1
        annot = False

        counting = [np.sum([1 for (i,j) in tList if j == (k+1)]) for k in range(dim)]
        print(counting)
        print(np.sum(counting))

        plot_names = ["threshgrad50.0"]
        labels = ["MLE"]
        estimations = [obtain_average_estimation(file_name, number, dim, number_estimations) for file_name in plot_names]

        sns.set_theme()
        fig, axr = plt.subplots(1, len(plot_names))
        ax = axr#.T
        hex_list = ['#FF3333', '#FFFFFF', '#33FF49']

        blah = get_continuous_cmap(hex_list)
        # blah.set_bad(color=np.array([1,1,1,1]))
        # wrong_heatmap = np.sign(heat_matrix)-np.sign(-heat_estimated)*(heat_matrix != 0.0)
        # sns.heatmap(wrong_heatmap, ax=ax[2], cmap=get_continuous_cmap(hex_list), center=0, annot=True)

        for ref, estimation in enumerate(estimations):
            if plot_names[ref][0:4] == "tick":
                # if plot_names[ref][5:9] == "beta":
                #     mu_est = estimation[:dim]
                #     alpha_est = np.mean(estimation[dim:].reshape((dim, dim, dim)), axis=0)
                #     beta_est = beta
                # else:
                mu_est = estimation[:dim]
                alpha_est = estimation[dim:].reshape((dim, dim))
                alpha_est[np.abs(alpha_est) <= 1e-16] = 0
                #beta_est = beta

            else:
                mu_est = estimation[:dim]
                alpha_est = estimation[dim:-dim].reshape((dim, dim))
                alpha_est[np.abs(alpha_est) <= 1e-16] = 0
                beta_est = estimation[-dim:]
                #beta_est[beta_est < 1e-10] = 1
                # print(alpha_est)
            heat_estimated = alpha_est# / beta_est
            # heat_estimated[np.abs(heat_estimated) <= 0.01] = 0

            sns.heatmap(heat_estimated, ax=ax, cmap=get_continuous_cmap(hex_list), center=0, annot=annot, linewidths=.5)
            ax.set_title(labels[ref])

            # aux = sign * np.abs(np.abs(np.sign(heat_matrix)) - np.abs(np.sign(heat_estimated)))

            # false_0 = 1-np.sum(sign * np.abs(np.abs(np.sign(heat_matrix)) - np.abs(np.sign(heat_estimated))) == -1) / (
            #     np.sum(sign == -1))
            # false_non_0 = 1-np.sum(sign * np.abs(np.abs(np.sign(heat_matrix)) - np.abs(np.sign(heat_estimated))) == 1) / (
            #     np.sum(sign == 1))

            # aux = np.zeros((dim, dim))
            # aux[np.abs(np.sign(heat_estimated) - np.sign(heat_matrix)) == 2] = -2
            # aux[(heat_estimated != 0.0) * (heat_matrix == 0.0)] = 1
            # aux[(heat_estimated == 0.0) * (heat_matrix != 0.0)] = -1
            # sns.heatmap(aux, ax=ax[ref][1], cmap=get_continuous_cmap(['#000000', '#9B59B6', '#FFFFFF', '#E67E22']), annot=annot, linewidths=.5, vmin=-2, vmax=1)
            # ax[ref][1].set_title(str(np.round(false_0, 2)) + " " + str(np.round(false_non_0, 2)))
    fig2, ax2 = plt.subplots()
    sns.heatmap(beta_est.reshape(len(filtre_dict_orig),1), ax=ax2)
    plt.show()