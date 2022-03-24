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
    mu = np.zeros((250, 1))
    alpha = np.zeros((250, 250))
    beta = np.zeros((250, 1))

    number_estimations = np.zeros((250, 250))
    for number in range(1, 11):
        a_file = open("traitements2/train" + str(number) + ".pkl", "rb")
        tList, filtre_dict_orig, orig_dict_filtre = pickle.load(a_file)
        dim = len(filtre_dict_orig)
        a_file.close()

        aux = [[1 if j in orig_dict_filtre.keys() else 0 for j in range(1, 251)] if i in orig_dict_filtre.keys() else [0 for j in range(1, 251)] for i in range(1, 251)]

        number_estimations += np.array(aux)

        # for i in orig_dict_filtre.keys():
        #     number_estimations[i-1] += 1

        plot_names = ["threshgrad50.0"]
        labels = ["MLE-90"]
        estimation = obtain_average_estimation(plot_names[0], number, dim, 1)
        mu_est = estimation[:dim]
        if plot_names[0] == "tick":
            alpha_est = estimation[dim:].reshape((dim, dim))
            beta_est = 4.5 * np.ones((dim,))

        else:
            alpha_est = estimation[dim:-dim].reshape((dim, dim))
            alpha_est[np.abs(alpha_est) <= 1e-16] = 0
            beta_est = estimation[-dim:]

        #print(filtre_dict_orig)

        for i in range(1, dim+1):
            mu[filtre_dict_orig[i] - 1] += mu_est[i - 1]
            aux = []
            for j in range(250):
                if j+1 in filtre_dict_orig.values():
                    aux += [alpha_est[i - 1, orig_dict_filtre[j+1] - 1]]
                else:
                    aux += [0]

            alpha[filtre_dict_orig[i] - 1, :] += np.array(aux)
            beta[filtre_dict_orig[i] - 1] += beta_est[i - 1]

    number_estimations[number_estimations == 0] = 1
    mu /= np.amax(number_estimations, axis=1).reshape((250,1))
    alpha /= number_estimations
    beta /= np.amax(number_estimations, axis=1).reshape((250,1))

    sns.set_theme()
    fig, axr = plt.subplots(1, len(plot_names))
    ax = axr#.T
    hex_list = ['#FF3333', '#FFFFFF', '#33FF49']
    blah = get_continuous_cmap(hex_list)
    blah.set_bad(color=np.array([1,1,1,1]))

    # for i in range(250):
    #     alpha[i, i] = 0
    a_file = open("traitements2/kept_dimensions.pkl", "rb")
    estimated_mask = pickle.load(a_file)
    print(np.sum(estimated_mask))
    a_file.close()

    #print(alpha[estimated_mask[0], :][:, estimated_mask[0]].shape)
    heatmap = alpha[estimated_mask[0], :][:, estimated_mask[0]]/(beta[estimated_mask[0], :])

    print("1-norm:", np.linalg.norm(np.maximum(heatmap, 0), ord=1))
    print("Spectral radius:", np.max(np.abs(np.linalg.eigvals(np.maximum(heatmap, 0)))))
    mask = heatmap == 0
    #print(heatmap.shape, mask.shape)

    sns.heatmap(heatmap, ax=ax, cmap=blah, center=0, annot=False, linewidths=.5, mask=mask)
    fig2, ax2 = plt.subplots()
    sns.heatmap(beta[estimated_mask[0], :], ax=ax2)

    fig3, ax3 = plt.subplots(1, 4, sharey=True)

    diag_list = []
    beta_list = []
    min_alpha, max_alpha = 0, 0
    min_beta, max_beta = 0, 0
    for z, file_name in enumerate(["grad", "threshgrad90.0"]):
        mu = np.zeros((250, 1))
        alpha = np.zeros((250, 250))
        beta = np.zeros((250, 1))

        number_estimations = np.zeros((250, 250))
        for number in range(1, 11):
            a_file = open("traitements2/train" + str(number) + ".pkl", "rb")
            tList, filtre_dict_orig, orig_dict_filtre = pickle.load(a_file)
            dim = len(filtre_dict_orig)
            a_file.close()

            aux = [
                [1 if j in orig_dict_filtre.keys() else 0 for j in range(1, 251)] if i in orig_dict_filtre.keys() else [
                    0 for j in range(1, 251)] for i in range(1, 251)]

            number_estimations += np.array(aux)

            # for i in orig_dict_filtre.keys():
            #     number_estimations[i-1] += 1

            labels = ["MLE"]
            estimation = obtain_average_estimation(file_name, number, dim, 1)
            mu_est = estimation[:dim]
            alpha_est = estimation[dim:-dim].reshape((dim, dim))
            alpha_est[np.abs(alpha_est) <= 1e-16] = 0
            beta_est = estimation[-dim:]

            #print(filtre_dict_orig)

            for i in range(1, dim + 1):
                mu[filtre_dict_orig[i] - 1] += mu_est[i - 1]
                aux = []
                for j in range(250):
                    if j + 1 in filtre_dict_orig.values():
                        aux += [alpha_est[i - 1, orig_dict_filtre[j + 1] - 1]]
                    else:
                        aux += [0]

                alpha[filtre_dict_orig[i] - 1, :] += np.array(aux)
                beta[filtre_dict_orig[i] - 1] += beta_est[i - 1]

        number_estimations[number_estimations == 0] = 1
        mu /= np.amax(number_estimations, axis=1).reshape((250, 1))
        alpha /= number_estimations
        beta /= np.amax(number_estimations, axis=1).reshape((250, 1))

        beta_list += [beta[estimated_mask[0], :]]
        diag_list += [np.diag(alpha[estimated_mask[0], :][:, estimated_mask[0]])]

        min_alpha = np.minimum(np.min(diag_list[-1].reshape((223, 1))/beta_list[-1].reshape((223, 1))), min_alpha)
        max_alpha = np.maximum(np.max(diag_list[-1].reshape((223, 1))/beta_list[-1].reshape((223, 1))), max_alpha)
        min_beta = np.minimum(np.min(beta_list[-1]), min_beta)
        max_beta = np.maximum(np.max(beta_list[-1]), max_beta)

    aux = diag_list[1].reshape((223, 1))/beta_list[1].reshape((223, 1))
    aux2 = np.sort(aux, axis=None)
    aux3 = np.argsort(aux, axis=None)

    a = beta_list[0].reshape((223, 1))[aux3].reshape((223, 1))
    b = beta_list[1].reshape((223, 1))[aux3].reshape((223, 1))
    c = (diag_list[0].reshape((223, 1))/beta_list[0].reshape((223, 1)))[aux3].reshape((223, 1))
    d = (diag_list[1].reshape((223, 1))/beta_list[1].reshape((223, 1)))[aux3].reshape((223, 1))

    sns.heatmap(a, ax=ax3[0])
    sns.heatmap(b, ax=ax3[3])

    sns.heatmap(c, ax=ax3[1], cmap=blah, center=0)
    sns.heatmap(d, ax=ax3[2], cmap=blah, center=0)

    ax3[1].set_title("grad")
    ax3[2].set_title("thresh")
    plt.show()