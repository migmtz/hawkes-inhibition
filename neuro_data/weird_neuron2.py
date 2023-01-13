import numpy as np
import csv
from ast import literal_eval as make_tuple
import pickle
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
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
    with open("estimation_resamples/_resamples_" + str(number) + file_name, 'r') as read_obj:
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
    only_heatmap = True
    heat_weird = np.zeros((20, 250))
    for number in range(1, 21):
        a_file = open("resamples/resample" + str(number) + "", "rb")
        tList, filtre_dict_orig, orig_dict_filtre = pickle.load(a_file)
        dim = len(filtre_dict_orig)
        a_file.close()

        aux = [[1 if j in orig_dict_filtre.keys() else 0 for j in range(1, 251)] if i in orig_dict_filtre.keys() else [0 for j in range(1, 251)] for i in range(1, 251)]

        number_estimations += np.array(aux)

        # for i in orig_dict_filtre.keys():
        #     number_estimations[i-1] += 1

        plot_names = ["minmax"]
        labels = ["Grad"]
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
            mu[int(filtre_dict_orig[i]) - 1] += mu_est[i - 1]
            aux = []
            for j in range(250):
                if j+1 in filtre_dict_orig.values():
                    aux += [alpha_est[i - 1, orig_dict_filtre[j+1] - 1]]
                else:
                    aux += [0]

            alpha[int(filtre_dict_orig[i]) - 1, :] += np.array(aux)
            if int(filtre_dict_orig[i]) == 15:
                heat_weird[number - 1, :] += np.array(aux)

    number_estimations[number_estimations == 0] = 1
    mu /= np.amax(number_estimations, axis=1).reshape((250,1))
    alpha /= number_estimations
    beta /= np.amax(number_estimations, axis=1).reshape((250,1))

    # fig, axr = plt.subplots(1, len(plot_names))
    # ax = axr#.T
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
    heatmap = alpha[estimated_mask[0], :][:, estimated_mask[0]]#/(beta[estimated_mask[0], :])
    heatmap_beta = alpha[estimated_mask[0], :][:, estimated_mask[0]]/(beta[estimated_mask[0], :])
    print(alpha)

    print("1-norm:", np.linalg.norm(np.maximum(heatmap, 0), ord=1))
    print("Spectral radius:", np.max(np.abs(np.linalg.eigvals(np.maximum(heatmap, 0)))))
    mask = heatmap == 0
    #print(heatmap.shape, mask.shape)

    heatmap2 = np.sign(alpha[estimated_mask[0], :][:, estimated_mask[0]])/(beta[estimated_mask[0], :])


    fig, ax = plt.subplots(figsize=(10, 12))
    print(np.sum(np.abs(heatmap_beta) > 0))
    g = sns.heatmap(np.sign(heat_weird), ax=ax, cmap=blah, center=0, annot=False, linewidths=0.0, xticklabels=10, yticklabels=10)
    ax.set_title("sign(alph)/beta")

    plt.tight_layout()
    #figalt.savefig('heatmap_real_minmax.pdf', bbox_inches='tight', format="pdf", quality=90)
    plt.show()