import numpy as np
import csv
from ast import literal_eval as make_tuple
import seaborn as sns
from dictionary_parameters import dictionary as param_dict
from matplotlib import pyplot as plt
from matplotlib import cm
from class_and_func.colormaps import get_continuous_cmap
from metrics import relative_squared_loss


def obtain_average_estimation(file_name, number, dim, number_estimations):
    n = 0
    if file_name[0:4] == "tick":
        if file_name[5:9] == "beta":
            result = np.zeros((dim + dim * dim * dim,))
        else:
            result = np.zeros((dim + dim * dim,))
    else:
        result = np.zeros((2 * dim + dim * dim,))

    if file_name[0:4] == "conf":
        with open("sample_" + str(number_estimations) + "/estimation_" + str(number) + '_file/_estimation' + str(number) + file_name, 'r') as read_obj:
            print(file_name)
            csv_reader = csv.reader(read_obj)
            for row in csv_reader:
                if n < number_estimations:
                    result += np.array([float(i) for i in row])
                    n += 1
    else:
        with open("estimation_"+str(number)+'_file/_estimation'+str(number)+file_name, 'r') as read_obj:
            csv_reader = csv.reader(read_obj)
            for row in csv_reader:
                if n < number_estimations:
                    result += np.array([float(i) for i in row])
                    n += 1
    result /= n
    print(file_name, n)

    return result

def obtain_errors_estimation(file_name, number, theta, number_estimations):
    n = 0
    errors = np.zeros((number_estimations, 4))

    if file_name[0:4] == "conf":
        with open("sample_" + str(number_estimations) + "/estimation_" + str(number) + '_file/_estimation' + str(number) + file_name, 'r') as read_obj:
            print(file_name)
            csv_reader = csv.reader(read_obj)
            for row in csv_reader:
                if n < number_estimations:
                    estimation = np.array([float(i) for i in row])
                    errors[n, :] += relative_squared_loss(theta, estimation)
                    n += 1
    elif file_name[0:4] == "tick":
        with open("estimation_"+str(number)+'_file/_estimation'+str(number)+file_name, 'r') as read_obj:
            csv_reader = csv.reader(read_obj)
            for row in csv_reader:
                if n < number_estimations:
                    estimation = np.array([float(i) for i in row])
                    errors[n, :] += relative_squared_loss(theta, np.concatenate((estimation, beta.squeeze())))
                    n += 1
    else:
        with open("estimation_"+str(number)+'_file/_estimation'+str(number)+file_name, 'r') as read_obj:
            csv_reader = csv.reader(read_obj)
            for row in csv_reader:
                if n < number_estimations:
                    estimation = np.array([float(i) for i in row])
                    errors[n, :] += relative_squared_loss(theta, estimation)
                    n += 1
    print(file_name, n)

    return errors

colors = ["orange", "orange", "r", "r", "g", "b"]


if __name__ == "__main__":
    number = 7
    theta = param_dict[number]
    dim = int(np.sqrt(1 + theta.shape[0]) - 1)
    mu = theta[:dim]
    alpha = theta[dim:-dim].reshape((dim, dim))
    beta = theta[-dim:]
    number_estimations = 25
    annot = False

    # plot_names = ["grad", "threshgrad25.0", "approx", "tick_bfgs"]
    # labels = ["MLE", "MLE-$0.25$", "Approx", "Lst-sq"]
    plot_names = ["grad", "threshgrad5.0", "confminmax", "confinterval", "approx", "tick_bfgs"]
    labels = ["MLE", "MLE-$0.05$", "CfE", "CfSt", "Approx", "Lst-sq"]
    estimations = [obtain_average_estimation(file_name, number, dim, number_estimations) for file_name in plot_names]

    sns.set_theme()
    fig_or, ax_or = plt.subplots()
    plt.tight_layout()
    fig, axr = plt.subplots(2, len(plot_names), figsize=(18, 5))
    ax = axr.T
    hex_list = ['#FF3333', '#FFFFFF', '#33FF49']

    blah = get_continuous_cmap(hex_list)
    # blah.set_bad(color=np.array([1,1,1,1]))

    heat_matrix = alpha/beta
    sign = heat_matrix.copy()
    sign[sign != 0] = -1
    sign[sign == 0] = 1
    print(sign)
    # mask = heat_matrix == 0

    sns.heatmap(heat_matrix, ax=ax_or, cmap=blah, center=0, annot=annot, square=True, linewidths=.5, xticklabels=range(1,dim+1), yticklabels=range(1,dim+1))#, mask=mask)
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
            alpha_est[np.abs(alpha_est) <= 1e-15] = 0
            beta_est = beta
            print("Error estimation " + plot_names[ref],
                  relative_squared_loss(theta, np.concatenate((estimation, beta.squeeze()))))
        else:
            mu_est = estimation[:dim]
            alpha_est = estimation[dim:-dim].reshape((dim, dim))
            alpha_est[np.abs(alpha_est) <= 1e-15] = 0
            beta_est = estimation[-dim:]
            print("Error estimation"+plot_names[ref], relative_squared_loss(theta, estimation))
            #print(alpha_est)
            print(np.sqrt((beta -beta_est)**2), beta, beta_est)
        heat_estimated = alpha_est / beta_est
        # heat_estimated[np.abs(heat_estimated) <= 0.01] = 0

        g = sns.heatmap(heat_estimated, ax=ax[ref][0], cmap=get_continuous_cmap(hex_list), center=0, annot=annot, linewidths=.5, xticklabels=range(1,dim+1), yticklabels=range(1,dim+1))
        g.set_xticklabels(g.get_xticklabels(), rotation=0)
        g.set_xticklabels([])
        if ref > 0:
            g.set_yticklabels([])
        ax[ref][0].set_title(labels[ref])

        # aux = sign * np.abs(np.abs(np.sign(heat_matrix)) - np.abs(np.sign(heat_estimated)))

        # false_0 = 1-np.sum(sign * np.abs(np.abs(np.sign(heat_matrix)) - np.abs(np.sign(heat_estimated))) == -1) / (
        #     np.sum(sign == -1))
        # false_non_0 = 1-np.sum(sign * np.abs(np.abs(np.sign(heat_matrix)) - np.abs(np.sign(heat_estimated))) == 1) / (
        #     np.sum(sign == 1))

        aux = np.zeros((dim, dim))
        aux[np.abs(np.sign(heat_estimated) - np.sign(heat_matrix)) == 2] = -2
        aux[(heat_estimated != 0.0) * (heat_matrix == 0.0)] = 1
        aux[(heat_estimated == 0.0) * (heat_matrix != 0.0)] = -1
        g = sns.heatmap(aux, ax=ax[ref][1], cmap=get_continuous_cmap(['#000000', '#9B59B6', '#FFFFFF', '#E67E22']), annot=annot, linewidths=.5, vmin=-2, vmax=1, xticklabels=range(1,dim+1), yticklabels=range(1,dim+1))
        g.set_xticklabels(g.get_xticklabels(), rotation=0)
        if ref > 0:
            g.set_yticklabels([])
        # ax[ref][1].set_title(str(np.round(false_0, 2)) + " " + str(np.round(false_non_0, 2)))

    #plt.tight_layout()
    fig_or.savefig('revision_jcgs/eps_images/Realheat.eps', bbox_inches='tight', format="eps")
    fig.savefig('revision_jcgs/eps_images/heatmap_hori.eps', bbox_inches='tight', format="eps")

    fig_box, ax_box = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    for ref, file_name in enumerate(plot_names):
        errors = obtain_errors_estimation(file_name, number, theta, number_estimations)
        for i in range(3):
            if ref <= 4:
                boxplot = ax_box[i].boxplot(errors[:, i], positions=[ref], patch_artist=True)
            else:
                if i != 2:
                    boxplot = ax_box[i].boxplot(errors[:, i], positions=[ref], patch_artist=True)
                    #ax_box[i].set_xticklabels(labels[ref], fontdict={"fontsize": 13})
            try:
                for j, patch in enumerate(boxplot['boxes']):
                    if len(plot_names[ref]) == 10:
                        alpha = 0.75
                        hatch = "///"
                    elif len(plot_names[ref]) > 10:
                        alpha = 1.0
                        hatch = "///"
                    else:
                        alpha = 1.0
                        hatch = ""
                    patch.set(alpha=alpha, hatch=hatch, facecolor=colors[ref])
            except:
                pass
    ax_box[0].set_title("$\\mu_i$")
    ax_box[1].set_title("$\\alpha_{ij}$")
    ax_box[2].set_title("$\\beta_i$")
    ax_box[2].set_xticks(range(6))
    ax_box[2].set_xticklabels(labels, fontdict={"fontsize": 13})

    fig_box.savefig('revision_jcgs/eps_images/boxplots_10_dim.eps', bbox_inches='tight', format="eps")

    #plt.show()