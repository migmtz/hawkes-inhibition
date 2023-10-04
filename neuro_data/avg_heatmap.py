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

        plot_names = ["grad"]
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

    print("1-norm:", np.linalg.norm(np.maximum(heatmap, 0), ord=1))
    print("Spectral radius:", np.max(np.abs(np.linalg.eigvals(np.maximum(heatmap, 0)))))
    mask = heatmap == 0
    #print(heatmap.shape, mask.shape)

    heatmap2 = np.sign(alpha[estimated_mask[0], :][:, estimated_mask[0]])/(beta[estimated_mask[0], :])

    fig, ax = plt.subplots()
    g = sns.heatmap(heatmap2, ax=ax, cmap=blah, center=0, annot=False, linewidths=0.0, mask=mask, xticklabels=10, yticklabels=10, square=True)
    g.set_xticklabels(range(1, np.sum(estimated_mask), 10), rotation=0)
    g.set_yticklabels(range(1, np.sum(estimated_mask), 10), rotation=90)
    ax.set_title("sign(alph)/beta")

    only_matrix = True

    if not(only_matrix):
        figalt, axalt = plt.subplots(figsize=(10, 12))
        blah = LinearSegmentedColormap.from_list('Custom', hex_list, len(hex_list))
        galt = sns.heatmap(np.sign(heatmap), ax=axalt, cmap=blah, center=0, annot=False, linewidths=0.0, mask=mask, xticklabels=10,
                        yticklabels=10, square=False, cbar=True, cbar_kws={"orientation": "horizontal", "pad":0.05})
        galt.set_xticklabels(range(1, np.sum(estimated_mask), 10), rotation=0)
        galt.set_yticklabels(range(1, np.sum(estimated_mask), 10), rotation=90)
        colorbar = galt.collections[0].colorbar
        colorbar.set_ticks([-0.667, 0, 0.667])
        colorbar.set_ticklabels(['Inhibiting interaction', 'No interaction', 'Exciting interaction'])
        axalt.set_title("$(\\tilde{\\alpha}_{ij})_{ij}^+$")
        #figalt.savefig('heatmap_estimation.pdf', bbox_inches='tight', format="pdf", quality=90)


        # fig6, ax6 = plt.subplots()
        # #sns.heatmap(np.sign(heatmap2[arg_aux][:,arg_aux]), ax=ax6, cmap=blah, center=0, annot=False, linewidths=0.0, mask=mask[arg_aux][:,arg_aux], xticklabels=10, yticklabels=10, square=True)
        # sns.heatmap(heatmap[arg_aux][:,arg_aux], ax=ax6, cmap=blah, center=0, annot=False, linewidths=0.0, mask=mask[arg_aux][:,arg_aux], xticklabels=10, yticklabels=10, square=True)

        fig7, ax7 = plt.subplots(2, 2, gridspec_kw={'width_ratios': [7, 1], "height_ratios": [1, 7]}, figsize=(10,10))
        heatmap = np.sign(heatmap)
        sum_hor = np.sum(np.abs(heatmap), axis=0)
        sum_ver = np.sum(np.abs(heatmap), axis=1)
        sum = sum_hor + sum_ver
        arg_aux = np.argsort(-sum)
        g7 = sns.heatmap(heatmap[arg_aux][:, arg_aux], ax=ax7[1, 0], cmap=blah, center=0, annot=False, linewidths=0.0,
                    mask=mask[arg_aux][:, arg_aux], xticklabels=10, yticklabels=10, cbar=False, cbar_kws={"orientation": "horizontal", "pad":0.05})

        # colorbar = g7.collections[0].colorbar
        # colorbar.set_ticks([-0.667, 0, 0.667])
        # colorbar.set_ticklabels(['Inhibiting interaction', 'No interaction', 'Exciting interaction'])
        # #axalt.set_title("$(\\tilde{\\alpha}_{ij})_{ij}^+$")

        ax7[0, 0].bar(x=np.arange(len(sum_hor))+0.5, height=sum_hor[arg_aux], width=1, linewidth=0.0)
        ax7[0, 0].set_xlim(0, len(sum_hor))
        ax7[0, 0].get_xaxis().set_visible(False)
        ax7[0, 0].set_title("Giving interactions")

        ax7[1, 1].barh(y=np.arange(len(sum_ver)) + 0.5, width=sum_ver[arg_aux], height=1, linewidth=0.0)
        ax7[1, 1].set_ylim(len(sum_hor), 0)
        ax7[1, 1].get_yaxis().set_visible(True)
        ax7[1, 1].tick_params("y", left=False)
        ax7[1, 1].get_yaxis().set_ticklabels([])
        ax7[1, 1].tick_params("x", top=True, labeltop=True, bottom=False, labelbottom=False)
        ax7[1, 1].yaxis.set_label_position("right")
        ax7[1, 1].set_ylabel("Receiving interactions", rotation=270, labelpad=20)

        fig7.subplots_adjust(wspace=0.05, hspace=0.05)

        fig7.delaxes(ax7[0,1])
        #fig7.savefig('heatmap_estimation_ordered.pdf', bbox_inches='tight', format="pdf", quality=90)

        # fig2, ax2 = plt.subplots()
        # sns.heatmap(beta[estimated_mask[0], :], ax=ax2, cmap=blah, center=0)
        #
        # fig3, ax3 = plt.subplots(2,1)
        # heatmap = np.sign(heatmap)
        # pos_giv = np.sum(heatmap*(heatmap > 0), axis=0)  # horizontal
        # pos_rec = np.sum(heatmap*(heatmap > 0), axis=1)  # vertical
        #
        # neg_giv = np.sum(heatmap*(heatmap < 0), axis=0)  # horizontal
        # neg_rec = np.sum(heatmap*(heatmap < 0), axis=1)  # vertical
        # arg_aux = np.argsort(-(pos_giv - neg_giv))
        #
        # ax3[0].plot(pos_rec[arg_aux], label="Positive receiving", c="g")
        # ax3[0].plot(neg_rec[arg_aux], label="Negative receiving", c="r")
        # ax3[1].plot(pos_giv[arg_aux], label="Positive giving", c="g")
        # ax3[1].plot(neg_giv[arg_aux], label="Negative giving", c="r")
        # ax3[0].legend()
        # ax3[1].legend()
        sns.set_theme()
        fig3ndiag, ax3ndiag = plt.subplots(2, 1, figsize=(10,8))
        pos_giv = np.sum(heatmap * (heatmap > 0), axis=0) - (heatmap.diagonal() * (heatmap.diagonal() > 0)) # horizontal
        pos_rec = np.sum(heatmap * (heatmap > 0), axis=1) - (heatmap.diagonal() * (heatmap.diagonal() > 0)) # vertical

        neg_giv = np.sum(heatmap * (heatmap < 0), axis=0) - (heatmap.diagonal() * (heatmap.diagonal() < 0))# horizontal
        neg_rec = np.sum(heatmap * (heatmap < 0), axis=1) - (heatmap.diagonal() * (heatmap.diagonal() < 0))# vertical
        arg_aux = np.argsort(-(pos_giv - neg_giv))

        ax3ndiag[0].plot(pos_rec[arg_aux], label="Positive", c="g")
        ax3ndiag[0].plot(neg_rec[arg_aux], label="Negative", c="r")
        ax3ndiag[1].plot(pos_giv[arg_aux], label="Positive", c="g")
        ax3ndiag[1].plot(neg_giv[arg_aux], label="Negative", c="r")
        ax3ndiag[0].set_title("Receiving interactions")
        ax3ndiag[1].set_title("Giving interactions")
        ax3ndiag[0].legend()
        ax3ndiag[1].legend()
        print("Receving null ?", np.sum((pos_rec - neg_rec) == 0), np.arange(len(pos_rec))[(pos_rec - neg_rec) == 0])
        print("Giving null ?", np.sum((pos_giv - neg_giv) == 0), np.arange(len(pos_rec))[(pos_giv - neg_giv) == 0])
        print("Giving null how many receiving", (pos_rec - neg_rec)[(pos_giv - neg_giv) == 0])
        #fig3ndiag.savefig('giv_rec_int.pdf', bbox_inches='tight', format="pdf", quality=90)

        # fig3pos, ax3pos = plt.subplots(2, 1)
        # heatmappos = heatmap[heatmap.diagonal() > 0, :][:, heatmap.diagonal() > 0]
        # pos_giv = np.sum(heatmappos * (heatmappos > 0), axis=0) - 1  # horizontal
        # pos_rec = np.sum(heatmappos * (heatmappos > 0), axis=1) - 1  # vertical
        #
        # neg_giv = np.sum(heatmappos * (heatmappos < 0), axis=0)  # horizontal
        # neg_rec = np.sum(heatmappos * (heatmappos < 0), axis=1)  # vertical
        # arg_aux = np.argsort(-(pos_giv - neg_giv))
        #
        # ax3pos[0].plot(pos_rec[arg_aux], label="Positive receiving", c="g")
        # ax3pos[0].plot(neg_rec[arg_aux], label="Negative receiving", c="r")
        # ax3pos[1].plot(pos_giv[arg_aux], label="Positive giving", c="g")
        # ax3pos[1].plot(neg_giv[arg_aux], label="Negative giving", c="r")
        # ax3pos[0].legend()
        # ax3pos[1].legend()
        # fig3pos.suptitle("Positive auto")
        #
        # fig3neg, ax3neg = plt.subplots(2, 1)
        # heatmapneg = heatmap[heatmap.diagonal() < 0,:][:, heatmap.diagonal() < 0]
        # pos_giv = np.sum(heatmapneg * (heatmapneg > 0), axis=0)  # horizontal
        # pos_rec = np.sum(heatmapneg * (heatmapneg > 0), axis=1)  # vertical
        #
        # neg_giv = np.sum(heatmapneg * (heatmapneg < 0), axis=0) -1  # horizontal
        # neg_rec = np.sum(heatmapneg * (heatmapneg < 0), axis=1) -1  # vertical
        # arg_aux = np.argsort(-(pos_giv - neg_giv))
        #
        # ax3neg[0].plot(pos_rec[arg_aux], label="Positive receiving", c="g")
        # ax3neg[0].plot(neg_rec[arg_aux], label="Negative receiving", c="r")
        # ax3neg[1].plot(pos_giv[arg_aux], label="Positive giving", c="g")
        # ax3neg[1].plot(neg_giv[arg_aux], label="Negative giving", c="r")
        # ax3neg[0].legend()
        # ax3neg[1].legend()
        # fig3neg.suptitle("Negative auto")

        # fig4, ax4 = plt.subplots()
        # for i in range(heatmap_beta.shape[0]):
        #     heatmap_beta[i, i] = 0.0
        # ax4.scatter(heatmap_beta, heatmap_beta.T)

        # fig3, ax3 = plt.subplots()
        # sns.heatmap(np.sum(heatmap, axis=0, keepdims=True), ax=ax3, cmap=blah, center=0)
        #
        # fig4, ax4 = plt.subplots()
        # sns.heatmap(np.sum(heatmap, axis=1, keepdims=True), ax=ax4, cmap=blah, center=0)
        #
        # fig5, ax5 = plt.subplots()
        # sns.heatmap(np.sign(np.multiply(np.sum(heatmap, axis=1, keepdims=True), np.sum(heatmap, axis=0, keepdims=True).T)), ax=ax5, cmap=blah, center=0)
        #
        #sum_hor = np.sum(np.abs(np.sign(heatmap)), axis=0)
        #sum_ver = np.sum(np.abs(np.sign(heatmap)), axis=1)
        #
        # fig8, ax8 = plt.subplots()
        # arg_hor = np.argsort(-(sum_hor - sum_ver))
        # ax8.plot((sum_hor - sum_ver)[arg_hor])
        # ax8.set_title("Giv - Rec")

        # curve, = ax7[0, 1].plot(np.flip(sum_ver[arg_aux]))
        # newx = curve.get_xdata()
        # newy = curve.get_ydata()
        # curve2, = ax7[1, 1].plot(sum_ver[arg_aux])
        # curve2.set_xdata(newy)
        # curve2.set_ydata(newx)
        # ax7[1, 1].set_xlim(ax7[0, 1].get_ylim())
        # ax7[1, 1].set_ylim(ax7[0, 1].get_xlim())

        # # fig3, ax3 = plt.subplots(1, 4, sharey=True)
        #
        # diag_list = []
        # beta_list = []
        # min_alpha, max_alpha = 0, 0
        # min_beta, max_beta = 0, 0
        # for z, file_name in enumerate(["grad", "threshgrad90.0"]):
        #     mu = np.zeros((250, 1))
        #     alpha = np.zeros((250, 250))
        #     beta = np.zeros((250, 1))
        #
        #     number_estimations = np.zeros((250, 250))
        #     for number in range(1, 11):
        #         a_file = open("traitements2/train" + str(number) + ".pkl", "rb")
        #         tList, filtre_dict_orig, orig_dict_filtre = pickle.load(a_file)
        #         dim = len(filtre_dict_orig)
        #         a_file.close()
        #
        #         aux = [
        #             [1 if j in orig_dict_filtre.keys() else 0 for j in range(1, 251)] if i in orig_dict_filtre.keys() else [
        #                 0 for j in range(1, 251)] for i in range(1, 251)]
        #
        #         number_estimations += np.array(aux)
        #
        #         # for i in orig_dict_filtre.keys():
        #         #     number_estimations[i-1] += 1
        #
        #         labels = ["MLE"]
        #         estimation = obtain_average_estimation(file_name, number, dim, 1)
        #         mu_est = estimation[:dim]
        #         alpha_est = estimation[dim:-dim].reshape((dim, dim))
        #         alpha_est[np.abs(alpha_est) <= 1e-16] = 0
        #         beta_est = estimation[-dim:]
        #
        #         #print(filtre_dict_orig)
        #
        #         for i in range(1, dim + 1):
        #             mu[filtre_dict_orig[i] - 1] += mu_est[i - 1]
        #             aux = []
        #             for j in range(250):
        #                 if j + 1 in filtre_dict_orig.values():
        #                     aux += [alpha_est[i - 1, orig_dict_filtre[j + 1] - 1]]
        #                 else:
        #                     aux += [0]
        #
        #             alpha[filtre_dict_orig[i] - 1, :] += np.array(aux)
        #             beta[filtre_dict_orig[i] - 1] += beta_est[i - 1]
        #
        #     number_estimations[number_estimations == 0] = 1
        #     mu /= np.amax(number_estimations, axis=1).reshape((250, 1))
        #     alpha /= number_estimations
        #     beta /= np.amax(number_estimations, axis=1).reshape((250, 1))
        #
        #     beta_list += [beta[estimated_mask[0], :]]
        #     diag_list += [np.diag(alpha[estimated_mask[0], :][:, estimated_mask[0]])]
        #
        #     min_alpha = np.minimum(np.min(diag_list[-1].reshape((223, 1))/beta_list[-1].reshape((223, 1))), min_alpha)
        #     max_alpha = np.maximum(np.max(diag_list[-1].reshape((223, 1))/beta_list[-1].reshape((223, 1))), max_alpha)
        #     min_beta = np.minimum(np.min(beta_list[-1]), min_beta)
        #     max_beta = np.maximum(np.max(beta_list[-1]), max_beta)
        #
        # aux = diag_list[1].reshape((223, 1))/beta_list[1].reshape((223, 1))
        # aux2 = np.sort(aux, axis=None)
        # aux3 = np.argsort(aux, axis=None)
        #
        # a = beta_list[0].reshape((223, 1))[aux3].reshape((223, 1))
        # b = beta_list[1].reshape((223, 1))[aux3].reshape((223, 1))
        # c = (diag_list[0].reshape((223, 1))/beta_list[0].reshape((223, 1)))[aux3].reshape((223, 1))
        # d = (diag_list[1].reshape((223, 1))/beta_list[1].reshape((223, 1)))[aux3].reshape((223, 1))
        #
        # sns.heatmap(a, ax=ax3[0])
        # sns.heatmap(b, ax=ax3[3])
        #
        # sns.heatmap(c, ax=ax3[1], cmap=blah, center=0)
        # sns.heatmap(d, ax=ax3[2], cmap=blah, center=0)
        #
        # ax3[1].set_title("grad")
        # ax3[2].set_title("thresh")
    plt.tight_layout()
    plt.show()