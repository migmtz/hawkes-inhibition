import numpy as np
import csv
from ast import literal_eval as make_tuple
import seaborn as sns
from dictionary_parameters import dictionary as param_dict
from matplotlib import pyplot as plt
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
    with open("estimation_"+str(number)+'_file/_estimation'+str(number)+file_name, 'r') as read_obj:
        csv_reader = csv.reader(read_obj)
        for row in csv_reader:
            if n < number_estimations:
                result += np.array([float(i) for i in row])
                n += 1
    result /= n

    return result


if __name__ == "__main__":
    dim = 10
    number = 4
    theta = param_dict[number]
    mu = theta[:dim]
    alpha = theta[dim:-dim].reshape((dim, dim))
    beta = theta[-dim:]
    number_estimations = 1
    annot = False

    plot_names = [""]
    num1 = [0, 1, 1, 2, 2]
    num2 = [1, 1, 2, 1, 2]
    estimations = []

    for file_name in plot_names:
        estimations += [obtain_average_estimation(file_name, number, dim, number_estimations)]

    sns.set_theme()
    fig, ax = plt.subplots(4,3)
    hex_list = ['#FF3333', '#FFFFFF', '#33FF49']

    heat_matrix = alpha/beta
    sns.heatmap(heat_matrix, ax=ax[0][0], cmap=get_continuous_cmap(hex_list), center=0, annot=annot)
    # wrong_heatmap = np.sign(heat_matrix)-np.sign(-heat_estimated)*(heat_matrix != 0.0)
    # sns.heatmap(wrong_heatmap, ax=ax[2], cmap=get_continuous_cmap(hex_list), center=0, annot=True)

    lista_ordenada = np.sort(np.abs(estimations[0][dim:-dim]))

    elbow = lista_ordenada[79]
    print(elbow)

    blah = np.sum(lista_ordenada**2)
    aux = 0
    place = 0
    seuil = 0.01
    for k in lista_ordenada:
        aux += (k**2)/blah
        if aux < seuil:
            place += 1

    lista_ordenada2 = np.sort(np.abs(theta[dim:-dim]))
    blah = np.sum(lista_ordenada2)
    aux2 = 0
    place2 = 0
    seuil2 = 0.01
    for k in lista_ordenada2:
        aux2 += (k) / blah
        print(aux2, aux2 < seuil2)
        if aux2 < seuil2:
            place2 += 1
    print(place2, lista_ordenada2[place2])
    fig2, ax2 = plt.subplots()

    ax2.plot([i for i in range(len(lista_ordenada))], lista_ordenada)
    ax2.plot([place, place], [np.min(lista_ordenada), np.max(lista_ordenada)])

    for ref, estimation in enumerate(estimations):
        if plot_names[ref][0:4] == "tick":
            if plot_names[ref][5:9] == "beta":
                mu_est = estimation[:dim]
                alpha_est = np.mean(estimation[dim:].reshape((dim, dim, dim)), axis=0)
                beta_est = beta
            else:
                mu_est = estimation[:dim]
                alpha_est = estimation[dim:].reshape((dim, dim))
                beta_est = beta
        else:
            mu_est = estimation[:dim]
            alpha_est = estimation[dim:-dim].reshape((dim, dim))
            beta_est = estimation[-dim:]
            print("Error estimation"+plot_names[ref], relative_squared_loss(theta, estimation))
        heat_estimated = alpha_est / beta_est
        heat_elbow = (alpha_est*(np.abs(alpha_est) > elbow))/beta_est
        heat_elbow2 = (alpha_est*(np.abs(alpha_est) > lista_ordenada[place]))/beta_est
        heat_elbow3 = (alpha*(np.abs(alpha) > lista_ordenada2[place2]))/beta_est
        # heat_estimated[np.abs(heat_estimated) <= 0.01] = 0


        sns.heatmap(heat_estimated, ax=ax[num1[ref]][num2[ref]], cmap=get_continuous_cmap(hex_list), center=0, annot=annot)
        sns.heatmap(np.abs(np.abs(np.sign(heat_matrix)) - np.abs(np.sign(heat_estimated))), ax=ax[num1[ref]][num2[ref]+1], cmap=get_continuous_cmap(hex_list), center=0,
                    annot=annot)
        sns.heatmap(heat_elbow, ax=ax[num1[ref]+1][num2[ref]], cmap=get_continuous_cmap(hex_list), center=0,
                    annot=annot)
        sns.heatmap(np.abs(np.abs(np.sign(heat_matrix)) - np.abs(np.sign(heat_elbow))),
                    ax=ax[num1[ref]+1][num2[ref] + 1], cmap=get_continuous_cmap(hex_list), center=0,
                    annot=annot)
        sns.heatmap(heat_elbow2, ax=ax[num1[ref]+2][num2[ref]], cmap=get_continuous_cmap(hex_list), center=0,
                    annot=annot)
        sns.heatmap(np.abs(np.abs(np.sign(heat_matrix)) - np.abs(np.sign(heat_elbow2))),
                    ax=ax[num1[ref]+2][num2[ref] + 1], cmap=get_continuous_cmap(hex_list), center=0,
                    annot=annot)
        sns.heatmap(heat_elbow3, ax=ax[num1[ref] + 3][num2[ref]], cmap=get_continuous_cmap(hex_list), center=0,
                    annot=annot)
        sns.heatmap(np.abs(np.abs(np.sign(heat_matrix)) - np.abs(np.sign(heat_elbow3))),
                    ax=ax[num1[ref] + 3][num2[ref] + 1], cmap=get_continuous_cmap(hex_list), center=0,
                    annot=annot)


    plt.show()