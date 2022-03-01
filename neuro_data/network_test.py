import numpy as np
import csv
from ast import literal_eval as make_tuple
import pickle
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import cm
from class_and_func.colormaps import get_continuous_cmap
import networkx as nx

from networkx.convert_matrix import to_numpy_matrix
from numpy.linalg import eig
from scipy.sparse import csc_matrix
from matplotlib.pyplot import cm
import networkx


class Newman2CommunityClassifier():

    def __init__(self, graph, B=None, m=None):
        self.G = graph
        self.A = to_numpy_matrix(graph)
        self.k = np.sum(self.A, axis=1)
        self.m = np.sum(self.k) / 2
        self.B = self.A - np.dot(self.k, self.k.transpose()) / (2 * self.m)
        self.leading_eigenvector = None
        self.category = {node: [] for node in self.G.nodes}
        self.s = None
        self.done = False
        self.Q = 0
        self.G_positive = None
        self.G_negative = None

    def fit(self):
        vals, vecs = eig(self.B)
        self.leading_eigenvector = np.ravel(vecs[:, np.argmax(vals)])
        self.s = np.array([1 if v >= 0 else -1 for v in self.leading_eigenvector])
        for i, node in enumerate(self.G.nodes):
            self.category[node].append(self.s[i])
        nodes = np.array(self.G.nodes)
        self.G_positive = self.G.subgraph(nodes[self.s == 1])
        self.G_negative = self.G.subgraph(nodes[self.s == -1])
        self.Q = np.einsum("i,ij,j", self.s, self.B, self.s) / (4 * self.m)
        if self.Q <= 0 or np.max(self.leading_eigenvector) * np.min(
                self.leading_eigenvector) > 0:  # All elements of the same sign or negative modularity
            self.done = True


def plot_communities(G, clf):
    # Labelize lists
    dict_aux = {}
    dict_labels = {}
    i = -1
    for key, val in clf.category.items():
        if dict_aux.get(tuple(val)) is None:
            i += 1
        a = dict_aux.setdefault(tuple(val), i)
        dict_labels.setdefault(key, a)
    print(dict_aux)
    # Plot parameters
    pos = networkx.kamada_kawai_layout(G)
    rainbow = cm.rainbow(np.linspace(0, 1, len(dict_aux)))

    plt.figure()
    for k in range(len(dict_aux)):
        nodes = [i for i in dict_labels.keys() if dict_labels[i] == k]
        networkx.draw_networkx_nodes(G, pos,
                                     nodelist=nodes,
                                     node_color=rainbow[k].reshape(1, 4),
                                     node_size=200,
                                     node_shape='o',
                                     label=str(k),
                                     alpha=0.8)

    networkx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)

    plt.legend()
    plt.show()


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

        plot_names = ["threshgrad95.0"]
        labels = ["MLE"]
        estimation = obtain_average_estimation(plot_names[0], number, dim, 1)
        mu_est = estimation[:dim]
        alpha_est = estimation[dim:-dim].reshape((dim, dim))
        alpha_est[np.abs(alpha_est) <= 1e-16] = 0
        beta_est = estimation[-dim:]

        print(filtre_dict_orig)

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
    print(alpha[0:2, 0:2], number_estimations[0:2, 0:2])
    alpha /= number_estimations
    print(alpha[0:2, 0:2])
    beta /= np.amax(number_estimations, axis=1).reshape((250,1))

    sns.set_theme()
    fig, axr = plt.subplots(1, len(plot_names))
    ax = axr#.T
    hex_list = ['#FF3333', '#FFFFFF', '#33FF49']
    blah = get_continuous_cmap(hex_list)
    blah.set_bad(color=np.array([0,0,0,1]))

    # for i in range(250):
    #     alpha[i, i] = 0
    a_file = open("traitements2/kept_dimensions.pkl", "rb")
    estimated_mask = pickle.load(a_file)
    a_file.close()

    print(alpha[estimated_mask[0], :][:, estimated_mask[0]].shape)

    heatmap = alpha[estimated_mask[0], :][:, estimated_mask[0]]

    n = np.sum(estimated_mask[0])
    G = nx.DiGraph()
    G.add_nodes_from(range(1, n+1))

    print(heatmap.shape, n)
    for i in range(n):
        for j in range(n):
            aux = heatmap[i, j]
            if aux != 0.0 and i!=j:
                G.add_edge(i+1, j+1)

    pos = nx.circular_layout(G)
    nodes = nx.draw_networkx_nodes(G, pos,node_size=1.5, node_color="indigo")
    edges = nx.draw_networkx_edges(
        G,
        pos,
        arrowstyle="->",
        arrowsize=5,
        width=0.5,
        alpha=0.3
    )

    # set alpha value for each edge

    #c = nx.clustering(G)

    #print(c)

    ax = plt.gca()
    ax.set_axis_off()
    # plt.show()
    clf2=Newman2CommunityClassifier(G)
    clf2.fit()
    plot_communities(G,clf2)
    plt.show()
