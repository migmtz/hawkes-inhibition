import numpy as np
import csv
import pickle
import seaborn as sns
from matplotlib import pyplot as plt
from dictionary_parameters import dictionary as param_dict
import networkx as nx

from networkx.convert_matrix import to_numpy_matrix
from numpy.linalg import eig
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
        networkx.draw_networkx_labels(G, pos, labels=None)

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
    number = 7
    theta = param_dict[number]
    dim = int(np.sqrt(1 + theta.shape[0]) - 1)
    mu = theta[:dim]
    alpha = theta[dim:-dim].reshape((dim, dim))
    beta = theta[-dim:]

    #print(alpha[estimated_mask[0], :][:, estimated_mask[0]].shape)

    n = dim
    fig, ax = plt.subplots(figsize=(8,8))
    G = nx.DiGraph()
    G.add_nodes_from(range(1, n+1))

    #print(heatmap.shape, n)
    for i in range(n):
        for j in range(n):
            aux = alpha[i, j]
            G.add_edge(i+1, j+1, weight=np.abs(aux), intensity=aux)

    np.random.seed(5)
    pos = nx.kamada_kawai_layout(G)
    pos = {1: np.array([0, 0]), 2: np.array([1, 0]), 3: np.array([0.5, -0.5]), 4: np.array([0, -1]), 5: np.array([1, -1]), 6: np.array([0.5, -1.5]), 7: np.array([0, -2]), 8: np.array([1, -2]), 9: np.array([0.5, -2.5]), 10: np.array([0.5, -3])}
    #nx.draw(G, with_labels=True)
    np.random.seed(5)
    sns.set_theme()
    #only = np.random.choice(range(1, 224), size=3, replace=False)
    only = range(1,224)
    print(only)
    pos_self = [i for i,j in G.edges() if G[i][j]["intensity"] >= 0 and i == j and i in only]
    neg_self = [i for i,j in G.edges() if G[i][j]["intensity"] < 0 and i == j and i in only]
    positive_strong = [(i, j) for i,j in G.edges() if G[i][j]["intensity"] > 0 and G[i][j]["intensity"]/beta[i-1] > 0.0 and i!= j and (i in only or j in only)]
    negative_strong = [(i, j) for i, j in G.edges() if G[i][j]["intensity"] < 0 and G[i][j]["intensity"]/beta[i-1] < -0.0 and i!= j and (i in only or j in only)]
    positive_weak = [(i, j) for i, j in G.edges() if
                       G[i][j]["intensity"] > 0 and G[i][j]["intensity"] / beta[i - 1] < 0.25 and i != j and (
                                   i in only or j in only)]
    negative_weak = [(i, j) for i, j in G.edges() if
                       G[i][j]["intensity"] < 0 and G[i][j]["intensity"] / beta[i - 1] > -0.25 and i != j and (
                                   i in only or j in only)]
    others = [(i, j) for i, j in G.edges() if G[i][j]["intensity"] < 0 and i != j and (i not in only and j not in only)]
    nodes_pos = nx.draw_networkx_nodes(G, pos, node_size=500, nodelist=pos_self, node_color="g", edgecolors="k", ax=ax)
    nodes_pos = nx.draw_networkx_labels(G, pos, ax=ax, font_size=8, font_color="w")
    nodes_neg = nx.draw_networkx_nodes(G, pos, node_size=500, nodelist=neg_self, node_color="r", edgecolors="k", ax=ax)
    weak_edges_pos = nx.draw_networkx_edges(G, pos, width=1, node_size=100, style=":", edgelist=positive_weak, edge_color="g",
                                              arrowsize=3, alpha=0.25, ax=ax)
    weak_edges_neg = nx.draw_networkx_edges(G, pos, width=1, node_size=100, style=":", edgelist=negative_weak, edge_color="r",
                                              arrowsize=3, alpha=0.25, ax=ax)
    strong_edges_pos = nx.draw_networkx_edges(G, pos, width=1.3, node_size=100, edgelist=positive_strong, edge_color="k",
                                              arrowsize=3, alpha=1, ax=ax)
    strong_edges_pos = nx.draw_networkx_edges(G, pos, width=1, node_size=100, edgelist=positive_strong, edge_color="g",
                                              arrowsize=3, alpha=1, ax=ax)
    strong_edges_neg = nx.draw_networkx_edges(G, pos, width=1.3, node_size=100, edgelist=negative_strong, edge_color="k",
                                              arrowsize=3, alpha=1, ax=ax)
    strong_edges_neg = nx.draw_networkx_edges(G, pos, width=1, node_size=100, edgelist=negative_strong, edge_color="r",
                                              arrowsize=3, alpha=1, ax=ax)
    extra = nx.draw_networkx_edges(G, pos, width=1, node_size=100, edgelist=others, edge_color="k", arrowsize=3,
                                   alpha=0.01, ax=ax)

    #plt.savefig("communities_d10.pdf", format="pdf")

    plt.show()
