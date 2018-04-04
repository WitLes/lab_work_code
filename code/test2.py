import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def demo_graph_generator():
    # undirected graph uses nx.Graph()
    # directed graph uses nx.Digraph()
    graph = nx.Graph()
    graph.add_edges_from([[0, 1, {"weight": 1}],
                          [0, 2, {"weight": 1}],
                          [0, 4, {"weight": 1}],
                          [0, 5, {"weight": 1}],
                          [0, 7, {"weight": 1}],
                          [1, 4, {"weight": 1}],
                          [1, 5, {"weight": 1}],
                          [1, 6, {"weight": 1}],
                          [2, 3, {"weight": 1}],
                          [2, 6, {"weight": 1}],
                          [3, 4, {"weight": 1}],
                          [3, 6, {"weight": 1}],
                          [4, 5, {"weight": 1}],
                          [4, 6, {"weight": 1}],
                          [5, 6, {"weight": 1}],
                          [5, 7, {"weight": 1}]])
    node_number = len(graph.nodes())
    edge_number = len(graph.edges())
    return graph, node_number, edge_number


def continuous_func_distribution2discrete(delta=0.01, l_t=5):
    # gauss distribution by default
    miu = 2
    sigma = 0.4
    discrete_wtd = list()
    discrete_mass = list()
    discrete_number = int(l_t / delta)
    for i in range(discrete_number):
        # here to change the formula of the distribution
        discrete_wtd.append(1 / np.sqrt(2 * np.pi) / sigma * np.exp(-((delta * i - miu) ** 2) / (2 * (sigma ** 2))))
        discrete_mass.append(discrete_wtd[i] * delta)

    # if there is something still needed, it must be the normalization step, which makes the sum of discrete_mass equals to 1.

    # discrete_wtd is a list of the discrete point of WTD.for example, list[1] = exp(delta), list[n] = exp(delta * n)
    # discrete_mass is the list of mass of each small section.
    # we return both of them for the convenience of further calculation.

    # the x-axis begins from 1, that is {1,2,...,l_t}, but note that the index of the "list" starts from 0.
    return discrete_wtd, discrete_mass

def init_f(delta=0.01, l_t=5):
    miu = 0.5
    # gauss distribution by default
    discrete_wtd = list()
    discrete_mass = list()
    discrete_number = int(l_t / delta)
    for i in range(discrete_number):
        # here to change the formula of the distribution
        discrete_wtd.append(1/miu * np.exp(-i*delta/miu) + np.random.random()*0.1)
        discrete_mass.append(discrete_wtd[i] * delta)

    # if there is something still needed, it must be the normalization step, which makes the sum of discrete_mass equals to 1.

    # discrete_wtd is a list of the discrete point of WTD.for example, list[1] = exp(delta), list[n] = exp(delta * n)
    # discrete_mass is the list of mass of each small section.
    # we return both of them for the convenience of further calculation.

    # the x-axis begins from 1, that is {1,2,...,l_t}, but note that the index of the "list" starts from 0.
    return discrete_wtd, discrete_mass

def init_kernel_function(h, l_t=5, delta=0.01):
    # gauss distribution by default
    miu = (l_t - 0) / 2
    sigma = h
    discrete_wtd = list()
    discrete_mass = list()
    discrete_number = int(l_t / delta)
    for i in range(discrete_number):
        # here to change the formula of the distribution
        discrete_wtd.append(1 / np.sqrt(2 * np.pi) / sigma * np.exp(-((delta * i - miu) ** 2) / (2 * (sigma ** 2))))
        discrete_mass.append(discrete_wtd[i] * delta)

    # if there is something still needed, it must be the normalization step, which makes the sum of discrete_mass equals to 1.

    # discrete_wtd is a list of the discrete point of WTD.for example, list[1] = exp(delta), list[n] = exp(delta * n)
    # discrete_mass is the list of mass of each small section.
    # we return both of them for the convenience of further calculation.

    # the x-axis begins from 1, that is {1,2,...,l_t}, but note that the index of the "list" starts from 0.
    return discrete_wtd, discrete_mass


def scatter_wtd(f):
    plt.scatter([i for i in range(len(f))], f)
    plt.show()

def normalize(f, delta=0.01):
    sum_f = sum(f) * delta
    f = [x/sum_f for x in f]
    return f

gaussian,pmf = init_kernel_function(0.02)
scatter_wtd(gaussian)

exp,exp_mass = init_f()
scatter_wtd(exp)

for i in range(100):
    exp = np.convolve(exp,gaussian,mode="same")

    scatter_wtd(exp)


