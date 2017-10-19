#! usr/bin/env python
# -*- coding: utf-8 -*-
# @aquamarine gy 2017.10.7


import networkx as nx
import matplotlib.pyplot as plt
import random
import numpy as np


# 生成随机数
# plt.hist(random_seed, bins=30,normed=True)
# plt.show()

# random_seed = np.random.gamma(2,3,1000)
# print(random_seed)


def topology_generating_type_tree(initial_graph):  # 生成拓扑结构
    try:
        initial_graph.add_nodes_from([i for i in range(0, 20)])
        initial_graph.add_weighted_edges_from([[0, 1, 0.0], [0, 2, 0.0], [0, 3, 0.0], [0, 4, 0.0]])
        initial_graph.add_weighted_edges_from([[1, 5, 0.0], [1, 6, 0.0], [1, 7, 0.0], [1, 8, 0.0]])
        initial_graph.add_weighted_edges_from([[2, 5, 0.0], [2, 6, 0.0], [2, 7, 0.0], [2, 8, 0.0]])
        initial_graph.add_weighted_edges_from([[3, 5, 0.0], [3, 6, 0.0], [3, 7, 0.0], [3, 8, 0.0]])
        initial_graph.add_weighted_edges_from([[4, 5, 0.0], [4, 6, 0.0], [4, 7, 0.0], [4, 8, 0.0]])
        initial_graph.add_weighted_edges_from(
            [[5, 9, 0], [5, 10, 0], [5, 11, 0], [5, 12, 0], [5, 13, 0], [5, 14, 0], [5, 15, 0]])
        initial_graph.add_weighted_edges_from(
            [[6, 9, 0], [6, 10, 0], [6, 11, 0], [6, 12, 0], [6, 13, 0], [6, 14, 0], [6, 15, 0]])
        initial_graph.add_weighted_edges_from(
            [[7, 9, 0], [7, 10, 0], [7, 11, 0], [7, 12, 0], [7, 13, 0], [7, 14, 0], [7, 15, 0]])
        initial_graph.add_weighted_edges_from(
            [[8, 9, 0], [8, 10, 0], [8, 11, 0], [8, 12, 0], [8, 13, 0], [8, 14, 0], [8, 15, 0]])
        initial_graph.add_weighted_edges_from([[9, 16, 0], [9, 17, 0], [9, 18, 0], [9, 19, 0]])
        initial_graph.add_weighted_edges_from([[10, 16, 0], [10, 17, 0], [10, 18, 0], [10, 19, 0]])
        initial_graph.add_weighted_edges_from([[11, 16, 0], [11, 17, 0], [11, 18, 0], [11, 19, 0]])
        initial_graph.add_weighted_edges_from([[12, 16, 0], [12, 17, 0], [12, 18, 0], [12, 19, 0]])
        initial_graph.add_weighted_edges_from([[13, 16, 0], [13, 17, 0], [13, 18, 0], [13, 19, 0]])
        initial_graph.add_weighted_edges_from([[14, 16, 0], [14, 17, 0], [14, 18, 0], [14, 19, 0]])
        initial_graph.add_weighted_edges_from([[15, 16, 0], [15, 17, 0], [15, 18, 0], [15, 19, 0]])
    except:
        return 0
    return 1


def topology_generator(type=0):  # 生成不同的网络模型
    if type == 0:
        BA = nx.random_graphs.barabasi_albert_graph(100, 3)
        return BA
    if type == 1:
        GNP = nx.fast_gnp_random_graph(20, 0.5)
        return GNP
    if type == 2:
        ER = nx.erdos_renyi_graph(40, 0.1)
        return ER


def draw_graph(graph, type=0):  # 选择生成的图像的拓扑形状
    if type == 0:
        pos = nx.spring_layout(graph)
    if type == 1:
        pos = nx.random_layout(graph)
    if type == 2:
        pos = nx.shell_layout(graph)
    plt.subplot(132)
    nx.draw_networkx(graph, pos, with_labels=True, node_size=4)


def weighted_graph_generator(graph, sample_graph):  # 将之前生成的网络模型转化成有向有权图
    for edge_tuple in sample_graph.edges():
        graph.add_weighted_edges_from([(edge_tuple[0], edge_tuple[1], np.random.gamma(2, 3, 1)[0])])
    return graph  # 返回生成好的有向有权图


def calculate_diffusive_arrival_times(graph, diffusive_source=0):
    DAT_PATH = nx.shortest_path(G, source=diffusive_source, weight="weight")
    DAT = nx.shortest_path_length(G, source=diffusive_source, weight="weight")
    print(DAT)
    print(DAT_PATH)

    return DAT_PATH, DAT  # 返回计算出的扩散到达路径和扩散到达时间


def draw_diffusion_tree(diffusive_arrival_times, type=0):
    Diffusion_Tree = nx.DiGraph()
    for node, value in diffusive_arrival_times.items():
        for i in range(0, len(value) - 1):
            Diffusion_Tree.add_weighted_edges_from([[value[i], value[i + 1], 0]])
    if type == 0:
        pos = nx.spring_layout(Diffusion_Tree)
    if type == 1:
        pos = nx.random_layout(Diffusion_Tree)
    if type == 2:
        pos = nx.shell_layout(Diffusion_Tree)
    plt.subplot(133)
    nx.draw_networkx(Diffusion_Tree, pos, with_labels=True, node_color='blue', node_size=4)  # 按参数构图
    return Diffusion_Tree


diffusive_source = 7
sample_graph = topology_generator(2)  # 生成一个无向无权的网络样本
plt.subplot(131)
nx.draw_networkx(sample_graph, pos=nx.spring_layout(sample_graph), node_size=4, node_color="black", with_labels=False)
G = nx.Graph()  # 初始化
G = weighted_graph_generator(G, sample_graph)  # 将样本重构成有向有权图
draw_graph(G, type=0)
DAT_PATH, DAT = calculate_diffusive_arrival_times(G, diffusive_source)
# 扩散过程的可视化
draw_diffusion_tree(DAT_PATH)  # 绘制扩散路径图
plt.title("diffusive source: %d" % diffusive_source)
plt.show()
