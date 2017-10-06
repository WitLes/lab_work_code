import networkx as nx
import matplotlib.pyplot as plt
from random import randint
import numpy as np


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


def topology_generator(type=0):
    if type == 0:
        BA = nx.random_graphs.barabasi_albert_graph(20, 2)
        return BA
    if type == 1:
        GNP = nx.fast_gnp_random_graph(20, 0.5)
        return GNP
    if type == 2:
        pass


def draw_graph(graph, type=0):
    if type == 0:
        pos = nx.spring_layout(graph)
    if type == 1:
        pos = nx.random_layout(graph)
    if type == 2:
        pos = nx.shell_layout(graph)
    nx.draw_networkx(graph, pos, with_labels=True, node_size=30)
    plt.show()


def directed_graph_generator(graph, sample_graph):
    for edge_tuple in sample_graph.edges():
        graph.add_weighted_edges_from([(edge_tuple[0], edge_tuple[1], np.random.gamma(2, 3, 1)[0])])
    return graph


# 确定每个节点在图像中的位置
pos_for_tree = {0: [1, 3.5], 1: [2, 2], 2: [2, 3], 3: [2, 4], 4: [2, 5], 5: [3, 2], 6: [3, 2.7], 7: [3, 3.6],
                8: [3, 4.4], 9: [4, 1], 10: [4, 2], 11: [4, 3], 12: [4, 4], 13: [4, 5], 14: [4, 6], 15: [4, 7],
                16: [5, 1], 17: [5, 2], 18: [5, 3], 19: [5, 4], 20: [5, 5]}

# 生成有向图
G = nx.DiGraph()
topology_generating_type_tree(G)
# infected_flag = [0 for i in range(0,20)]

# 每个连边上都取gamma分布的抽样
for i, j, k in G.edges(data=True):
    G.add_weighted_edges_from([[i, j, np.random.gamma(2, 3, 1)[0]]])

# 计算单元最短路径和路径长度
DAT_PATH = nx.shortest_path(G, source=0, weight="weight")
DAT = nx.shortest_path_length(G, source=0, weight="weight")
print(DAT)
print(DAT_PATH)
"""
for (i,j,k) in G.edges(data=True):
	print(i,"->",j,": ",k)
"""

nx.draw_networkx(G, pos=pos_for_tree, with_labels=True, node_color='red')  # 按参数构图
plt.axis('off')
plt.show()  # 显示图像

# 扩散过程的可视化
Diffusion_Tree = nx.DiGraph()
for node, value in DAT_PATH.items():
    print(value)
    for i in range(0, len(value) - 1):
        Diffusion_Tree.add_weighted_edges_from([[value[i], value[i + 1], 0]])

nx.draw_networkx(Diffusion_Tree, pos=pos_for_tree, with_labels=True, node_color='blue')  # 按参数构图
plt.axis('off')
plt.show()  # 显示图像
