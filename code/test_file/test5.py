from algorithm_realization.stn_reconstruction_lib import *


demo_graph, node_number, edge_number = open_file_data_graph("data/apollonian.txt")
#demo_graph, node_number, edge_number = er_graph_generator(node_number=100,link_probability=0.06,seed=None,directed=False)
print("  graph || nodes:", node_number, "; edges:", edge_number)
print("average shortest path length: ", nx.average_shortest_path_length(demo_graph))
print("average clustering coef: ", nx.average_clustering(demo_graph))
# generate data arrival times(dats), also dat_path.
# set dat_number to control the total number of dat.
dats, dat_path = dats_generator(demo_graph, dat_number=int(node_number), seed=False)
discrete_wtd, discrete_mass = continuous_func_distribution2discrete()
adj_mat = faster_topology_reconstruction_through_dats_based_on_wtd2(dats, discrete_wtd)
tp_fp = count_in_matrix(adj_mat) / 2
TP = FN = FP = TN = 0
edge_total = node_number*(node_number-1)/2

for edge in demo_graph.edges():
    if adj_mat[edge[0]][edge[1]] == 1:
        TP += 1
    else:
        FN += 1
FP = tp_fp - TP
TN = edge_total - TP - FN - FP
print("TPR: ", TP / (TP+FN), "FPR:", FP / (FP+TN))
print("Precision: ",TP/(TP+FP), "Recall: ",TP/(TP+FN))
print("------------------------------------------------")