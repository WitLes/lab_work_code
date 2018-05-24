from algorithm_realization.stn_reconstruction_lib import *


node_number = 1500
link_probability = 0.005
pop_percent = 7
demo_graph, node_number, edge_number = er_graph_generator(node_number=node_number,link_probability=link_probability)

print("ER  ", "graph || nodes:", node_number, "; edges:", edge_number)
print("average shortest path length: ", nx.average_shortest_path_length(demo_graph))
print("average clustering coef: ", nx.average_clustering(demo_graph))

def large_scale_network_reconstruction_for_script(node_number,link_probability,C,cutting_index):

    distribution_mode = "gaussian"
    dats, dat_path = dats_generator(demo_graph, mode=distribution_mode, dat_number=int(C * node_number), seed=False)
    discrete_wtd = pdf_generator(mode=distribution_mode)
    adj_mat = faster_topology_reconstruction_through_dats_based_on_wtd2(dats, discrete_wtd,cutting_index=cutting_index)
    tp_fp = count_in_matrix(adj_mat) / 2
    TP = FN = FP = TN = 0
    edge_total = node_number * (node_number - 1) / 2

    for edge in demo_graph.edges():
        if adj_mat[edge[0]][edge[1]] == 1:
            TP += 1
        else:
            FN += 1
    FP = tp_fp - TP
    TN = edge_total - TP - FN - FP
    TPR = TP / (TP + FN)
    FPR = FP / (FP + TN)
    Precision = TP / (TP + FP)
    Recall = TP / (TP + FN)
    F1 = 2*Precision*Recall/(Precision + Recall)
    return F1


c_list = [float(i/50) for i in range(50)]
final_c_list = []
for i in range(1):
    c_list.pop(0)

cutting_index_list = [28,29,30,31]
for k in range(10):
    print("\n")
    final_c = 0
    for c in c_list:
        f1_list = []
        print(".",end='',flush=True)
        for cutting in cutting_index_list:
            f1_list.append(large_scale_network_reconstruction_for_script(node_number=node_number,link_probability=link_probability,C=c,cutting_index=cutting))
        if max(f1_list) > 0.95:
            final_c = c
            break
    final_c_list.append(final_c)

print(final_c_list,sum(final_c_list)/len(final_c_list))