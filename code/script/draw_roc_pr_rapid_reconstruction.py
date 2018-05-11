from algorithm_realization.stn_reconstruction_lib import *


#file_name = ["data/scalefree.txt","data/polbooks.txt","data/football.txt","data/apollonian.txt","data/dolphins.txt","data/karate.txt","data/lattice2d.txt","data/miserables.txt","data/pseudofractal.txt","data/randomgraph.txt","data/scalefree.txt","data/sierpinski.txt","data/smallworld.txt","data/jazz.txt"]
mode_list = ["ER","BA","SW"]
distribution_list = ["gaussian","weibull","gumbel"]
mode_select = 0
distribution_select = 0

#demo_graph, node_number, edge_number = open_file_data_graph(name)
demo_graph, node_number, edge_number = networkx_graph_generator(mode=mode_list[mode_select])
# demo_graph, node_number, edge_number = demo_graph_generator()
print(mode_list[mode_select], "  graph || nodes:", node_number, "; edges:", edge_number)
print("average shortest path length: ", nx.average_shortest_path_length(demo_graph))
print("average clustering coef: ", nx.average_clustering(demo_graph))
# generate data arrival times(dats), also dat_path.
# set dat_number to control the total number of dat.
TPR_list = [[0 for j in range(10)]for i in range(50)]
FPR_list = [[0 for j in range(10)]for i in range(50)]
Precison_list = [[0 for j in range(10)]for i in range(50)]
Recall_list = [[0 for j in range(10)]for i in range(50)]

file = open("record.txt", "w")
for c in range(10):
    print(str(c),end=" ",flush=True)
    dats, dat_path = dats_generator(demo_graph, mode="gaussian",dat_number=int(0.3*node_number), seed=False)
    discrete_wtd = pdf_generator(mode="gaussian")

    for i in range(50):
        adj_mat = faster_topology_reconstruction_through_dats_based_on_wtd2(dats, discrete_wtd,cutting_index=i)
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
        if TP == 0:
            TPR_list[i][c]=0
            FPR_list[i][c]=0
            Precison_list[i][c]=0
            Recall_list[i][c]=0
        else:
            TPR = TP / (TP + FN)
            FPR = FP / (FP + TN)
            Precison = TP / (TP + FP)
            Recall = TP / (TP + FN)
            TPR_list[i][c] = TPR
            FPR_list[i][c] = FPR
            Precison_list[i][c] = Precison
            Recall_list[i][c] = Recall
print(TPR_list)
file.close()