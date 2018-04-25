from algorithm_realization.stn_reconstruction_lib import *


if __name__ == "__main__":
    demo_graph, node_number, edge_number = open_file_data_graph("../data/apollonian.txt")

    # demo_graph, node_number, edge_number = demo_graph_generator()
    print("graph || nodes:", node_number, "; edges:", edge_number)
    # generate data arrival times(dats), also dat_path.
    # set dat_number to control the total number of dat.
    dats, dat_path = dats_generator(demo_graph,mode="gaussian",dat_number=40, seed=True)
    discrete_wtd, discrete_mass = continuous_func_distribution2discrete()
    adj_mat = faster_topology_reconstruction_through_dats_based_on_wtd1(dats, discrete_wtd)
    edge_count = count_in_matrix(adj_mat)/2
    t = f = 0
    for edge in demo_graph.edges():
        if adj_mat[edge[0]][edge[1]] == 1:
            t += 1
        else:f += 1
    print("accuracy1: ",t/(t+f),"accuracy2:",edge_number/edge_count)
    draw_graph(demo_graph)
