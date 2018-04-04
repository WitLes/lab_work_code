from algorithm_realization.stn_reconstruction_lib import *

if __name__ == "__main__":
    # generate the graph topology, return the graph, the number of nodes, the number of edges.
    #demo_graph, node_number, edge_number = er_graph_generator(80, 0.1, seed=0, directed=True)
    demo_graph, node_number, edge_number = open_file_data_graph("data/smallworld.txt")
    #demo_graph, node_number, edge_number = demo_graph_generator()
    print("graph || nodes:", node_number, "; edges:", edge_number)

    # generate data arrival times(dats), also dat_path.
    # set dat_number to control the total number of dat.
    dats, dat_path = dats_generator(demo_graph, dat_number=80, seed=False)
    discrete_wtd, discrete_mass = continuous_func_distribution2discrete()

    graph_samples = mcmc_algorithm_with_gibbs_sampling(demo_graph, demo_graph, discrete_wtd, dats, delta=0.01, l_t=5)
