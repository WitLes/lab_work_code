from algorithm_realization.stn_reconstruction_lib import *

if __name__ == "__main__":
    # er_graph, node_number, edge_number = er_graph_generator(100, 0.06, seed=0, directed=True)
    demo_graph, node_number, edge_number = open_file_data_graph("../data/scalefree.txt")
    #demo_graph, node_number, edge_number = demo_graph_generator()
    print("graph || nodes:", node_number, "; edges:", edge_number)
    # generate data arrival times(dats), also dat_path.
    # set dat_number to control the total number of dat.
    dats, dat_path = dats_generator(demo_graph,mode="uniform", dat_number=node_number, seed=False)
    #print(dats)
    initial_wtd = init_discrete_wtd(dats)
    scatter_wtd(initial_wtd)
    wtd = wtd_estimation(demo_graph, dats ,initial_wtd, delta=0.01, l_t=5)
    scatter_wtd(wtd)
    #pdf_cvr = pdf_recover(dats,0,20)
    #scatter_wtd(pdf_cvr)