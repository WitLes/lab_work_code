from algorithm_realization.stn_reconstruction_lib import *

demo_graph, node_number, edge_number = networkx_graph_generator(mode="ER")
node_list = []
for node in demo_graph.nodes():
    node_list.append(node)

print(list(sorted(node_list)))