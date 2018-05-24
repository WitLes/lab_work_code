import graph_tool.all as gt
from algorithm_realization.stn_reconstruction_lib import *

# create networkx graph
file_name = ["data/scalefree.txt","data/polbooks.txt","data/football.txt","data/apollonian.txt","data/dolphins.txt","data/karate.txt","data/lattice2d.txt","data/miserables.txt","data/pseudofractal.txt","data/randomgraph.txt","data/scalefree.txt","data/sierpinski.txt","data/smallworld.txt","data/jazz.txt"]
file_name = ["../"+string for string in file_name]


color_list = [[0.64,0.16,0.16,1],[0.41,0.55,0.13,1],[0.06,0.3,0.54,1]]
mode_list = ["BA","SW","ER"]
color_index = 0
mode_index = 2
mode_select = 6
#demo_graph, node_number, edge_number = open_file_data_graph(file_name[mode_select])
demo_graph, node_number, edge_number = networkx_graph_generator(mode=mode_list[mode_index])

# create graph_tool graph
original_graph = gt.Graph(directed=False)
for edge in demo_graph.edges():
    original_graph.add_edge(edge[0],edge[1])
pos1 = gt.fruchterman_reingold_layout(original_graph)
#pos1 = gt.sfdp_layout(original_graph)
gt.graph_draw(original_graph,pos=pos1,vprops={"size":5,"fill_color":color_list[color_index]}, eprops={"color":[0.5, 0.5,0.5, 0.5],"marker_size":4},output_size=(400, 400), output="out.eps")
gt.graph_draw(original_graph,pos=pos1,vprops={"size":5,"fill_color":color_list[color_index]}, eprops={"color":[0.5, 0.5,0.5, 0.5],"marker_size":4},output_size=(400, 400), output="out.png")
gt.graph_draw(original_graph,pos=pos1,vprops={"size":5,"fill_color":color_list[color_index]}, eprops={"color":[0.5, 0.5,0.5, 0.5],"marker_size":4},output_size=(400, 400), output="out.svg")
