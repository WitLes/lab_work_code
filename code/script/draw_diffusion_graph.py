import graph_tool.all as gt
from algorithm_realization.stn_reconstruction_lib import *

# create networkx graph

color_list = [[0.64,0.16,0.16,1],[0.41,0.55,0.13,1],[0.06,0.3,0.54,1]]
mode_list = ["BA","SW","ER"]
color_index = 0
mode_index = 0
er_graph, node_number, edge_number = networkx_graph_generator(mode=mode_list[mode_index])
dat, dat_path = dats_generator(graph_topology=er_graph, mode="gaussian", dat_number=1, seed=True)
dat = dat[0]
dat_path = dat_path[0]

# create graph_tool graph
original_graph = gt.Graph(directed=False)
for edge in er_graph.edges():
    original_graph.add_edge(edge[0],edge[1])
pos1 = gt.fruchterman_reingold_layout(original_graph)
#pos1 = gt.sfdp_layout(original_graph)
gt.graph_draw(original_graph,pos=pos1,vprops={"size":5,"fill_color":color_list[color_index]}, eprops={"color":[0.5, 0.5,0.5, 0.5],"marker_size":4},output_size=(400, 400), output="ER.eps")
count = 0
for i in pos1:
    count += 1

diffusion_tree = gt.Graph(directed=True)
edge_set = set()
for source_node,node_list in dat_path.items():
    for i in range(len(node_list)-1):
        edge_set.add((node_list[i],node_list[i+1]))
for edge in edge_set:
    diffusion_tree.add_edge(edge[0],edge[1])
pos2 = gt.fruchterman_reingold_layout(diffusion_tree)
#pos2 = gt.sfdp_layout(diffusion_tree)
for i in range(count):
    pos2[i] = pos1[i]
gt.graph_draw(diffusion_tree,pos=pos2, vprops={"size":5,"fill_color":color_list[color_index]},eprops={"color":[0.5, 0.5,0.5, 0.5],"marker_size":4},output_size=(400, 400), output="diffusion.eps")

pos3 = gt.radial_tree_layout(diffusion_tree, diffusion_tree.vertex(dat[0]))
gt.graph_draw(diffusion_tree,pos=pos3, vprops={"size":5,"fill_color":color_list[color_index]},eprops={"color":[0.5, 0.5,0.5, 0.5],"marker_size":4},output_size=(400, 400), output="diffusion_ra.eps")