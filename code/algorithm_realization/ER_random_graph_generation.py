import numpy as np
import networkx as nx
import random 
import time
import matplotlib.pyplot as plt


def er_graph_generator(node_number=100, link_probability=0.06,seed=None, directed=False):
	# this function returns a ER random graph with input node number and edge link probability.
	# "seed" parameter is the random seed.Fixing it will let your graph fixed every time you run the function.
	# this funtion calls a networkx funtion "nx.erdos_renyi_graph()"
	er_graph = nx.erdos_renyi_graph(node_number, link_probability, seed, directed)
	weighted_er_graph = nx.Graph()
	for edge in er_graph.edges():
		weighted_er_graph.add_edge(edge[0],edge[1],weight=1)
	node_number = len(weighted_er_graph.nodes())
	edge_number = len(weighted_er_graph.edges())
	return weighted_er_graph,node_number,edge_number

def DAT_generator(graph_topology,wtd_array,diffusive_source):
	# This function returns the node arrival times of the (topology,waiting time distribution) by sampling from a gauss distribution.
	# DAT is a dict{node:time}.DAT_PATH is a dict{node:[node1,node1,node]}
	length = len(graph_topology.edges())
	i = 0
	for edge in graph_topology.edges(data=True):
		graph_topology[edge[0]][edge[1]]["weight"] = wtd_array[i]
		i += 1
	#print(graph_topology.edges(data=True))
	DAT_PATH = nx.shortest_path(graph_topology, source=diffusive_source, weight="weight")
	DAT = nx.shortest_path_length(graph_topology, source=diffusive_source, weight="weight")
	return DAT, DAT_PATH

def dats_generator(graph_topology,dat_number=1,seed=True):
	# This function returns the list of DATs.
	# dat_number is the number of DAT that you want.
	# You can set seed=True if you want the dats is always fixed whenever you run the sampling code.
	# If you want to change the distribution of waiting times(WTD). You can change the function used in the codes.
	#	|-->"wtd_array = np.random.normal(5, 1, length)"<--|
	length = len(graph_topology.edges())
	dat = []
	dat_path = []
	for i in range(dat_number):
		if seed == True:
			np.random.seed(i)
		diffusive_source = np.random.randint(100)
		# WTD can be changed here.
		wtd_array = np.random.normal(5, 1, length)

		dat_temp,dat_path_temp = DAT_generator(graph_topology,wtd_array,diffusive_source)
		dat.append(dat_temp)
		dat_path.append(dat_path_temp)
		#print(dat_path_temp)
	return dat, dat_path


def draw_diffusion_tree(diffusive_arrival_times, type=0):
    Diffusion_Tree = nx.DiGraph()
    for node, value in diffusive_arrival_times.items():
        for i in range(0, len(value) - 1):
            Diffusion_Tree.add_weighted_edges_from([[value[i], value[i + 1], 0]])
    if type == 0:
        pos = nx.spring_layout(Diffusion_Tree)
    if type == 1:
        pos = nx.random_layout(Diffusion_Tree)
    if type == 2:
        pos = nx.shell_layout(Diffusion_Tree)
    plt.subplot(133)
    nx.draw_networkx(Diffusion_Tree, pos, with_labels=True, node_color='blue', node_size=4)  # 按参数构图
    return Diffusion_Tree


def mcmc_algorithm_with_gibbs_sampling(discrete_wtd,dats):
	burn_in = 10
	max_lag = 10
	M_sample_size = 30
	graph_samples = list()
	survival_func = pdf2sf(discrete_wtd)
	hazard_func = discrete_wtd / survival_func

	graph = nx.Graph(directed=True)
	iteration = 0
	graph_sample_number = 0
	lag = 0
	while graph_sample_number < 30:
		iteration += 1
		for edge in edges:
			if edge in graph.edges():
				pass
			else:
				pass
		if iteration > burn_in:
			lag += 1
		if lag = max_lag:
			graph_samples.append(graph)		
			lag = 0	

	return graph_samples

def pdf2sf():
	pass
er_graph, node_number, edge_number = er_graph_generator(100,0.06,seed=0,directed=False)

dats,dat_path = dats_generator(er_graph,dat_number=1,seed=True)
print(er_graph.edges())



