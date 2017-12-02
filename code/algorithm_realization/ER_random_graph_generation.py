import numpy as np
import networkx as nx
import random 
import time
import matplotlib.pyplot as plt
import sympy as sp
from sympy.abc import a,x,y


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

def find_possible_edges_from_dats(dats):
	node_list_sum = list()
	node_list = list()
	edges_set = set()
	# select the first dat to select all the possile edges.Then use all the residual dats to restrict the possible edges.:
	for i in range(len(dats)):
		sorted_dat = sorted(dats[i].items(),key=lambda x:x[1])
		for (node, time) in sorted_dat:
			node_list.append(node)
		temp_edges = find_ordered_tuples_in_list(node_list)
		node_list = []
		for edge in temp_edges:
			edges_set.add(edge)

	return list(edges_set)


def generate_all_edges(graph):
	# IMPORTANT!! THE input graph must be a networkx graph.
	nodes = list()
	number = len(nodes)
	edges_list = []
	for node in graph.nodes():
		nodes.append(node)
	for i in range(number):
		for j in range(number):
			if i != j:
				edges_list.append((i,j))
	return edges_list




def find_ordered_tuples_in_list(list_in):
	length = len(list_in)
	possible_edges = list()
	for i in range(length):
		for j in range(i+1, length):
			possible_edges.append((list_in[i],list_in[j]))
	return possible_edges

def mcmc_algorithm_with_gibbs_sampling(graph,discrete_wtd,dats):
	edges = generate_all_edges(graph)
	burn_in = 10
	max_lag = 10
	survival_f = pdf2sf(discrete_wtd)
	hazard_f = pdf_sf2hazard(discrete_wtd, survival_f)
	M_sample_size = 30
	graph_samples = list()
	survival_func = pdf2sf(discrete_wtd)
	hazard_func = discrete_wtd / survival_func
	lambda_dict_list = [{} for i in range(len(dats))]
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
		if lag == max_lag:
			graph_samples.append(nx.Graph(graph))
			lag = 0

	return graph_samples


def pdf2sf(discrete_wtd):
	sum_mass = sum(discrete_wtd)
	sf = list()
	sf.append(sum_mass)
	for i in range(1,len(discrete_wtd)):
		point_value = sum_mass - sum(discrete_wtd[0:i])
		sf.append(point_value)
	return sf


def pdf_sf2hazard(discrete_wtd,survival_function):
	hazard_function = []
	for i in range(len(discrete_wtd)):
		hazard_function.append(discrete_wtd[i]/survival_function[i])
	return hazard_function


def continuous_exp_distribution2discrete(delta,l_t,miu=1):
	discrete_wtd = list()
	discrete_mass = list()
	discrete_wtd.append(0)
	discrete_mass.append(0)
	left = float(0)
	right = float(delta)
	for i in range(1, int(l_t/delta)):
		temp = sp.integrate(sp.exp(-miu*x), (x,left,right))
		discrete_mass.append(temp)
		discrete_wtd.append(discrete_mass[i] / delta)
		left = right
		right += delta
	# if there is something needed, it must be the normalization step, which makes the sum of discrete_mass equals to 1.

	# discrete_wtd is a list of the discrete point of WTD.for example, list[1] = exp(delta), list[n] = exp(delta * n)
	# discrete_mass is the list of mass of each small section.
	return discrete_wtd, discrete_mass


def scatter_wtd(f):
	plt.scatter([i for i in range(len(f))], f)
	plt.show()


er_graph, node_number, edge_number = er_graph_generator(100,0.06,seed=0,directed=False)
generate_all_edges(er_graph)
dats, dat_path = dats_generator(er_graph,dat_number=10,seed=True)

'''
discrete_wtd,discrete_mass = continuous_exp_distribution2discrete(0.1,5)
survival_f = pdf2sf(discrete_wtd)
hazard_f = pdf_sf2hazard(discrete_wtd, survival_f)
scatter_wtd(hazard_f)


'''
