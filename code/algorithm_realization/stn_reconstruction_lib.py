import numpy as np
import networkx as nx
import random
import time
import matplotlib.pyplot as plt
import math


def er_graph_generator(node_number=100, link_probability=0.06, seed=None, directed=False):
    # this function returns a ER random graph with input node number and edge link probability.
    # "seed" parameter is the random seed.Fixing it will let your graph fixed every time you run the function.
    # this funtion calls a networkx funtion "nx.erdos_renyi_graph()"

    er_graph = nx.erdos_renyi_graph(node_number, link_probability, seed, directed)
    if not directed:
        weighted_er_graph = nx.Graph()
    elif directed:
        weighted_er_graph = nx.DiGraph()

    for edge in er_graph.edges():
        weighted_er_graph.add_edge(edge[0], edge[1], weight=1)
        # weighted_er_graph is used to generate the DATs in the following dat_generator() method which uses the Dijstra Algorithm to calculate the shortest distances of each nodes in the diffusion process.

    node_number = len(weighted_er_graph.nodes())
    edge_number = len(weighted_er_graph.edges())

    return weighted_er_graph, node_number, edge_number


def open_file_data_graph(file_path):
    with open(file_path, encoding="utf8") as data:
        data_list = []
        for line in data.readlines():
            line = line.strip("\n")
            line = line.split(" ")
            line = [int(item) for item in line]
            line.append({"weight": 1})
            data_list.append(line)
        data.close()
    graph = nx.Graph()
    graph.add_edges_from(data_list)
    node_number = graph.number_of_nodes()
    edge_number = graph.number_of_edges()
    return graph, node_number, edge_number


def demo_graph_generator():
    # undirected graph uses nx.Graph()
    # directed graph uses nx.Digraph()
    graph = nx.Graph()
    graph.add_edges_from([[0, 1, {"weight": 1}],
                          [0, 2, {"weight": 1}],
                          [0, 4, {"weight": 1}],
                          [0, 5, {"weight": 1}],
                          [0, 7, {"weight": 1}],
                          [1, 4, {"weight": 1}],
                          [1, 5, {"weight": 1}],
                          [1, 6, {"weight": 1}],
                          [2, 3, {"weight": 1}],
                          [2, 6, {"weight": 1}],
                          [3, 4, {"weight": 1}],
                          [3, 6, {"weight": 1}],
                          [4, 5, {"weight": 1}],
                          [4, 6, {"weight": 1}],
                          [5, 6, {"weight": 1}],
                          [5, 7, {"weight": 1}]])

    node_number = len(graph.nodes())
    edge_number = len(graph.edges())
    return graph, node_number, edge_number


def draw_original_data_distribution():
    NUM = 10000
    data = np.random.normal(1, 2, NUM)


def DAT_generator(graph_topology, wtd_array, diffusive_source):
    # This function returns the node arrival times of the (topology,waiting time distribution) by sampling from a gauss distribution.
    # DAT is a dict{node:time}.DAT_PATH is a dict{node:[node1,node1,node]}
    # length = len(graph_topology.edges())

    # **This function is used in the dats_generator() function

    i = 0
    # First, it randomly generates a weighted graph with random edge weights.
    for edge in graph_topology.edges(data=True):
        graph_topology[edge[0]][edge[1]]["weight"] = wtd_array[i]
        i += 1

    # print(graph_topology.edges(data=True))

    # Second, calculate the shortest path and shortest path length of every node in the weighted graph and those are the DAT and DAT_PATH.
    # the path length from the source to each node is the diffusive arrival time(DAT)
    # and the set of the passed nodes of each diffusive path is the DAT_PATH.
    DAT_PATH = nx.shortest_path(graph_topology, source=diffusive_source, weight="weight")
    DAT = nx.shortest_path_length(graph_topology, source=diffusive_source, weight="weight")

    return DAT, DAT_PATH


def dats_generator(graph_topology, dat_number=1, seed=True):
    # This function returns the list of DATs.
    # dat_number is the number of DAT that you want.
    # You can set seed=True if you want the dats is always fixed whenever you run the sampling code.
    # If you want to change the distribution of waiting times(WTD). You can change the function used in the codes.
    #	|-->"wtd_array = np.random.normal(5, 1, length)"<--|
    # Seed = TRUE,FIXED   Seed = False,Random

    length = len(graph_topology.edges())
    dat = []
    dat_path = []

    for i in range(dat_number):
        if seed == True:
            np.random.seed(i)
        diffusive_source = np.random.randint(len(graph_topology.nodes()))

        # The default WTD can be changed here.
        # Now we use the normal distribution as the waiting time distribution.
        # the average value and standard error of the Gaussian are 2.5 and 0.4.
        # wtd_array = np.random.normal(1, 0.4, length)
        j = 0
        wtd_array = list()
        while j < length:
            num = np.random.normal(2.5, 0.2)
            # num = np.random.beta(0.5,0.5)*5
            if num > 0:
                wtd_array.append(num)
                j += 1

        # Use the function DAT_generator() to create each DAT and DAT_PATH.
        dat_temp, dat_path_temp = DAT_generator(graph_topology, wtd_array, diffusive_source)
        dat.append(dat_temp)
        dat_path.append(dat_path_temp)
        # print(dat_path_temp)

    return dat, dat_path


def draw_diffusion_tree(diffusive_arrival_times, type=0):
    # Not used.
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
    nx.draw_networkx(Diffusion_Tree, pos, with_labels=True, node_color='blue', node_size=4)

    return Diffusion_Tree


def find_possible_edges_from_dats(dats):
    # not used
    node_list_sum = list()
    node_list = list()
    edges_set = set()
    # select the first dat to select all the possile edges.Then use all the residual dats to restrict the possible edges.:
    for i in range(len(dats)):
        sorted_dat = sorted(dats[i].items(), key=lambda x: x[1])
        for (node, time) in sorted_dat:
            node_list.append(node)
        temp_edges = find_ordered_tuples_in_list(node_list)
        node_list = []
        for edge in temp_edges:
            edges_set.add(edge)

    return list(edges_set)


def generate_all_edges(graph):
    # IMPORTANT!! THE input graph must be a networkx graph.
    # use the serial number of the nodes to generate all the possible edges in the graph.
    nodes = list()
    number = len(nodes)
    edges_list = list()

    # find the serial numbers of all nodes for the edges' generation.
    for node in graph.nodes():
        nodes.append(node)

    # all edges except the self loops (i,i).

    # undirected graph
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            if i != j:
                edges_list.append((i, j))
    """
    # directed graph    
    for i in range(len(nodes)):
        for j in range(len(nodes)):
            if i != j:
                edges_list.append((i, j))
    """

    # return a list of all possible edges in the given graph.
    return edges_list


def find_ordered_tuples_in_list(list_in):
    # not used
    length = len(list_in)
    possible_edges = list()
    for i in range(length):
        for j in range(i + 1, length):
            possible_edges.append((list_in[i], list_in[j]))
    return possible_edges


def initialize_lambda(graph, cas_num):
    # initialize the diagonal element of Laplace Matrix:{lambda_i_v} of the graph
    # demo of lambda_dict_list: [{dat1},{dat2},{dat3},{dat4}...]
    # initialize the lambda dict list
    nodes = graph.nodes()
    lambda_dict_list = [{} for i in range(cas_num)]

    # set all the elements to 1e-5
    for i in range(len(lambda_dict_list)):
        for node in nodes:
            lambda_dict_list[i][node] = 0.00001
    # return the list of dictionaries of lambda
    return lambda_dict_list


def calculate_gain_del(edge, dats, lambda_dict_list, discrete_wtd, survival_func, hazard_f, delta, l_t):
    # serial number of node u and v
    u = edge[0]
    v = edge[1]
    max_index = l_t / delta

    # initialize the marginal gain.
    gain = 0
    adder = 0

    for i in range(len(dats)):

        # find t_u(u's arrival time) and t_v(v's arrival time) in the i th dat.
        t_u = dats[i][u]
        t_v = dats[i][v]

        dst_node = u if t_u > t_v else v

        # lambda of i th dat, node dst
        lambda_i_dst = lambda_dict_list[i][dst_node]

        # index
        d_uv_index = abs(int((t_v - t_u) / delta))

        # case 1
        if 0 < d_uv_index < max_index:
            """
             temp = survival_func[d_uv_index] - discrete_wtd[d_uv_index] * delta + ((1 - hazard_f[d_uv_index] * delta) *
                                                                                   discrete_wtd[d_uv_index]) / (
                                                                                      lambda_i_dst - hazard_f[
                                                                                       d_uv_index])
            """
            adder = - np.log(survival_func[d_uv_index]*2 - discrete_wtd[d_uv_index] * delta + (
            (1 - hazard_f[d_uv_index] * delta) * discrete_wtd[d_uv_index]) / (lambda_i_dst - hazard_f[d_uv_index]))
            # adder = np.log(1 - (hazard_f[d_uv_index] / lambda_i_dst)) - np.log(survival_func[d_uv_index])
        else:
            pass
            # adder = -999
        """ # not to consider the situations of out range.
        # case 2
        elif d_uv_index <= 0:
            adder = float(0)

        # case 3
        elif d_uv_index > max_index:
            adder = float(9999)
        """
        # cumulation
        gain += adder
        """
        if gain < -100:
            gain = - 100
        """

    return gain


def calculate_gain_add(edge, dats, lambda_dict_list, discrete_wtd, survival_func, hazard_f, delta, l_t):
    # same as calculate_gain_del

    # serial number of node u and v
    u = edge[0]
    v = edge[1]
    max_index = l_t / delta

    # initialize the marginal gain.
    gain = 0
    adder = 0

    for i in range(len(dats)):

        # find t_u(u's arrival time) and t_v(v's arrival time) in the i th dat.
        t_u = dats[i][u]
        t_v = dats[i][v]

        dst_node = u if t_u > t_v else v

        # lambda of i th dat, node v
        lambda_i_dst = lambda_dict_list[i][dst_node]

        # index
        d_uv_index = abs(int((t_v - t_u) / delta))
        # print("d_uv_index:",d_uv_index)
        # case 1
        if 0 < d_uv_index < max_index:
            adder = np.log(
                survival_func[d_uv_index]*0.8 - discrete_wtd[d_uv_index] * delta + discrete_wtd[d_uv_index] / lambda_i_dst)
            # if hazard_f[d_uv_index] / lambda_i_v <= 0:
            # print(hazard_f[d_uv_index],lambda_i_v,"wocaonixuema")
            # adder = np.log(1 + (hazard_f[d_uv_index] / lambda_i_dst)) + np.log(survival_func[d_uv_index])
        else:
            pass
        """
        # case 2
        elif d_uv_index <= 0:
            adder = 0

        # case 3
        elif d_uv_index > max_index:
            adder = float(-9999)
        """
        # cumulation
        gain += adder
        """
        if gain < -100:
            gain = -100
        """
    # print(gain)
    return gain


def renew_lambda_del(edge, dats, lambda_dict_list, survival_func, hazard_f, delta, l_t):
    # renew the lambda of the node v in each dat

    # index of node u,v
    u = edge[0]
    v = edge[1]
    max_index = l_t / delta

    for i in range(len(dats)):
        t_u = dats[i][u]
        t_v = dats[i][v]

        d_uv_index = abs(int((t_v - t_u) / delta))

        dst_node = u if t_u > t_v else v

        # case 1: valid time interval
        if 0 < d_uv_index < max_index:
            # print("cao")
            lambda_dict_list[i][dst_node] = (lambda_dict_list[i][dst_node] - hazard_f[d_uv_index]) / (1 - hazard_f[d_uv_index] * delta)

            # lambda_i_v = lambda_i_v - hazard_f[d_uv_index]
            # print("del:",lambda_i_v,"  d_uv_index: ", d_uv_index)
            # time.sleep(1)
        """
        # case 2: d_uv < 0
        elif d_uv_index <= 0:
            continue

        # case 3: d_uv > l_t
        elif d_uv_index > max_index:
            continue
        """

    # return the modified lambda_dict_list
    return lambda_dict_list


def renew_lambda_add(edge, dats, lambda_dict_list, survival_func, hazard_f, delta, l_t):
    # same as renew_lambda_del
    # renew the lambda of the node v in each dat

    # index of node u,v
    u = edge[0]
    v = edge[1]
    max_index = int(l_t / delta)

    for i in range(len(dats)):
        t_u = dats[i][u]
        t_v = dats[i][v]
        d_uv_index = abs(int((t_v - t_u) / delta))

        dst_node = u if t_u > t_v else v

        # case 1:valid time interval
        if 0 < d_uv_index < max_index:
            lambda_dict_list[i][dst_node] = (1 - hazard_f[d_uv_index] * delta) * lambda_dict_list[i][dst_node] + hazard_f[d_uv_index]
        """
        # case 2:d_uv < 0
        elif d_uv_index <= 0:
            continue

        # case 3: d_uv > l_t
        elif d_uv_index > max_index:
            continue
        """
    return lambda_dict_list


def mcmc_iteration_temporal_result(source_graph, sample_graph):
    # evaluate the result of the topology reconstruction

    TP_number = FP_number = 0

    for edge in sample_graph.edges():
        if edge in source_graph.edges():
            TP_number += 1
        else:
            FP_number += 1

    source_edge_number = len(source_graph.edges())
    sample_graph_number = len(sample_graph.edges())

    print("accuracy:", TP_number / (TP_number + FP_number + 1), " | ", "completeness: ", TP_number / source_edge_number)


def mcmc_algorithm_with_gibbs_sampling(demo_graph, input_graph, discrete_wtd, dats, delta=0.01, l_t=5):
    #  MCMC Algorithm with Gibbs Sampling
    CAS_NUM = len(dats)  # the number of dats

    # parameters
    M_SAMPLE_SIZE = 10
    BURN_IN = 5
    MAX_LAG = 5
    iteration = 0
    graph_sample_number = 0
    lag = 0

    # discrete wtd,survival f and hazard f
    survival_f = pdf2sf_using_density(discrete_wtd)
    # survival_f = pdf2sf_using_density(discrete_wtd)
    hazard_f = pdf_sf2hazard(discrete_wtd, survival_f)

    # hazard_func = list(map(lambda x: x[0] / x[1], zip(discrete_wtd, survival_f)))
    # print(len(hazard_f))

    # the final output: M sample graphs list
    graph_samples = list()
    # all the possible edges which may occur in the graph.
    possible_edges = generate_all_edges(input_graph)

    # the diagonal element of Laplace Matrix
    lambda_dict_list = initialize_lambda(input_graph, CAS_NUM)

    # the graph used in the iteration, as a temporal graph
    # Whether to use directed graph or not??
    graph_iteration = nx.Graph()

    # the main loop
    while graph_sample_number < M_SAMPLE_SIZE:

        print("iteration:", iteration, "  ", "sampled number:", graph_sample_number)
        iteration += 1

        # using the edges generated by "generate_all_edges(input_graph)"
        for edge in possible_edges:
            # time.sleep(2)
            # if the edge is in the current graph, calculate whether to delete it or not using marginal gain
            if edge in graph_iteration.edges():
                marginal_gain = 1
                marginal_gain += calculate_gain_del(edge, dats, lambda_dict_list, discrete_wtd, survival_f, hazard_f,
                                                    delta, l_t)
                p_uv = 1 / (1 + np.exp(-marginal_gain))
                # print("del:", marginal_gain)
                # time.sleep(1)
                # set a acceptance rate
                if np.random.rand() < p_uv:

                    # renew graph states
                    graph_iteration.remove_edge(edge[0], edge[1])
                    for i in range(CAS_NUM):
                        lambda_dict_list = renew_lambda_del(edge, dats, lambda_dict_list, survival_f, hazard_f, delta,
                                                            l_t)

            # if the edge is NOT in the current graph, calculate whether to add it or not
            else:
                marginal_gain = -1
                marginal_gain += calculate_gain_add(edge, dats, lambda_dict_list, discrete_wtd, survival_f, hazard_f,
                                                    delta, l_t)
                p_uv = 1 / (1 + np.exp(-marginal_gain))

                # test
                # print("add:", marginal_gain)
                # time.sleep(1)

                # print(p_uv)
                # acceptance rate
                if np.random.rand() < p_uv:

                    # renew states
                    graph_iteration.add_edge(edge[0], edge[1])
                    for i in range(CAS_NUM):
                        lambda_dict_list = renew_lambda_add(edge, dats, lambda_dict_list, survival_f, hazard_f, delta,
                                                            l_t)

        print("edges number: ", len(graph_iteration.edges()))
        mcmc_iteration_temporal_result(demo_graph, graph_iteration)

        # fist let the markov chain goes into the stable state
        # BURN_IN is the initial iteration number
        if iteration > BURN_IN:
            lag += 1

        # store a sample of the graph if lag is up to max_lag, the recount lag
        if lag == MAX_LAG:
            graph_samples.append(nx.Graph(graph_iteration))
            graph_sample_number += 1
            lag = 0
        # draw_graph(graph_iteration)

    # return the list of graph samples
    return graph_samples


def cutting_operation(graph_samples):
    count_dict = {}

    for graph in graph_samples:
        for edge in graph.edges:
            if edge in count_dict:
                count_dict[edge] += 1
            else:
                count_dict[edge] = 1
    cutting_graph = nx.Graph()
    for key,value in count_dict.items():
        if value > 6:
            cutting_graph.add_edge(key[0],key[1])
    return cutting_graph


def waiting_time_distribution_iteration(graph_samples, dats):
    wtd_estimation = list()
    pass


def pdf_generator(mode="gaussian",delta=0.01,l_t=5):

    if mode == "gaussian":
        return gaussian_pdf_generate(miu=2.5,sigma=0.2,l_t=l_t,delta=delta)
    elif mode == "weibull":
        # error!
        return weibull_pdf_generate(a=1,b=2.5,l_t=l_t,delta=delta)
    elif mode == "uniform":
        return uniform_pdf_generate(a=1,b=3,l_t=l_t,delta=delta)
    elif mode == "gumbel":
        return gumbel_pdf_generate(a=3,b=2,l_t=l_t,delta=delta)
    elif mode == "beta":
        return beta_pdf_generate(a=9,b=2,l_t=l_t,delta=delta)
    elif mode == "bimodal":
        return bimodal_pdf_generate(a1=2,a2=9,b1=8,b2=3,l_t=l_t,delta=delta)
    elif mode == "pareto":
        return pareto_pdf_generate(a=0.3,b=0.5,l_t=l_t,delta=delta)
    elif mode == "jshape":
        pass
    elif mode == "exponential":
        return exponential_pdf_generate(b=1,l_t=l_t,delta=delta)


def jshape_pdf_generate(a=0.3,b=4,c=0.5,l_t=5,delta=0.01):
    # under working
    discrete_wtd = list()
    discrete_mass = list()
    discrete_number = int(l_t / delta)
    for i in range(0,discrete_number):
        x = i * delta/l_t
        print(x)
        # here to change the formula of the distribution
        discrete_wtd.append(math.gamma(a+b)/(math.gamma(a)*math.gamma(b))*(x**(a-1))*((1-x)**(b-1)))
        discrete_mass.append(discrete_wtd[i] * delta)
    # scatter_wtd(discrete_mass)
    # scatter_wtd(discrete_wtd)

    # normalization
    mass_sum = sum(discrete_mass)
    discrete_mass = [x / mass_sum for x in discrete_mass]
    discrete_wtd = [x / mass_sum for x in discrete_wtd]
    # scatter_wtd(discrete_mass)
    # scatter_wtd(discrete_wtd)

    # discrete_wtd is a list of the discrete point of WTD.for example, list[1] = exp(delta), list[n] = exp(delta * n)
    # discrete_mass is the list of mass of each small section.
    # we return both of them for the convenience of further calculation.

    # the x-axis begins from 1, that is {1,2,...,l_t}, but note that the index of the "list" starts from 0.
    return discrete_wtd, discrete_mass


def beta_pdf_generate(a=9,b=2,l_t=5,delta=0.01):
    discrete_wtd = list()
    discrete_mass = list()
    discrete_number = int(l_t / delta)
    for i in range(0,discrete_number):
        x = i * delta/l_t
        print(x)
        # here to change the formula of the distribution
        discrete_wtd.append(math.gamma(a+b)/(math.gamma(a)*math.gamma(b))*(x**(a-1))*((1-x)**(b-1)))
        discrete_mass.append(discrete_wtd[i] * delta)
    # scatter_wtd(discrete_mass)
    # scatter_wtd(discrete_wtd)

    # normalization
    mass_sum = sum(discrete_mass)
    discrete_mass = [x / mass_sum for x in discrete_mass]
    discrete_wtd = [x / mass_sum for x in discrete_wtd]
    # scatter_wtd(discrete_mass)
    # scatter_wtd(discrete_wtd)

    # discrete_wtd is a list of the discrete point of WTD.for example, list[1] = exp(delta), list[n] = exp(delta * n)
    # discrete_mass is the list of mass of each small section.
    # we return both of them for the convenience of further calculation.

    # the x-axis begins from 1, that is {1,2,...,l_t}, but note that the index of the "list" starts from 0.
    return discrete_wtd, discrete_mass


def bimodal_pdf_generate(a1=2,a2=9,b1=8,b2=3,l_t=5,delta=0.01):
    discrete_wtd = list()
    discrete_mass = list()
    discrete_number = int(l_t / delta)
    # normalization
    beta1_wtd,beta1_mass = beta_pdf_generate(a1,b1,l_t,delta)
    beta2_wtd,beta2_mass = beta_pdf_generate(a2, b2, l_t, delta)
    for i in range(discrete_number):
        discrete_wtd.append(beta1_wtd[i] + beta2_wtd[i])
        discrete_mass.append(beta1_mass[i] + beta2_mass[i])
    mass_sum = sum(discrete_mass)
    discrete_mass = [x / mass_sum for x in discrete_mass]
    discrete_wtd = [x / mass_sum for x in discrete_wtd]
    # scatter_wtd(discrete_mass)
    # scatter_wtd(discrete_wtd)

    # discrete_wtd is a list of the discrete point of WTD.for example, list[1] = exp(delta), list[n] = exp(delta * n)
    # discrete_mass is the list of mass of each small section.
    # we return both of them for the convenience of further calculation.

    # the x-axis begins from 1, that is {1,2,...,l_t}, but note that the index of the "list" starts from 0.
    return discrete_wtd, discrete_mass


def pareto_pdf_generate(a=0.3,b=0.5,l_t=5,delta=0.01):
    discrete_wtd = list()
    discrete_mass = list()
    discrete_number = int(l_t / delta)
    for i in range(0,discrete_number):
        x = i * delta
        # here to change the formula of the distribution
        if x > a:
            discrete_wtd.append(b/a*(x/a)**(-b-1))
            discrete_mass.append(discrete_wtd[i] * delta)
        else:
            discrete_wtd.append(0)
            discrete_mass.append(0)
    # scatter_wtd(discrete_mass)
    # scatter_wtd(discrete_wtd)

    # normalization
    mass_sum = sum(discrete_mass)
    discrete_mass = [x / mass_sum for x in discrete_mass]
    discrete_wtd = [x / mass_sum for x in discrete_wtd]
    # scatter_wtd(discrete_mass)
    # scatter_wtd(discrete_wtd)

    # discrete_wtd is a list of the discrete point of WTD.for example, list[1] = exp(delta), list[n] = exp(delta * n)
    # discrete_mass is the list of mass of each small section.
    # we return both of them for the convenience of further calculation.

    # the x-axis begins from 1, that is {1,2,...,l_t}, but note that the index of the "list" starts from 0.
    return discrete_wtd, discrete_mass


def exponential_pdf_generate(a=3,b=2,l_t=5,delta=0.01):
    discrete_wtd = list()
    discrete_mass = list()
    discrete_number = int(l_t / delta)
    for i in range(0,discrete_number):
        x = i * delta
        # here to change the formula of the distribution
        discrete_wtd.append(1/b*np.exp(-x/b))
        discrete_mass.append(discrete_wtd[i] * delta)
    # scatter_wtd(discrete_mass)
    # scatter_wtd(discrete_wtd)

    # normalization
    mass_sum = sum(discrete_mass)
    discrete_mass = [x / mass_sum for x in discrete_mass]
    discrete_wtd = [x / mass_sum for x in discrete_wtd]
    # scatter_wtd(discrete_mass)
    # scatter_wtd(discrete_wtd)

    # discrete_wtd is a list of the discrete point of WTD.for example, list[1] = exp(delta), list[n] = exp(delta * n)
    # discrete_mass is the list of mass of each small section.
    # we return both of them for the convenience of further calculation.

    # the x-axis begins from 1, that is {1,2,...,l_t}, but note that the index of the "list" starts from 0.
    return discrete_wtd, discrete_mass


def gumbel_pdf_generate(a=3,b=2,l_t=5,delta=0.01):
    discrete_wtd = list()
    discrete_mass = list()
    discrete_number = int(l_t / delta)
    discrete_wtd.append(0)
    discrete_mass.append(0)
    for i in range(1,discrete_number):
        x = i * delta
        # here to change the formula of the distribution
        discrete_wtd.append(a*b*(x**(-a-1))*np.exp(-b*(x**(-a))))
        discrete_mass.append(discrete_wtd[i] * delta)
    # scatter_wtd(discrete_mass)
    # scatter_wtd(discrete_wtd)

    # normalization
    mass_sum = sum(discrete_mass)
    discrete_mass = [x / mass_sum for x in discrete_mass]
    discrete_wtd = [x / mass_sum for x in discrete_wtd]
    # scatter_wtd(discrete_mass)
    # scatter_wtd(discrete_wtd)

    # discrete_wtd is a list of the discrete point of WTD.for example, list[1] = exp(delta), list[n] = exp(delta * n)
    # discrete_mass is the list of mass of each small section.
    # we return both of them for the convenience of further calculation.

    # the x-axis begins from 1, that is {1,2,...,l_t}, but note that the index of the "list" starts from 0.
    return discrete_wtd, discrete_mass


def weibull_pdf_generate(a=1,b=2.5,l_t=5,delta=0.01):
    discrete_wtd = list()
    discrete_mass = list()
    discrete_number = int(l_t / delta)
    for i in range(discrete_number):
        x = i * delta
        # here to change the formula of the distribution
        discrete_wtd.append(b/(a**b)*(x**(b-1))*np.exp(-(x/a)**b))
        discrete_mass.append(discrete_wtd[i] * delta)
    # scatter_wtd(discrete_mass)
    # scatter_wtd(discrete_wtd)

    # normalization
    mass_sum = sum(discrete_mass)
    discrete_mass = [x / mass_sum for x in discrete_mass]
    discrete_wtd = [x / mass_sum for x in discrete_wtd]
    # scatter_wtd(discrete_mass)
    # scatter_wtd(discrete_wtd)

    # discrete_wtd is a list of the discrete point of WTD.for example, list[1] = exp(delta), list[n] = exp(delta * n)
    # discrete_mass is the list of mass of each small section.
    # we return both of them for the convenience of further calculation.

    # the x-axis begins from 1, that is {1,2,...,l_t}, but note that the index of the "list" starts from 0.
    return discrete_wtd, discrete_mass


def gaussian_pdf_generate(miu=2.5,sigma=0.2,l_t=5,delta=0.01):
    # gauss distribution by default
    discrete_wtd = list()
    discrete_mass = list()
    discrete_number = int(l_t / delta)
    for i in range(discrete_number):
        # here to change the formula of the distribution
        discrete_wtd.append(1 / np.sqrt(2 * np.pi) / sigma * np.exp(-((delta * i - miu) ** 2) / (2 * (sigma ** 2))))
        discrete_mass.append(discrete_wtd[i] * delta)
    # scatter_wtd(discrete_mass)
    # scatter_wtd(discrete_wtd)

    # normalization
    mass_sum = sum(discrete_mass)
    discrete_mass = [x / mass_sum for x in discrete_mass]
    discrete_wtd = [x / mass_sum for x in discrete_wtd]
    # scatter_wtd(discrete_mass)
    # scatter_wtd(discrete_wtd)

    # discrete_wtd is a list of the discrete point of WTD.for example, list[1] = exp(delta), list[n] = exp(delta * n)
    # discrete_mass is the list of mass of each small section.
    # we return both of them for the convenience of further calculation.

    # the x-axis begins from 1, that is {1,2,...,l_t}, but note that the index of the "list" starts from 0.
    return discrete_wtd, discrete_mass


def uniform_pdf_generate(a=1,b=3,l_t=5,delta=0.01):
    # gauss distribution by default
    discrete_wtd = list()
    discrete_mass = list()
    discrete_number = int(l_t / delta)
    for i in range(discrete_number):
        # here to change the formula of the distribution
        if a < i*delta < b:
            discrete_wtd.append(1/abs(b-a)/delta)
            discrete_mass.append(discrete_wtd[i] * delta)
        else:
            discrete_wtd.append(0)
            discrete_mass.append(0)
    # scatter_wtd(discrete_mass)
    # scatter_wtd(discrete_wtd)

    # normalization
    mass_sum = sum(discrete_mass)
    discrete_mass = [x / mass_sum for x in discrete_mass]
    discrete_wtd = [x / mass_sum for x in discrete_wtd]
    # scatter_wtd(discrete_mass)
    # scatter_wtd(discrete_wtd)

    # discrete_wtd is a list of the discrete point of WTD.for example, list[1] = exp(delta), list[n] = exp(delta * n)
    # discrete_mass is the list of mass of each small section.
    # we return both of them for the convenience of further calculation.

    # the x-axis begins from 1, that is {1,2,...,l_t}, but note that the index of the "list" starts from 0.
    return discrete_wtd, discrete_mass


def continuous_func_distribution2discrete(delta=0.01, l_t=5):
    # gauss distribution by default
    miu = 2.5
    sigma = 0.2
    discrete_wtd = list()
    discrete_mass = list()
    discrete_number = int(l_t / delta)
    for i in range(discrete_number):
        # here to change the formula of the distribution
        discrete_wtd.append(1 / np.sqrt(2 * np.pi) / sigma * np.exp(-((delta * i - miu) ** 2) / (2 * (sigma ** 2))))
        discrete_mass.append(discrete_wtd[i] * delta)
    # scatter_wtd(discrete_mass)
    # scatter_wtd(discrete_wtd)

    # normalization
    mass_sum = sum(discrete_mass)
    discrete_mass = [x / mass_sum for x in discrete_mass]
    discrete_wtd = [x / mass_sum for x in discrete_wtd]
    # scatter_wtd(discrete_mass)
    # scatter_wtd(discrete_wtd)

    # discrete_wtd is a list of the discrete point of WTD.for example, list[1] = exp(delta), list[n] = exp(delta * n)
    # discrete_mass is the list of mass of each small section.
    # we return both of them for the convenience of further calculation.

    # the x-axis begins from 1, that is {1,2,...,l_t}, but note that the index of the "list" starts from 0.
    return discrete_wtd, discrete_mass


def continuous_exp_distribution2discrete_using_integration(delta, l_t, miu=1):
    # not used
    discrete_wtd = list()
    discrete_mass = list()
    discrete_wtd.append(0)
    discrete_mass.append(0)
    left = float(0)
    right = float(delta)
    for i in range(1, int(l_t / delta)):
        temp = sp.integrate(sp.exp(-miu * x), (x, left, right))
        discrete_mass.append(temp)
        discrete_wtd.append(discrete_mass[i] / delta)
        left = right
        right += delta
    # if there is something needed, it must be the normalization step, which makes the sum of discrete_mass equals to 1.

    # discrete_wtd is a list of the discrete point of WTD.for example, list[1] = exp(delta), list[n] = exp(delta * n)
    # discrete_mass is the list of mass of each small section.
    return discrete_wtd, discrete_mass


def continuous_gaussian_distribution2discrete_using_integration(delta, l_t, miu=2, sigma=0.4):
    # not used
    discrete_wtd = list()
    discrete_mass = list()
    discrete_wtd.append(0)
    discrete_mass.append(0)
    left = float(0)
    right = float(delta)
    for i in range(1, int(l_t / delta)):
        temp = sp.integrate(sp.sqf_norm(-miu * x), (x, left, right))
        discrete_mass.append(temp)
        discrete_wtd.append(discrete_mass[i] / delta)
        left = right
        right += delta
    # if there is something needed, it must be the normalization step, which makes the sum of discrete_mass equals to 1.
    # discrete_wtd is a list of the discrete point of WTD.for example, list[1] = exp(delta), list[n] = exp(delta * n)
    # discrete_mass is the list of mass of each small section.
    return discrete_wtd, discrete_mass


def pdf2sf_using_mass(discrete_mass):
    # !!IMPORTANT!! Note that the discrete survival function is calculated by the WTD_MASS,not wtd pdf.

    # !!important!! input parameter is the mass of wtd
    sum_mass = sum(discrete_mass)

    sf = list()
    for i in range(0, len(discrete_mass)):
        sf.append(sum_mass - sum(discrete_mass[0:i]))

    normalize_sf(sf)
    return sf


def pdf2sf_using_density(discrete_wtd, delta=0.01):
    sf = list()
    sf.append(1)
    for i in range(1, len(discrete_wtd)):
        sf.append(1 - sum(discrete_wtd[0:i]) * delta)
    return sf


def is_same_pdf(pdf1, pdf2):
    div = list()
    for i in range(len(pdf1)):
        div.append(abs(pdf1[i] / pdf2[i]) if pdf2[i] != 0 else 0)

    if min(div) > 0.9 and max(div) < 1.1:
        return True
    else:
        return False


def sf2hazard(sf, delta=0.01):
    hazard = list()
    hazard.append(0)
    for i in range(1, len(sf)):
        haz = (1 - sf[i] / sf[i - 1]) / delta
        hazard.append(haz)

    return hazard


def pdf_sf2hazard(wtd, sf):
    hazard = list()
    for i in range(len(wtd)):
        hazard.append(wtd[i] / sf[i] if sf[i] != 0 else 0)

    return hazard


def normalize_pdf(f, delta=0.01):
    sum_f = sum(f) * delta
    f = [x / sum_f for x in f]
    return f


def normalize_sf(f):
    if len(f) == 0:
        print("ERROR! survival function: length: 0 !")
    return [x / f[0] for x in f]


def scatter_wtd(f):
    plt.scatter([i for i in range(len(f))], f)
    plt.show()


def draw_graph(graph):
    nx.draw(graph,pso=nx.spring_layout)
    plt.show()


def init_discrete_wtd(dats, delta=0.01, l_t=5):
    miu = 0
    for dat in dats:
        dat_list = sorted(dat.items(), key=lambda x: x[1])
        miu += dat_list[1][1]
    miu /= len(dats)
    init_wtd = list()

    # exponential distribution
    for i in range(int(l_t / delta)):
        init_wtd.append(1 / miu * np.exp(-i * delta / miu))

    # gaussian distribution
    # for i in range(int(l_t/delta)):
    #   init_wtd.append(1/np.sqrt(2*np.pi)/0.2 * np.exp(-(i*delta - 4) ** 2/(0.2**2)))
    # normalization
    sum_mass = sum(init_wtd) * delta
    init_wtd = [x / sum_mass for x in init_wtd]
    # scatter_wtd(init_wtd)
    return init_wtd


def Kolmogorov_Smirnov_distance(last_wtd, present_wtd):
    # Kolmogorov-Smirnov distance of the two given wtd.
    max_distance = float(0)
    for i in range(len(last_wtd)):
        max_distance = max(max_distance, abs(present_wtd[i] - last_wtd[i]))
    return max_distance


def get_adj_edge_from_node(graph, node):
    # get the adjacent edges of a node in the undirected graph.
    adj_list = list()
    for edge in graph.edges():
        if node in edge:
            adj_list.append(edge)
    return adj_list


def calculate_sum_hazard(dat, hazard, dst_node, edges_list, delta):
    sum_hazard = 0

    # edges_list is the edges adjacent to the dst_node for the compute of accumulative hazard rate.
    for edge in edges_list:
        u = edge[0]
        v = edge[1]
        '''
        d_uv_index = abs(int((dat[u] - dat[v]) / delta))
        sum_hazard += hazard[d_uv_index]
        '''
        if u == dst_node:
            if dat[u] > dat[v]:
                d_uv_index = abs(int((dat[u] - dat[v]) / delta))
                sum_hazard += hazard[d_uv_index]
        elif v == dst_node:
            if dat[u] < dat[v]:
                d_uv_index = abs(int((dat[u] - dat[v]) / delta))
                sum_hazard += hazard[d_uv_index]

    return sum_hazard


def init_kernel_function(h, l_t=5, delta=0.01):
    # gauss distribution by default

    # parameters
    miu = (l_t - 0) / 2
    sigma = h

    discrete_wtd = list()
    discrete_mass = list()
    discrete_number = int(l_t / delta)
    for i in range(discrete_number):
        # here to change the formula of the distribution
        discrete_wtd.append(1 / np.sqrt(2 * np.pi) / sigma * np.exp(-((delta * i - miu) ** 2) / (2 * (sigma ** 2))))
        discrete_mass.append(discrete_wtd[i] * delta)

    # normalization
    discrete_wtd = normalize_pdf(discrete_wtd)
    discrete_mass = normalize_pdf(discrete_mass)

    # discrete_wtd is a list of the discrete point of WTD.for example, list[1] = exp(delta), list[n] = exp(delta * n)
    # discrete_mass is the list of mass of each small section.
    # we return both of them for the convenience of further calculation.

    # the x-axis begins from 1, that is {1,2,...,l_t}, but note that the index of the "list" starts from 0.
    return discrete_wtd


def wtd_estimation(graph_topology, dats, last_wtd, delta=0.01, l_t=5):
    # Kernel width
    # h = 0.01* l_t
    h = 0.05
    # index length
    INDEX_LEN = len(last_wtd)

    # wtd, mass, survival, hazard initialization
    present_wtd = [0 for i in range(INDEX_LEN)]

    # init discrete kernel function
    kernel_func = init_kernel_function(h, l_t, delta)

    while True:

        # update
        survival_f = pdf2sf_using_density(last_wtd)
        hazard_f = pdf_sf2hazard(last_wtd, survival_f)

        # auxiliary	variable
        tdoa = [0 for i in range(INDEX_LEN)]
        beta = [0 for i in range(INDEX_LEN)]

        for dat in dats:

            for edge in graph_topology.edges():
                u = edge[0]
                v = edge[1]
                dst_node = u if dat[u] > dat[v] else v

                # discrete index in the list
                d_uv_index = abs(int((dat[u] - dat[v]) / delta))

                if 0 < d_uv_index < (INDEX_LEN):
                    # accumulate tdoa and beta
                    tdoa[d_uv_index] += 1
                    adj_edge_list = get_adj_edge_from_node(graph_topology, dst_node)
                    beta[d_uv_index] += hazard_f[d_uv_index] / calculate_sum_hazard(dat, hazard_f, dst_node,
                                                                                    adj_edge_list, delta)

        # tdoa = [10000*x for x in tdoa]
        # beta = [10000*x for x in beta]
        present_wtd = [0 for i in range(INDEX_LEN)]

        # accumulate tdoa and beta to get the present wtd against the last wtd
        for i in range(INDEX_LEN):
            if tdoa[i] > 0:
                present_wtd[i] += beta[i]
                for j in range(i + 1, INDEX_LEN):
                    present_wtd[j] += (tdoa[i] - beta[i]) * 0.05 * last_wtd[j] / survival_f[i]

        # scatter_wtd(present_wtd)
        # use kernel function to smooth the wtd
        present_wtd = np.convolve(present_wtd, kernel_func, mode="same")
        # normalization
        present_wtd = normalize_pdf(present_wtd)

        if Kolmogorov_Smirnov_distance(last_wtd, present_wtd) < 0.0001:
            scatter_wtd(present_wtd)
            break

        print(Kolmogorov_Smirnov_distance(last_wtd, present_wtd))
        # scatter_wtd(present_wtd)
        last_wtd = present_wtd[:]

        # scatter_wtd(survival_f)
        # scatter_wtd(hazard_f)

    # scatter_wtd(present_wtd)
    last_wtd = present_wtd[:]

    return present_wtd


def calculate_wtd_prob(dat,wtd,sf,i,j,delta=0.01,l_t=5):
    d_uv_index = int(abs(dat[i] - dat[j])/delta)
    if 0 <= d_uv_index < int(l_t/delta):
        return wtd[d_uv_index]*sf[d_uv_index]
    else:
        return -10


def faster_topology_reconstruction_through_dats_based_on_wtd1(dats, wtd):
        CAS = len(dats)
        NODE_NUMBER = len(dats[0])
        probobility_matrix = np.zeros((NODE_NUMBER,NODE_NUMBER),dtype=float)
        time_aggregated_graph = nx.Graph()
        sf = pdf2sf_using_density(wtd, delta=0.01)
        for dat in dats:
            dat_sorted_list = sorted(dat.items(),key=lambda x: x[1], reverse=False)
            for i in range(NODE_NUMBER):
                for j in range(i+1,NODE_NUMBER):
                    adder = calculate_wtd_prob(dat, wtd,sf, i, j, delta=0.01, l_t=5)
                    probobility_matrix[i][j] += adder
                    probobility_matrix[j][i] += adder

        max_num = probobility_matrix.max()
        min_num = probobility_matrix.min()
        print("max: ", max_num, "min:", min_num)
        probobility_matrix *= 1/max_num

        for i in range(NODE_NUMBER):
            for j in range(NODE_NUMBER):
                if probobility_matrix[i][j] > 0.4:
                    probobility_matrix[i][j] = 1
                else:
                    probobility_matrix[i][j] = 0
        return probobility_matrix



def pdf_recover(dats,i,j,delta=0.01,l_t=5):
    time_interval_list = []
    out_off_range_count = 0
    pdf_recover = [0 for i in range(int(l_t/delta/10))]
    for dat in dats:
        if int(abs((dat[i] - dat[j])/delta)) < 500:
            time_interval_list.append(int(abs((dat[i] - dat[j])/delta)))
        else:
            out_off_range_count += 1

    for dt in time_interval_list:
        pdf_recover[dt//10] += 1
    pdf_recover.append(out_off_range_count)
    return pdf_recover



def faster_topology_reconstruction_through_dats_based_on_wtd2(dats, wtd):
    CAS = len(dats)
    NODE_NUMBER = len(dats[0])
    probobility_matrix = np.zeros((NODE_NUMBER, NODE_NUMBER), dtype=float)
    time_aggregated_graph = nx.Graph()
    sf = pdf2sf_using_density(wtd, delta=0.01)
    cutting = 0.005
    cutting_index = 0
    for i in range(len(sf)):
        if sf[i] < cutting:
            cutting_index = i//10
            break
    print(cutting_index)
    for i in range(NODE_NUMBER):
        for j in range(i+1, NODE_NUMBER):
            pdf_rcvr = pdf_recover(dats,i,j,delta=0.01,l_t=5)
            if pdf_rcvr[50] > 1:
                probobility_matrix[i][j] = float(-100)
                probobility_matrix[j][i] = float(-100)
            elif sum(pdf_rcvr[cutting_index:50]) > 1:
                probobility_matrix[i][j] = float(-100)
                probobility_matrix[j][i] = float(-100)
            else:
                probobility_matrix[i][j] = 1
                probobility_matrix[j][i] = 1

    max_num = probobility_matrix.max()
    min_num = probobility_matrix.min()
    print("max: ", max_num, "min:", min_num)
    probobility_matrix *= 1 / max_num

    for i in range(NODE_NUMBER):
        for j in range(NODE_NUMBER):
            if probobility_matrix[i][j] > 0.4:
                probobility_matrix[i][j] = 1
            else:
                probobility_matrix[i][j] = 0
    return probobility_matrix


def count_in_matrix(m):
    count = 0
    for line in m:
        for item in line:
            if item:
                count += 1
    return count


def rapid_recstrct_network_test():
    file_name = ["data/scalefree.txt","data/polbooks.txt","data/football.txt","data/apollonian.txt","data/dolphins.txt","data/karate.txt","data/lattice2d.txt","data/miserables.txt","data/pseudofractal.txt","data/randomgraph.txt","data/scalefree.txt","data/sierpinski.txt","data/smallworld.txt","data/jazz.txt"]
    for name in file_name:
        demo_graph, node_number, edge_number = open_file_data_graph(name)

        # demo_graph, node_number, edge_number = demo_graph_generator()
        print(name, "  graph || nodes:", node_number, "; edges:", edge_number)
        print("average shortest path length: ", nx.average_shortest_path_length(demo_graph))
        print("average clustering coef: ", nx.average_clustering(demo_graph))
        # generate data arrival times(dats), also dat_path.
        # set dat_number to control the total number of dat.
        dats, dat_path = dats_generator(demo_graph, dat_number=int(node_number), seed=False)
        discrete_wtd, discrete_mass = pdf_generator(mode="exponential")
        adj_mat = faster_topology_reconstruction_through_dats_based_on_wtd2(dats, discrete_wtd)
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
        print(TP,FP,TN,FN)
        print("TPR: ", TP / (TP+FN), "FPR:", FP / (FP+TN))
        print("Precision: ",TP/(TP+FP), "Recall: ",TP/(TP+FN))
        print("------------------------------------------------")

if __name__ == "__main__":
    print("module: stn_reconstruction_lib")
