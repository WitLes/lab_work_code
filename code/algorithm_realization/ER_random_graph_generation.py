import numpy as np
import networkx as nx
import random
import time
import matplotlib.pyplot as plt
import sympy as spu


def er_graph_generator(node_number=100, link_probability=0.06, seed=None, directed=False):
    # this function returns a ER random graph with input node number and edge link probability.
    # "seed" parameter is the random seed.Fixing it will let your graph fixed every time you run the function.
    # this funtion calls a networkx funtion "nx.erdos_renyi_graph()"

    er_graph = nx.erdos_renyi_graph(node_number, link_probability, seed, directed)
    if directed == False:
        weighted_er_graph = nx.Graph()
    if directed == True:
        weighted_er_graph = nx.DiGraph()

    for edge in er_graph.edges():
        weighted_er_graph.add_edge(edge[0], edge[1], weight=1)
        # weighted_er_graph is used to generate the DATs in the following dat_generator() method which uses the Dijstra Algorithm to calculate the shortest distances of each nodes in the diffusion process.

    node_number = len(weighted_er_graph.nodes())
    edge_number = len(weighted_er_graph.edges())

    return weighted_er_graph, node_number, edge_number

def open_file_data_graph():
    # file name:
    with open("../data/dolphins.txt", encoding="utf8") as data:
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
    nx.draw(graph)
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
        wtd_array = np.random.normal(2, 0.4, length)

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
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            if i != j:
                edges_list.append((i, j))

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


def initialize_lambda(graph):
    # initialize the diagonal element of Laplace Matrix:{lambda_i_v} of the graph

    # initialize the lambda dict list
    nodes = graph.nodes()
    lambda_dict_list = [{} for i in range(len(dats))]

    # set all the elements to 1e-5
    for i in range(len(lambda_dict_list)):
        for node in nodes:
            lambda_dict_list[i][node] = 0.00001

    # return the list of dictionaries of lambda
    return lambda_dict_list


def calculate_gain_del(edge, dats, lambda_dict_list, discrere_wtd, survival_func, hazard_f, delta, l_t):
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

        # lambda of i th dat, node v
        lambda_i_v = lambda_dict_list[i][v]

        # index
        d_uv_index = abs(int((t_v - t_u) / delta))


        # case 1
        if 0 < d_uv_index < max_index:
            temp = survival_func[d_uv_index] - discrete_wtd[d_uv_index]*delta + ((1 - hazard_f[d_uv_index] * delta) * discrete_wtd[d_uv_index]) / (lambda_i_v - hazard_f[d_uv_index])
            adder = - np.log(survival_func[d_uv_index] - discrete_wtd[d_uv_index]*delta + ((1 - hazard_f[d_uv_index] * delta) * discrete_wtd[d_uv_index]) / (lambda_i_v - hazard_f[d_uv_index]))
            # adder = np.log(1 - (hazard_f[d_uv_index] / lambda_i_v)) - np.log(survival_func[d_uv_index])
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


def calculate_gain_add(edge, dats, lambda_dict_list,discrete_wtd, survival_func, hazard_f, delta, l_t):
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

        # lambda of i th dat, node v
        lambda_i_v = lambda_dict_list[i][v]

        # index
        d_uv_index = abs(int((t_v - t_u) / delta))
        # print("d_uv_index:",d_uv_index)
        # case 1
        if 0 < d_uv_index < max_index:
            adder = np.log(survival_func[d_uv_index] - discrete_wtd[d_uv_index] * delta + discrete_wtd[d_uv_index] / lambda_i_v)
            #if hazard_f[d_uv_index] / lambda_i_v <= 0:
                # print(hazard_f[d_uv_index],lambda_i_v,"wocaonixuema")
            #adder = np.log(1 + (hazard_f[d_uv_index] / lambda_i_v)) + np.log(survival_func[d_uv_index])
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

        dst_node = - 1
        if t_u > t_v:
            dst_node = u
        else:
            dst_node = v

        lambda_i_dst = lambda_dict_list[i][dst_node]
        # case 1: valid time interval
        if 0 < d_uv_index < max_index:
            # print("cao")
            lambda_i_dst = (lambda_i_dst - hazard_f[d_uv_index]) / (1 - hazard_f[d_uv_index] * delta)


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

        lambda_dict_list[i][dst_node] = lambda_i_dst
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

        dst_node = - 1
        if t_u > t_v:
            dst_node = u
        else:
            dst_node = v

        lambda_i_dst = lambda_dict_list[i][dst_node]

        # case 1:valid time interval
        if 0 < d_uv_index < max_index:
            d_uv_index = int((t_v - t_u) / delta)
            lambda_i_dst = (1 - hazard_f[d_uv_index] * delta) * lambda_i_dst + hazard_f[d_uv_index]
        """
        # case 2:d_uv < 0
        elif d_uv_index <= 0:
            continue

        # case 3: d_uv > l_t
        elif d_uv_index > max_index:
            continue
        """

        lambda_dict_list[i][dst_node] = lambda_i_dst
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

    print("accuracy:", TP_number / (TP_number + FP_number+1), " | ", "completeness: ", TP_number / source_edge_number)


def mcmc_algorithm_with_gibbs_sampling(input_graph, discrete_wtd, discrete_mass, dats, delta=0.01, l_t=5):
    #  MCMC Algorithm with Gibbs Sampling

    # all the possible edges which may occur in the graph.
    edges = generate_all_edges(input_graph)
    C = len(dats)  # the number of dats

    # parameters
    M_sample_size = 30
    burn_in = 10
    max_lag = 10
    iteration = 0
    graph_sample_number = 0
    lag = 0

    # discrete wtd,survival f and hazard f
    survival_f = pdf2sf(discrete_mass)
    hazard_f = pdf_sf2hazard(discrete_wtd, survival_f)

    # hazard_func = list(map(lambda x: x[0] / x[1], zip(discrete_wtd, survival_f)))
    # print(len(hazard_f))
    # the final output: M sample graphs list
    graph_samples = list()

    # the diagonal element of Laplace Matrix
    lambda_dict_list = initialize_lambda(input_graph)

    # the graph used in the iteration, as a temporal graph
    # Whether to use directed graph or not??
    graph_iteration = nx.Graph()

    # the main loop
    while graph_sample_number < M_sample_size:

        print("iteration:", iteration, "  ", "sampled number:", graph_sample_number)
        iteration += 1

        # using the edges generated by "generate_all_edges(input_graph)"
        for edge in edges:
            # time.sleep(2)
            u = edge[0]
            v = edge[1]

            # if the edge is in the current graph, calculate whether to delete it or not using marginal gain
            if edge in graph_iteration.edges():
                marginal_gain = 1
                marginal_gain += calculate_gain_del(edge, dats, lambda_dict_list, discrete_wtd, survival_f, hazard_f, delta, l_t)
                p_uv = 1 / (1 + np.exp(-marginal_gain))
                # print("del:", marginal_gain)
                # time.sleep(1)
                # set a acceptance rate
                if np.random.rand() < p_uv:

                    # renew graph states
                    graph_iteration.remove_edge(u, v)
                    for i in range(len(lambda_dict_list)):
                        lambda_dict_list = renew_lambda_del(edge, dats, lambda_dict_list, survival_f, hazard_f, delta, l_t)

            # if the edge is NOT in the current graph, calculate whether to add it or not
            else:
                marginal_gain = -1
                marginal_gain += calculate_gain_add(edge, dats, lambda_dict_list, discrete_wtd, survival_f, hazard_f, delta, l_t)
                p_uv = 1 / (1 + np.exp(-marginal_gain))

                # test
                # print("add:", marginal_gain)
                # time.sleep(1)

                # print(p_uv)
                # acceptance rate
                if np.random.rand() < p_uv:

                    # renew states
                    graph_iteration.add_edge(u, v)
                    for i in range(len(lambda_dict_list)):
                        lambda_dict_list = renew_lambda_add(edge, dats, lambda_dict_list, survival_f, hazard_f, delta,
                                                            l_t)

        print("edges number: ", len(graph_iteration.edges()))
        mcmc_iteration_temporal_result(demo_graph, graph_iteration)

        # fist let the markov chain goes into the stable state
        # burn_in is the initial iteration number
        if iteration > burn_in:
            lag += 1

        # store a sample of the graph if lag is up to max_lag, the recount lag
        if lag == max_lag:
            graph_samples.append(nx.Graph(graph_iteration))
            graph_sample_number += 1
            lag = 0

    # return the list of graph samples
    return graph_samples


def cutting_operation(graph_samples):
    pass


def waiting_time_distribution_iteration(graph_samples, dats):
    wtd_estimation = list()
    pass


def pdf2sf(discrete_mass):
    # !!IMPORTANT!! Note that the discrete survival function is calculated by the WTD_MASS,not wtd pdf.

    # !!important!! input parameter is the mass of wtd
    sum_mass = sum(discrete_mass)

    sf = list()
    for i in range(0, len(discrete_mass)):
        point_value = sum_mass - sum(discrete_mass[0:i])
        sf.append(point_value)
    return sf


def pdf_sf2hazard(discrete_wtd, survival_function):
    hazard_function = []
    for i in range(len(discrete_wtd)):
        hazard_function.append(discrete_wtd[i] / survival_function[i])
    return hazard_function


def continuous_func_distribution2discrete(delta=0.01, l_t=5):
    # gauss distribution by default
    miu = 2
    sigma = 0.4
    discrete_wtd = list()
    discrete_mass = list()
    discrete_number = int(l_t / delta)
    for i in range(discrete_number):
        # here to change the formula of the distribution
        discrete_wtd.append(1 / np.sqrt(2 * np.pi) / sigma * np.exp(-((delta * i - miu) ** 2) / (2 * (sigma ** 2))))
        discrete_mass.append(discrete_wtd[i] * delta)

    # if there is something still needed, it must be the normalization step, which makes the sum of discrete_mass equals to 1.

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


def sf2hazard(sf):
    hazard = list()
    hazard.append(0)
    for i in range(1, len(sf)):
        haz = 1 - sf[i] / sf[i - 1]
        hazard.append(haz)
    return hazard


def scatter_wtd(f):
    plt.scatter([i for i in range(len(f))], f)
    plt.show()


# generate the graph topology, return the graph, the number of nodes, the number of edges.
#er_graph, node_number, edge_number = er_graph_generator(100, 0.06, seed=0, directed=True)
demo_graph, node_number, edge_number = open_file_data_graph()
#demo_graph, node_number, edge_number = demo_graph_generator()
print("graph || nodes:", node_number, "; edges:", edge_number)

# generate data arrival times(dats), also dat_path.
# set dat_number to control the total number of dat.
dats, dat_path = dats_generator(demo_graph, dat_number=40, seed=True)
discrete_wtd, discrete_mass = continuous_func_distribution2discrete()

# draw wtd, sf, hazard_f
"""
scatter_wtd(discrete_wtd)
scatter_wtd(discrete_mass)
sf = pdf2sf(discrete_mass)
scatter_wtd(sf)
hazard_f = sf2hazard(sf)
scatter_wtd(hazard_f)
"""

graph_samples = mcmc_algorithm_with_gibbs_sampling(demo_graph, discrete_wtd, discrete_mass, dats, delta=0.01, l_t=5)
