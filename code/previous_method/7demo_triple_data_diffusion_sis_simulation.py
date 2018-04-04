# !/usr/bin/python
# -*- coding: utf-8 -*-
# @Aquamarine GY 2017.11.2


import numpy as np
from graph_tool.all import *
from gi.repository import Gtk, Gdk, GdkPixbuf, GObject
import sys, os, os.path
import time
import random

a = np.array([[0, 1, 1], [1, 2, 2], [2, 3, 1], [2, 3, 3], [3, 4, 5], [1, 4, 2], [4, 5, 3], [1, 5, 2], [5, 6, 2]])
b = [[0, 5, 1], [5, 6, 2], [6, 4, 1], [6, 4, 3], [4, 2, 5], [5, 2, 2], [5, 1, 2], [1, 2, 3], [1, 3, 3]]


def read_workspace_dataset(file_name):
    # 将workspace数据集的数据转化成(u,v,t)的三元组形式
    file = open(file_name, "r")
    triple_dataspace_list = list()
    for line in file.readlines():
        triple_dataspace_list.append([int(x) for x in line.split(' ')])
    standard_set = np.array(triple_dataspace_list)
    standard_set = np.column_stack((standard_set[:, 1], standard_set[:, 2], standard_set[:, 0]))
    return standard_set


def read_high_school_2013_dataset(file_name):
    # 将high_school_2013数据集的数据转化成(u,v,t)的三元组形式
    file = open(file_name, "r")
    triple_dataspace_list = []
    count = 0
    for line in file.readlines():
        triple_dataspace_list.append([int(x) for x in line.split(' ')[0:3]])
    # print(triple_dataspace_list)
    standard_set = np.array(triple_dataspace_list)
    standard_set = np.column_stack((standard_set[:, 1], standard_set[:, 2], standard_set[:, 0]))
    return standard_set


def generate_top_dict(triples):
    top_dict = dict()
    for triple in triples:  # 先确定节点
        top_dict[triple[0]] = {}
        top_dict[triple[1]] = {}
    for triple in triples:  # 再确定连边
        top_dict[triple[0]][triple[1]] = []
        top_dict[triple[1]][triple[0]] = []
    for triple in triples:  # 最后计入到达时间
        top_dict[triple[0]][triple[1]].append(triple[2])
        top_dict[triple[1]][triple[0]].append(triple[2])
    return top_dict


def triples_time_slice(sorted_triples):
    # the structure of the slice:{t1:[[],[],[],...],t2:[[],[],...],...}
    temp_time = sorted_triples[0][2]
    cur_list = list()
    slice_dict = dict()
    for item in sorted_triples:
        if item[2] == temp_time:
            cur_list.append(list(item))
        else:
            if len(cur_list) > 0:
                slice_dict[temp_time] = cur_list
            cur_list = list()
            cur_list.append(list(item))
            temp_time = item[2]
    if len(cur_list) > 0:
        for i in cur_list:
            slice_dict[i[2]] = [i]
    return slice_dict


def calculate_startTime_endTime_timeWindowLength(sorted_triples):
    # return start_time, end_time, time_interval, time_window_length
    start_time = sorted_triples[0][2]
    end_time = sorted_triples[-1][2]
    time_window_length = end_time - start_time
    return start_time, end_time, time_window_length


def sis_simulation_method_oneStep(sorted_triples, diffusive_source, min_time_interval, beta=1, recover_rate=0.00,
                                  repeat=1):
    slice_dict = triples_time_slice(sorted_triples)
    start_time, end_time, time_window_length = calculate_startTime_endTime_timeWindowLength(data)
    node_state_dict = dict()  # indicate the state of each node
    infected_node_dict = dict()  # a dict of infected nodes, for the convenience of searching (s,i) pairs.
    node_infected_count_dict = dict()  # the number of each node's infection in the whole time line.
    edge_diffusion_count_dict = dict()  # the number of each edge's infection, that means how many times does each edge diffuse virus.
    #  calculate the occurrence frequency of each edge.
    #  It measures the importance of each edge in the graph while diffusing.
    for item in sorted_triples:  # nodes of the graph
        node_state_dict[item[0]] = 0
        node_state_dict[item[1]] = 0
        node_infected_count_dict[item[0]] = 0
        node_infected_count_dict[item[1]] = 0
    node_state_dict[diffusive_source] = 1
    infected_node_dict[diffusive_source] = 1
    node_infected_count_dict[diffusive_source] += 1
    # initiate the vertex state, the slice of triples by different time.
    # while running the diffusion process in time order, just update the state of each node.
    # The animation just need the states of nodes in each time step.Then we can show the pictures in time ascending order.
    temp_time = start_time
    # print(slice_dict)
    while temp_time <= end_time:
        if temp_time in slice_dict:
            sis_diffuse(slice_dict[temp_time], infected_node_dict, node_state_dict, node_infected_count_dict, beta)
            # if there are some triples in the current time, just run the diffusion process at a probability beta.
            # The input are triples at current time,node state dict and diffusion rate beta.

        sis_recover(node_state_dict, infected_node_dict, recover_rate)
        # Every single time step there is a recover rate for each infected node.
        temp_time += min_time_interval

    print(node_infected_count_dict)


def sis_diffuse(temp_time_data_list, infected_node_dict, node_state_dict, node_infected_count_dict, beta):
    temp_record_dict = dict()
    # Use a temp dict to record the new infected node in the current time step.
    # Avoiding multi infection in a single path at a single time step.
    # For example, if nodeA was infected at time t1. And in the next iteration, nodeB was also infected by nodeA at time t1.
    # But this situation is not allowed in our algorithm.
    for item in temp_time_data_list:
        if (item[0] in infected_node_dict) and (item[1] not in infected_node_dict):
            if random.random() < beta:
                temp_record_dict[item[1]] = item[2]
                print("infected: ", item[1], "time: ", item[2])
                state[graph_in_graph_tool.vertex(item[1])] = INFECTED
                continue
        if (item[1] in infected_node_dict) and (item[0] not in infected_node_dict):
            if random.random() < beta:
                temp_record_dict[item[0]] = item[2]
                state[graph_in_graph_tool.vertex(item[0])] = INFECTED
                print("infected: ", item[0], "time: ", item[2])
                continue
    for key in temp_record_dict.keys():
        infected_node_dict[key] = 1
        node_state_dict[key] = 1
        node_infected_count_dict[key] += 1


def sis_recover(node_state_dict, infected_node_dict, recover_rate):
    copy_infected_dict = dict(infected_node_dict)
    for key, value in copy_infected_dict.items():
        if random.random() < recover_rate:
            recover_node = key
            state[graph_in_graph_tool.vertex(recover_node)] = UNINFECTED
            infected_node_dict.pop(key)
            print("recover: ", recover_node)


def update_state():

    global temp_time, slice_dict, infected_node_dict, node_state_dict,node_infected_count_dict, beta, recover_rate,min_time_inverval, start_time, end_time
    if temp_time <= end_time:
        print(temp_time)
        if temp_time in slice_dict:
            temp_record_dict = dict()
            # Use a temp dict to record the new infected node in the current time step.
            # Avoiding multi infection in a single path at a single time step.
            # For example, if nodeA was infected at time t1. And in the next iteration, nodeB was also infected by nodeA at time t1.
            # But this situation is not allowed in our algorithm.
            for item in slice_dict[temp_time]:
                if (item[0] in infected_node_dict) and (item[1] not in infected_node_dict):
                    if random.random() < beta:
                        temp_record_dict[item[1]] = item[2]
                        print("infected: ", item[1], "time: ", item[2])
                        state[graph_in_graph_tool.vertex(item[1])] = INFECTED
                        continue
                if (item[1] in infected_node_dict) and (item[0] not in infected_node_dict):
                    if random.random() < beta:
                        temp_record_dict[item[0]] = item[2]
                        state[graph_in_graph_tool.vertex(item[0])] = INFECTED
                        print("infected: ", item[0], "time: ", item[2])
                        continue
            for key in temp_record_dict.keys():
                infected_node_dict[key] = 1
                node_state_dict[key] = 1
                node_infected_count_dict[key] += 1
        copy_infected_dict = dict(infected_node_dict)
        for key, value in copy_infected_dict.items():
            if random.random() < recover_rate:
                recover_node = key
                state[graph_in_graph_tool.vertex(recover_node)] = R
                infected_node_dict.pop(key)
                print("recover: ", recover_node)
        # Every single time step there is a recover rate for each infected node.
        temp_time += min_time_interval
    else:
        temp_time = start_time

    win.graph.regenerate_surface()
    win.graph.queue_draw()

    # We need to return True so that the main loop will call this function more
    # than once.
    return True


work_space_data = read_workspace_dataset("data/dataset_workspace.dat")
top_dict = generate_top_dict(work_space_data)

graph_in_graph_tool = Graph(directed=False)
graph_in_graph_tool.add_vertex(10000)
for key1, value_dict in top_dict.items():
    for key2 in value_dict.keys():
        if graph_in_graph_tool.edge(key1, key2) not in graph_in_graph_tool.edges():
            graph_in_graph_tool.add_edge(key1, key2)
# Add vertices and edges to the new graph in graph_tool.
# Make sure there is only one edge between each pair of vertices.

removed = graph_in_graph_tool.new_vertex_property("bool")
removed.a = True
for v in graph_in_graph_tool.vertices():
    if v.out_degree() > 0:
        removed[v] = False
graph_in_graph_tool.set_vertex_filter(removed, inverted=True)
# Because of the messy nodes number, we should remove the nodes that are not appeared in the real triples.

S = [1, 1, 1, 1]
I = [0, 0, 0, 1]
R = [0.5,0.5,0.5,1]
state = graph_in_graph_tool.new_vertex_property("vector<double>")
for v in graph_in_graph_tool.vertices():
    state[v] = S
pos = sfdp_layout(graph_in_graph_tool)

UNINFECTED = [1, 1, 1, 1]  # White color  suspected
INFECTED = [0, 0, 0, 1]  # Black color  infected

state = graph_in_graph_tool.new_vertex_property("vector<double>")
newly_infected = graph_in_graph_tool.new_vertex_property("bool")
for v in graph_in_graph_tool.vertices():
    state[v] = UNINFECTED

refresh_count = 0
offscreen = sys.argv[1] == "offscreen" if len(sys.argv) > 1 else False
max_count = 500


diffusive_source = 17
min_time_interval=20
beta=1
recover_rate=0.001

sorted_triples = work_space_data
slice_dict = triples_time_slice(sorted_triples)
start_time, end_time, time_window_length = calculate_startTime_endTime_timeWindowLength(sorted_triples)
node_state_dict = dict()  # indicate the state of each node
infected_node_dict = dict()  # a dict of infected nodes, for the convenience of searching (s,i) pairs.
node_infected_count_dict = dict()  # the number of each node's infection in the whole time line.
edge_diffusion_count_dict = dict()  # the number of each edge's infection, that means how many times does each edge diffuse virus.
#  calculate the occurrence frequency of each edge.
#  It measures the importance of each edge in the graph while diffusing.
for item in sorted_triples:  # nodes of the graph
    node_state_dict[item[0]] = 0
    node_state_dict[item[1]] = 0
    node_infected_count_dict[item[0]] = 0
    node_infected_count_dict[item[1]] = 0
node_state_dict[diffusive_source] = 1
infected_node_dict[diffusive_source] = 1
node_infected_count_dict[diffusive_source] += 1
state[diffusive_source] = INFECTED
# initiate the vertex state, the slice of triples by different time.
# while running the diffusion process in time order, just update the state of each node.
# The animation just need the states of nodes in each time step.Then we can show the pictures in time ascending order.
temp_time = start_time


if offscreen and not os.path.exists("./frames"):
    os.mkdir("./frames")
if not offscreen:
    win = GraphWindow(graph_in_graph_tool, pos, geometry=(500, 400),
                      edge_color=[0.6, 0.6, 0.6, 1],
                      vertex_fill_color=state,
                      vertex_halo=newly_infected,
                      vertex_halo_color=[0.8, 0, 0, 0.6])
else:
    count = 0
    win = Gtk.OffscreenWindow()
    win.set_default_size(500, 400)
    win.graph = GraphWidget(graph_in_graph_tool, pos,
                            edge_color=[0.6, 0.6, 0.6, 1],
                            vertex_fill_color=state,
                            vertex_halo=newly_infected,
                            vertex_halo_color=[0.8, 0, 0, 0.6])
    win.add(win.graph)

# Bind the function above as an 'idle' callback.
cid = GObject.idle_add(update_state)

# We will give the user the ability to stop the program by closing the window.
win.connect("delete_event", Gtk.main_quit)

# Actually show the window, and start the main loop.
win.show_all()
Gtk.main()
