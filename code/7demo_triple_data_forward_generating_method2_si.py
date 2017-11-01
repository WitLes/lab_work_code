# !/usr/bin/python
# -*- coding: utf-8 -*-
# @Aquamarine GY 2017.10.6


import numpy as np
from graph_tool.all import *
from gi.repository import Gtk, Gdk, GdkPixbuf, GObject
import sys, os, os.path
import time

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


def find_diffusion_sourve_arrival_time(top_dict, source):
    # 找到三元组生成的字典中，源节点的到达时间，也就是源节点第一次与别的节点交互的时间
    time_list = list()
    for key in top_dict[source].keys():
        time_list.extend(top_dict[source][key])
    min_time = min(time_list)
    return min_time


def triple_diffusing_method(triples, diffusive_source):
    # 输入三元组的集合和扩散源节点，输出所有节点的扩散到达时间，同时记录下来所有节点的传播父节点
    # output { arrival_time：所有节点到达时间的字典 parent_node_dict：所有节点的传播父节点 }
    print("start running...please wait...")
    top_dict = {}
    parent_node_dict = dict()
    for triple in triples:  # 先确定节点
        top_dict[triple[0]] = {}
        top_dict[triple[1]] = {}
    for triple in triples:  # 再确定连边
        top_dict[triple[0]][triple[1]] = []
        top_dict[triple[1]][triple[0]] = []
    for triple in triples:  # 最后计入到达时间
        top_dict[triple[0]][triple[1]].append(triple[2])
        top_dict[triple[1]][triple[0]].append(triple[2])
    # print(find_diffusion_sourve_arrival_time(top_dict,diffusive_source))
    # 生成的top_dict包含所有的时间和节点信息
    arrival_time = dict()
    arrival_time[diffusive_source] = find_diffusion_sourve_arrival_time(top_dict, diffusive_source)
    diffuse(top_dict, diffusive_source, arrival_time, parent_node_dict)
    for key in top_dict.keys():
        if not arrival_time.__contains__(key):
            arrival_time[key] = -1
    return arrival_time, parent_node_dict


def diffuse(top_dict, source_node, arrival_time, parent_node_dict):
    # 从源节点开始，遍历其相邻的节点，如果相邻节点的到达时间中，有比扩散而来的节点的到达时间大的，就可以更新该节点的到达时间，
    # 并递归地遍历该节点的下一个节点，如果到达没有被更新，也就不需要再继续递归
    # print("now: ", source_node, "  arrival time: ", arrival_time[source_node])
    for key, value in top_dict[source_node].items():
        # print(key, value)
        for time in value:
            if time >= arrival_time[source_node]:
                if arrival_time.__contains__(key):
                    if time < arrival_time[key]:
                        arrival_time[key] = time
                        parent_node_dict[key] = source_node
                        diffuse(top_dict, key, arrival_time, parent_node_dict)
                else:
                    arrival_time[key] = time
                    parent_node_dict[key] = source_node
                    diffuse(top_dict, key, arrival_time, parent_node_dict)

    return True


def find_path(parent_node_dict, source):
    # 计算所有节点的传播路径
    # input { parent_node_dict：所有节点的传播父节点的字典 source：扩散源节点 }
    # output { diffuse_path：所有节点传播路径的字典 }
    parent_node_dict[source] = -2
    diffuse_path = dict()
    temp_value = None
    temp_key = None
    for key, value in parent_node_dict.items():
        diffuse_path[key] = list()
        diffuse_path[key].append(key)
        temp_key = key
        temp_value = value
        while temp_value != -2:
            diffuse_path[key].append(temp_value)
            temp_key = temp_value
            temp_value = parent_node_dict[temp_value]
    for key, value in diffuse_path.items():
        diffuse_path[key] = diffuse_path[key][::-1]
    return diffuse_path

def find_diffusion_source_arrival_time_in_method2(sorted_triples, diffusive_source, directed=False):
    if directed == False:
        for item in sorted_triples:
            if item[0] == diffusive_source or item[1] == diffusive_source:
                return item[2]
    else:
        for item in sorted_triples:
            if item[0] == diffusive_source:
                return item[2]


def triple_diffusing_method2_oneStepEachTimestamp(sorted_triples, diffusive_source,beta=1,r=0):
    dat = dict()
    infected_dict = dict()  # the dict consists of nodes which have been infected in the network.This dict is just an identification of infected nodes, and it's not used in diffusion process.
    tempTimestamp_triples_list = list()  # This list is the set of all the edges at current time.
    temp_time = find_diffusion_source_arrival_time_in_method2(sorted_triples,
                                                              diffusive_source)  # A mark of current time.
    dat[diffusive_source] = temp_time
    count = 0
    infected_dict[diffusive_source] = -2
    triples_length = len(sorted_triples)
    if temp_time > sorted_triples[0][2]:
        while sorted_triples[count][2] < temp_time:
            count += 1
    # start from the time when the source node was infected.
    while count < triples_length:
        while count < triples_length and temp_time == sorted_triples[count][2]:
            tempTimestamp_triples_list.append([sorted_triples[count][0], sorted_triples[count][1]])
            count += 1
        diffuse_in_method2_oneStep(tempTimestamp_triples_list, infected_dict, dat, temp_time,beta)
        tempTimestamp_triples_list = list()
        if count < triples_length:
            temp_time = sorted_triples[count][2]
    return dat, infected_dict


def diffuse_in_method2_oneStep(tempTimestamp_triples_list, infected_dict, dat, temp_time,beta):
    temp_infected_dict = dict()  # Add all the items in temp_dict to infected_dict and dat at the last step.
    for item in tempTimestamp_triples_list:
        has_v1 = item[0] in infected_dict
        has_v2 = item[1] in infected_dict
        if (not has_v1) and has_v2:
            if random.random() < beta:
                temp_infected_dict[item[0]] = item[1]
                dat[item[0]] = temp_time
        if has_v1 and (not has_v2):
            if random.random() < beta:
                temp_infected_dict[item[1]] = item[0]
                dat[item[1]] = temp_time
    infected_dict.update(temp_infected_dict)


def diffuse_in_method2_multiStep(tempTimestamp_triples_list, infected_dict, dat, temp_time):
    temp_infected_dict = dict()  # Add all the items in temp_dict to infected_dict and dat at the last step.
    flag_count = 0
    flag = 0
    while flag_count != 0 or flag == 0:
        # If there is no update in the flag list, there is no need to repeat the loop again.
        temp_infected_dict = dict()
        flag = 1
        len_list = len(tempTimestamp_triples_list)
        i = 0
        flag_count = 0
        while i < len_list:
            has_v1 = tempTimestamp_triples_list[i][0] in infected_dict
            has_v2 = tempTimestamp_triples_list[i][1] in infected_dict
            if (not has_v1) and has_v2:
                temp_infected_dict[tempTimestamp_triples_list[i][0]] = tempTimestamp_triples_list[i][1]
                dat[tempTimestamp_triples_list[i][0]] = temp_time
                tempTimestamp_triples_list.pop(i)
                flag_count += 1
                len_list -= 1
            if has_v1 and (not has_v2):
                temp_infected_dict[tempTimestamp_triples_list[i][1]] = tempTimestamp_triples_list[i][0]
                dat[tempTimestamp_triples_list[i][1]] = temp_time
                tempTimestamp_triples_list.pop(i)
                flag_count += 1
                len_list -= 1
            i += 1
        infected_dict.update(temp_infected_dict)


def triple_diffusing_method2_multiStepEachTimestamp(sorted_triples, diffusive_source):
    dat = dict()  # the first arrival time of each node.
    infected_dict = dict()  # the dict consists of nodes which have been infected in the network.This dict is just an identification of infected nodes, and it's not used in diffusion process.
    tempTimestamp_triples_list = list()  # This list contains all the edges at current time.
    temp_time = find_diffusion_source_arrival_time_in_method2(sorted_triples,
                                                              diffusive_source)  # A mark of current time.
    dat[diffusive_source] = temp_time
    count = 0
    infected_dict[diffusive_source] = -2
    triples_length = len(sorted_triples)
    if temp_time > sorted_triples[0][2]:
        while sorted_triples[count][2] < temp_time:
            count += 1
    while count < triples_length:
        while count < triples_length and temp_time == sorted_triples[count][2]:
            tempTimestamp_triples_list.append([sorted_triples[count][0], sorted_triples[count][1]])
            count += 1
        diffuse_in_method2_multiStep(tempTimestamp_triples_list, infected_dict, dat, temp_time)
        tempTimestamp_triples_list = list()
        if count < triples_length:
            temp_time = sorted_triples[count][2]
    return dat, infected_dict


def diffusion_class(dat, source):
    sorted_dat = sorted(dat.items(), key=lambda x: x[1])
    sorted_dict = dict()
    count = 0
    time = dat[source]
    for tuple in sorted_dat:
        if tuple[1] == time:
            if sorted_dict.__contains__(count):
                sorted_dict[count].append(tuple[0])
            else:
                sorted_dict[count] = list()
                sorted_dict[count].append(tuple[0])
        else:
            time = tuple[1]
            count += 1
            sorted_dict[count] = list()
            sorted_dict[count].append(tuple[0])

    return sorted_dict


def update_state():
    global refresh_count
    newly_infected.a = False
    # visit the nodes in random order
    vs = list(diffusion_tree_in_graph_tool.vertices())
    dict_length = len(sorted_diffusion_node_dict)
    time.sleep(0.05)
    if refresh_count == dict_length - 1:
        for v in diffusion_tree_in_graph_tool.vertices():
            state[v] =UNINFECTED
            refresh_count = 0
            continue
    for li in sorted_diffusion_node_dict[refresh_count]:
        state[diffusion_tree_in_graph_tool.vertex(li)] = INFECTED
    refresh_count += 1
    # Filter out the recovered vertices
    # g.set_vertex_filter(removed, inverted=True)

    # The following will force the re-drawing of the graph, and issue a
    # re-drawing of the GTK window.
    diffusion_tree_in_graph_tool.set_vertex_filter(removed, inverted=True)
    win.graph.regenerate_surface()
    win.graph.queue_draw()

    # if doing an offscreen animation, dump frame to disk
    if offscreen:
        global count
        pixbuf = win.get_pixbuf()
        pixbuf.savev(r'./frames/sirs%06d.png' % count, 'png', [], [])
        if count > max_count:
            sys.exit(0)
        count += 1

    # We need to return True so that the main loop will call this function more
    # than once.
    return True


source = 17

'''
dat,parent_node_dict= triple_diffusing_method(a,source)
dat_path = find_path(parent_node_dict,source)
'''

work_space_data = read_workspace_dataset("data/dataset_workspace.dat")
dat, parent_node_dict =triple_diffusing_method2_oneStepEachTimestamp(work_space_data, source, beta=0.7)
dat_path = find_path(parent_node_dict, source)


'''
high_school_2013_data = read_high_school_2013_dataset("data/dataset_high_school_2013.csv")
dat, parent_node_dict = triple_diffusing_method(high_school_2013_data,source)
dat_path = find_path(parent_node_dict,source)
'''


sorted_diffusion_node_dict = diffusion_class(dat,source)
diffusion_tree_in_graph_tool = Graph(directed=True)
diffusion_tree_in_graph_tool.add_vertex(10000)
for key, value in dat_path.items():
    for i in range(len(value) - 1):
        if diffusion_tree_in_graph_tool.edge(value[i], value[i + 1]) not in diffusion_tree_in_graph_tool.edges():
            diffusion_tree_in_graph_tool.add_edge(value[i], value[i + 1])

print(diffusion_tree_in_graph_tool)

removed = diffusion_tree_in_graph_tool.new_vertex_property("bool")
removed.a = False
node_count = 0
for v in diffusion_tree_in_graph_tool.vertices():
    if v.out_degree() == 0 and v.in_degree() == 0:
        node_count +=1
        removed[v] = True
diffusion_tree_in_graph_tool.set_vertex_filter(removed, inverted=True)

UNINFECTED = [1, 1, 1, 1]  # White color  suspected
INFECTED = [0, 0, 0, 1]  # Black color  infected
pos = sfdp_layout(diffusion_tree_in_graph_tool)

state = diffusion_tree_in_graph_tool.new_vertex_property("vector<double>")
newly_infected = diffusion_tree_in_graph_tool.new_vertex_property("bool")
for v in diffusion_tree_in_graph_tool.vertices():
    state[v] = UNINFECTED


refresh_count = 0
offscreen = sys.argv[1] == "offscreen" if len(sys.argv) > 1 else False
max_count = 500
if offscreen and not os.path.exists("./frames"):
    os.mkdir("./frames")
if not offscreen:
    win = GraphWindow(diffusion_tree_in_graph_tool, pos, geometry=(500, 400),
                      edge_color=[0.6, 0.6, 0.6, 1],
                      vertex_fill_color=state,
                      vertex_halo=newly_infected,
                      vertex_halo_color=[0.8, 0, 0, 0.6])
else:
    count = 0
    win = Gtk.OffscreenWindow()
    win.set_default_size(500, 400)
    win.graph = GraphWidget(diffusion_tree_in_graph_tool, pos,
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
