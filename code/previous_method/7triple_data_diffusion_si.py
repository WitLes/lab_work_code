# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Aquamarine GY 2017.10.30

'''
Structure of top_dict:{node_1:{node_2:[Tn],node_3:[Tn]}, node2:{node1:[Tn],},node_3:{Tn}}
'''
import networkx as nx
import matplotlib.pyplot as plt
import random
import numpy as np

a = np.array([[0, 1, 1], [1, 2, 2], [2, 3, 1], [2, 3, 3], [3, 4, 5], [1, 4, 2], [4, 5, 3], [1, 5, 2], [5, 6, 2]])
b = [[0, 5, 1], [5, 6, 2], [6, 4, 1], [6, 4, 3], [4, 2, 5], [5, 2, 2], [5, 1, 2], [1, 2, 3], [1, 3, 3]]


def read_workspace_dataset(file_name):
    # 将workspace数据集的数据转化成(u,v,t)的三元组形式
    file = open(file_name, "r")
    triple_dataspace_list = []
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


def find_diffusion_sourve_arrival_time_in_method1(top_dict, source):
    # 找到三元组生成的字典中，源节点的到达时间，也就是源节点第一次与别的节点交互的时间
    time_list = list()
    for key in top_dict[source].keys():
        time_list.extend(top_dict[source][key])
    min_time = min(time_list)
    return min_time


def triple_diffusing_method1(triples, diffusive_source):
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
    arrival_time[diffusive_source] = find_diffusion_sourve_arrival_time_in_method1(top_dict, diffusive_source)
    diffuse_in_method1(top_dict, diffusive_source, arrival_time, parent_node_dict)
    for key in top_dict.keys():
        if not arrival_time.__contains__(key):
            arrival_time[key] = -1
    return arrival_time, parent_node_dict


def diffuse_in_method1(top_dict, source_node, arrival_time, parent_node_dict):
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
                        diffuse_in_method1(top_dict, key, arrival_time, parent_node_dict)
                else:
                    arrival_time[key] = time
                    parent_node_dict[key] = source_node
                    diffuse_in_method1(top_dict, key, arrival_time, parent_node_dict)

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
    # This function uses the time consequence of triples to calculate the arrival time and path with out generating a graph(dict).
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
        diffuse_in_method2_oneStep(tempTimestamp_triples_list, infected_dict, dat, temp_time,beta,r)
        tempTimestamp_triples_list = list()
        if count < triples_length:
            temp_time = sorted_triples[count][2]
    return dat, infected_dict


def diffuse_in_method2_oneStep(tempTimestamp_triples_list, infected_dict, dat, temp_time,beta,r):
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
    infected_list = [[key, value] for key, value in infected_dict.items()]
    for item in infected_list:
        if random.random() < r:
            infected_dict.pop(item[0])
    infected_dict.update(temp_infected_dict)
   # print(infected_dict)


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


def is_two_dat_equal(dat1, dat2):
    list_diff = list()
    for key, value in dat1.items():
        if not dat2[key] == value:
            list_diff.append([key, value, dat2[key]])
    if len(list_diff) == 0:
        return True, list_diff
    else:
        return False, list_diff


def is_two_dat_path_equal(dat_path1, dat_path2):
    flag = True
    list_diff = list()
    for key, value in dat_path1.items():
        if not dat_path2[key] == value:
            list_diff.append([key, value, dat_path2[key]])
            flag = False
    return flag, list_diff


source = 1

'''
dat,parent_node_dict= triple_diffusing_method(a,source)
dat_path = find_path(parent_node_dict,source)
'''

'''
data = read_workspace_dataset("data/dataset_workspace.dat")
dat, parent_node_dict = triple_diffusing_method2_multiStepEachTimestamp(data, source)
dat_path = find_path(parent_node_dict, source)
'''

data = read_high_school_2013_dataset("../data/high2013.txt")
dat, parent_node_dict = triple_diffusing_method1(data,source)
dat_path = find_path(parent_node_dict,source)

print(dat)

print(find_diffusion_source_arrival_time_in_method2(data, source))

a, dat_parent = triple_diffusing_method2_oneStepEachTimestamp(data, source, beta=1,r=0)
print(a)


