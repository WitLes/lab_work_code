# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Aquamarine GY 2017.11.1

'''
Structure of top_dict:{node_1:{node_2:[Tn],node_3:[Tn]}, node2:{node1:[Tn],},node_3:{Tn}}
'''
import networkx as nx
import matplotlib.pyplot as plt
import random
import numpy as np


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


def triple_diffusing_method2_oneStepEachTimestamp(sorted_triples, diffusive_source, beta=1, r=0):
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
        diffuse_in_method2_oneStep(tempTimestamp_triples_list, infected_dict, dat, temp_time, beta, r)
        tempTimestamp_triples_list = list()
        if count < triples_length:
            temp_time = sorted_triples[count][2]
    return dat, infected_dict


def diffuse_in_method2_oneStep(tempTimestamp_triples_list, infected_dict, dat, temp_time, beta, r):
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


def duplicate_triples(sorted_triples, repeat=1):
    sorted_triples = list(sorted_triples)
    if repeat == 1:
        return sorted_triples
    else:
        pass


def sis_simulation_method_oneStep(sorted_triples, diffusive_source, min_time_interval, beta=1, recover_rate=0.0001,
                                  repeat=1):
    sorted_triples = duplicate_triples(sorted_triples, repeat=1)
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
                continue
        if (item[1] in infected_node_dict) and (item[0] not in infected_node_dict):
            if random.random() < beta:
                temp_record_dict[item[0]] = item[2]
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
            infected_node_dict.pop(key)
            print("recover: ", recover_node)


data = read_workspace_dataset("../data/workplace.txt")

sis_simulation_method_oneStep(data, 17, 20)
