# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Aquamarine GY 2017.10.6


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


source = 1

'''
dat,parent_node_dict= triple_diffusing_method(a,source)
dat_path = find_path(parent_node_dict,source)
'''
'''
work_space_data = read_workspace_dataset("data/dataset_workspace.dat")
dat, parent_node_dict = triple_diffusing_method(work_space_data, source)
dat_path = find_path(parent_node_dict, source)
'''

high_school_2013_data = read_high_school_2013_dataset("data/dataset_high_school_2013.csv")
dat, parent_node_dict = triple_diffusing_method(high_school_2013_data,source)
dat_path = find_path(parent_node_dict,source)
print(len(dat))

print("diffusion arrival time: ", dat)
print("parent node of every single node: ", parent_node_dict)
print("diffusion path: ", dat_path)
count = 0
for key, value in dat_path.items():
    print(key, ":", value)
