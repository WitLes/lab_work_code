# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Aquamarine GY 2017.10.6

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


recovery_rate = 0
diffusive_rate = 0.8
source = 17

'''
dat,parent_node_dict= triple_diffusing_method(a,source)
dat_path = find_path(parent_node_dict,source)
'''

work_space_data = read_workspace_dataset("data/dataset_workspace.dat")

infected_node_dict_and_parent = dict()
infected_node_dict_and_time = dict()
infected_node_dict_and_parent[source] = -2
infected_node_dict_and_time[source] = 0
for triple in work_space_data:
    a = infected_node_dict_and_parent.__contains__(triple[0])
    b = infected_node_dict_and_parent.__contains__(triple[1])
    if a & (not b):
        if random.random() > diffusive_rate:
            infected_node_dict_and_parent[triple[1]] = triple[0]
            infected_node_dict_and_time[triple[1]] = triple[2]
    if b & (not a):
        if random.random() > diffusive_rate:
            infected_node_dict_and_parent[triple[0]] = triple[1]
            infected_node_dict_and_time[triple[0]] = triple[2]
dat_path = find_path(infected_node_dict_and_parent, source)
print(infected_node_dict_and_parent)
print(dat_path)


    

