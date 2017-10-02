import networkx as nx
import matplotlib.pyplot as plt
import random
import numpy as np

a = np.array([[0, 1, 1], [1, 2, 2], [2, 3, 1], [2, 3, 3], [3, 4, 5], [1, 4, 2], [4, 5, 3], [1, 5, 2], [5, 6, 3]])
b = [[0, 5, 1], [5, 6, 2], [6, 4, 1], [6, 4, 3], [4, 2, 5], [5, 2, 2], [5, 1, 2], [1, 2, 3], [1, 3, 3]]


def read_workspace_dataset(file_name):
    file = open(file_name,"r")
    triple_dataspace_list = []
    for line in file.readlines():
        triple_dataspace_list.append([int(x) for x in line.split(' ')])
    standard_set = np.array(triple_dataspace_list)
    standard_set = np.column_stack((standard_set[:,1],standard_set[:,2],standard_set[:,0]))
    return standard_set


def read_high_school_dataset(file_name):
    file = open(file_name, "r")
    triple_dataspace_list = []
    count = 0
    for line in file.readlines():
        triple_dataspace_list.append([int(x) for x in line.split(' ')[0:3]])
    # print(triple_dataspace_list)
    standard_set = np.array(triple_dataspace_list)
    standard_set = np.column_stack((standard_set[:, 1], standard_set[:, 2], standard_set[:, 0]))
    return standard_set
    pass


def triple_diffusing_method(triples,diffusive_source):
    print("start running...please wait...")
    top_dict = {}
    for triple in triples:   # 先确定节点
        top_dict[triple[0]] = {}
        top_dict[triple[1]] = {}
    for triple in triples:   # 再确定连边
        top_dict[triple[0]][triple[1]] = []
        top_dict[triple[1]][triple[0]] = []
    for triple in triples:   # 最后计入到达时间
        top_dict[triple[0]][triple[1]].append(triple[2])
        top_dict[triple[1]][triple[0]].append(triple[2])
    arrival_time = dict()
    arrival_time[diffusive_source] = 0
    diffuse(top_dict, diffusive_source, arrival_time)
    for key in top_dict.keys():
        if not arrival_time.__contains__(key):
            arrival_time[key] = -1
    return arrival_time


def diffuse(top_dict, source_node, arrival_time):
    # 从源节点开始，遍历其相邻的节点，如果相邻节点的到达时间中，有比扩散而来的节点的到达时间大的，就可以更新该节点的到达时间，
    # 并递归地遍历该节点的下一个节点，如果到达没有被更新，也就不需要再继续递归
    # print("now: ", source_node, "  arrival time: ", arrival_time[source_node])
    for key, value in top_dict[source_node].items():
        # print(key, value)
        for time in value:
            if time > arrival_time[source_node]:
                if arrival_time.__contains__(key):
                    if time < arrival_time[key]:
                        arrival_time[key] = time
                        diffuse(top_dict, key, arrival_time)
                else:
                    arrival_time[key] = time
                    diffuse(top_dict, key, arrival_time)

    return 0


# work_space_data = read_workspace_dataset("dataset_workspace.dat")
high_school_2013_data = read_high_school_dataset("dataset_high_school_2013.csv")

dat = triple_diffusing_method(high_school_2013_data,257)

print(dat)
