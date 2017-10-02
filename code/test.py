import numpy as np

def read_workspace_dataset(file_name):
    file = open(file_name,"r")
    triple_dataspace_list = []
    count = 0
    for line in file.readlines():
        triple_dataspace_list.append([int(x) for x in line.split(' ')])
    standard_set = np.array(triple_dataspace_list)
    standard_set = np.column_stack((standard_set[:,1],standard_set[:,2],standard_set[:,0]))
    return standard_set


data = read_workspace_dataset("dataset_workspace.dat")
print(data)