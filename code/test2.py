import numpy as np
import networkx as nx

with open("./data/dolphins.txt",encoding="utf8") as data:
    data_list = []
    for line in data.readlines():
        line = line.strip("\n")
        line = line.split(" ")
        line = [int(item) for item in line]
        line.append({"weight":1})
        data_list.append(line)
    print(data_list)
    data.close()

