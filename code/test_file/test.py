import networkx as nx
import numpy as np
import matplotlib.pyplot as plt




graph = nx.Graph()
graph.add_nodes_from(list(range(100)))
grid = nx.grid_2d_graph(10,10,create_using=graph)
nx.draw(grid)
plt.show()

file = open("grid36.txt","w")

for edge in grid.edges():
    node1 = edge[0][0]*10 +edge[0][1]
    node2 = edge[1][0]*10+edge[1][1]
    file.write(str(node1+1)+" "+str(node2+1)+"\n")

file.close()
