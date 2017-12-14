import numpy as np
import networkx as nx
import sympy as sp
from sympy.abc import a,x,y

er_graph = nx.erdos_renyi_graph(3,0.5, seed=None, directed=True)
print(type(er_graph))
graph = nx.DiGraph()
print(type(graph))
graph.add_edge(0,1,weight=1)
graph.add_edge(1,2,weight=1)
graph.add_edge(0,2,weight=2)
DAT_PATH = nx.shortest_path(graph, source=0, weight="weight")
DAT = nx.shortest_path_length(graph, source=0, weight="weight")
print(DAT_PATH)
print(DAT)

for edge in graph.edges():
	print(edge)
print("1")

if (1,0) in graph.edges():
	print("yes")
else:
	print("no")