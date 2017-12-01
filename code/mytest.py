import numpy as np
import networkx as nx

a = []
for i in range(10):
	np.random.seed(i)
	print(np.random.randint(100))
	a.append([np.random.normal() for i in range(5)])
print(a)