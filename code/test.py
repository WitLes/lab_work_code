import networkx as nx  
import matplotlib.pyplot as plt  
from random import randint  

G=nx.DiGraph()  
"""
G.add_nodes_from([0,1,2,3,4])
G.add_weighted_edges_from([(0,1,1.0),(1,2,7.5),(0,2,5.5),(1,3,4.0)])#添加边的权值  
G.add_weighted_edges_from([[0,1,2.0],])
print(G.neighbors(1))
for (i,j,k) in G.edges(1,data=True):
	print(i,"->",j,": ",k)
for (i,j,k) in G.edges(0,data=True):
	print(i,"->",j,": ",k)
"""
"""	
nx.draw(G,pos=nx.circular_layout(G),with_labels=True,node_color='red')#按参数构图  
plt.axis('off')
plt.show()#显示图像
"""
SC_N = nx.scale_free_graph(100)
nx.draw(SC_N,pos=nx.spring_layout(SC_N))
plt.show()