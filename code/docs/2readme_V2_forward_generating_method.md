# 正向生成算法 V2


### 1.拓扑结构生成
在V2程序中，分别使用了BA网络，GNP网络，ER网络生成我们需要的拓扑结构，对应的生成函数为


```python
BA = nx.random_graphs.barabasi_albert_graph(node_number, linking_number)
	# node_number是节点个数，linking_number是新节点加入图中时与其他节点的连边个数
GNP = nx.fast_gnp_random_graph(node_number, probability)
	# probability 加入边的连接概率
ER = nx.erdos_renyi_graph(node_number, probability)
```

### 2.生成对应的有向有权结构，WTD使用gamma分布
### 3.计算单源最短路径

<br>
# 主要代码
```python
sample_graph = topology_generator(0)  
	# 生成一个无向无权的网络样本
	# 0是BA，1是GNP，2是ER
G = nx.DiGraph()
	# 初始化
G = directed_and_weighted_graph_generator(G,sample_graph)  
	# 将样本重构成有向有权图
draw_graph(G,type=0)
DAT_PATH, DAT = calculate_diffusive_arrival_times(G,diffusive_source=0)
	# 扩散过程的可视化
draw_diffusion_tree(DAT_PATH)  # 绘制扩散路径图
plt.title("diffusive source:0")
plt.show()
```
<br>
# RUN
###example1
	EA graph
	node number 20
	每个节点可以从2个不同的节点扩散而来
	布局 spring_layout
	扩散源 0号节点
![](/Users/gaoyuan/Desktop/Figure_2.png)

###example2
	EA graph
	node number 100
	每个节点可以从3个不同节点扩散而来
	扩散源 6号节点
![](/Users/gaoyuan/Desktop/Figure_3.png)

###example3
	ER graph
	node number 20
	source 3号节点
![](/Users/gaoyuan/Desktop/Figure_4.png)


# 问题
1.在有些图中，因为连边的有向性，造成从部分扩散源出发时，某一部分的节点可能是不可达的。比如在example3中，虽然有30个节点，但从节点3出发的传播路径只能到达其中的一半节点，并没有可以到达所有节点的有向边。


