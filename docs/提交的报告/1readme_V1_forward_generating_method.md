# 正向生成算法
1.我自己随便画了一个拓扑结构,每一层的节点都连接下一层的所有节点

该图为有向图，例如{0，1}连边表示从0->1的有向边（大头为指向）

![avatar](Figure_2.png)


2.初始化每条边上的WTD为Gamma(2，3)分布


3.对图中所有的边依次遍历，生成每条边上的等待时间tao

4.结果生成了一个有向有权图，计算从0节点出发的单元最短路径，得到所有节点的到达时间


# 以其中一次run为例
###结果1
	下面是各个节点的最短路径，从0开始，到19结束

	[0]
	[0, 1]
	[0, 2]
	[0, 3]
	[0, 4]
	[0, 3, 5]
	[0, 1, 6]
	[0, 3, 7]
	[0, 3, 8]
	[0, 3, 8, 9]
	[0, 3, 8, 10]
	[0, 3, 7, 11]
	[0, 1, 6, 12]
	[0, 3, 7, 13]
	[0, 3, 7, 14]
	[0, 3, 8, 15]
	[0, 3, 7, 11, 16]
	[0, 3, 8, 9, 17]
	[0, 3, 8, 15, 18]
	[0, 3, 7, 11, 19]
###结果2
下面是各个节点的最短路径长度，从0节点开始，依次是所有节点的到达顺序和到达时间

	{0: 0,
	 3: 1.1609717388133962, 
	 7: 2.0388963241155587,
	 8: 2.2334418923747266, 
	 4: 2.7953286376664619, 
	 9: 2.8902439058736169, 
	 13: 2.9468467737465134, 
	 11: 3.1960944631236514, 
	 10: 3.2517739839294375, 
	 17: 3.8615848675726667, 
	 14: 3.9643765289793063, 
	 1: 4.116466501980816, 
	 15: 4.7536289155441196, 
	 19: 4.9699009186981717, 
	 5: 5.175612463580185, 
	 18: 5.1991985372319638, 
	 16: 5.2647997911200068, 
	 6: 5.3792855344856427, 
	 2: 7.3363245754165218, 
	 12: 7.8775652769271147}
###结果3
对应的扩散路径如下

![avatar](Figure_1.png)
