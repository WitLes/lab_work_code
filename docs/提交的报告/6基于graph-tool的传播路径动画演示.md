#7基于graph-tool的传播路径动画演示

在上一个程序中我们得到了：

1. 每个节点的到达时间组成的字典dat
2. 每个节点的传播父节点parent_node_dict
3. 每个节点的时效传播路径

为了实现扩散的特效，我们需要：

1. 生成由扩散路径组成的扩散树
2. 每个节点的到达时间按先后顺序排列

##1.扩散树
因为已经有了所有节点的扩散路径组成的字典dat_path，我们只需要使用graph_tool生成一个从dat_path中提取的没有重复边的图即可。

因此生成扩散树只需要遍历dat_path，将连边添加到扩散树的图中，重复连边只添加一次即可。

##2.到达时间的顺序排列

因为dat是一个字典，并不是有序的，先要生成一个有序的二元数组，表面每个节点的到达顺序和到达时间。

然后将相同到达时间的节点放在同一组中，形成若干个组，每一次时间更新时，感染一个组的节点即可。
