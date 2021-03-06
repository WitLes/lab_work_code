#SIS模型在真实三元组上的传播过程演示
#（workspace data）
之前我实现的动画，是考虑SI模型的情况，因此每个节点只有一个到达时间和到达路径，可以根据到达路径恢复出感染的扩散树。很方便地就可以实现动画演示。

这一次我使用SIS模型作为演示，每个节点可能会被多次感染，因此考虑了以下问题：

1. 另外使用多个字典储存网络中所有节点的状态信息。
2. 每一步只考虑一层的扩散，即不能再同一时刻纵深地感染多个节点。（我也写出了可以同时扩散多个节点的算法）
3. 要记录下来每个节点（或连边）被感染的次数，以表征该节点（或连边）在传播过程中的关键性。我用格外的一个字典记录了每个节点被感染的次数，跑出的结果显示，度越大的节点，被感染的次数越多，符合实际情况。（如果需要，可以再做出统计图，但有些麻烦就先没有做，结果是有的）
4. 之前考虑的扩散动画是以每个扩散时间发生的时间切片组成的。（只有在扩散发生时，才考虑显示这一帧动画，所以实际扩散的时间线和动画演示的时间线并不是成比例的）这次因为要考虑恢复率，所有被感染的节点在每一个时刻都可能恢复，所以我严格按照时间线进行SIS过程的模拟，动画中节点的被感染时间，恢复时间或再感染时间，都与真实的时间一一对应，只是进行了比例缩放。
5. 由于程序运行速度的限制，在python的graph_tool中，因为加入了算法，视频里的速度是动画运行的最快的传播速度了。。（可能我用了虚拟机的原因，但不是算法的问题，是graph-tool内的视频帧更新速度的问题，只跑算法的话是1s内的）我只是截取了其中的两个时间片段做成视频。在这个视频里面感染率是1，恢复率是0.001（因为每一个时刻下，每一个被感染的节点都可能恢复，而实际时效网络的时间线很长，所以这个恢复率需要很小，保证传播的进行）


关于四元组及相关分布的模拟，算法的思路和三元组一样，同样可以复现出来，但考虑到时间原因，我就先做了三元组。我是想先继续进行论文的复现。我打算在11月份先详细地把大论文按章节看一遍，先理清楚算法的思路，虽然不能完全理清数学细节，但中间有问题就可以写成报告反馈给你，如何？编程的任务也同时进行。

（前一段时间李老师给我挖了一个坑（剪视频），所以耽误的时间比较多 = =）