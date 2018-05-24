from algorithm_realization.stn_reconstruction_lib import *


nodes = [100*i for i in range(1,11)]
nodes.append(1500)
nodes.append(2000)
edges = [269,622,999,1358,1958,2105,2424,2762,2838,2950,5680,7952]
C = [0.426, 0.282, 0.214, 0.174,0.16, 0.12, 0.1, 0.098, 0.08, 0.06,0.055,0.052]

plt.figure(figsize=(8, 5))
plt.style.use('ggplot')
# plt.title("", fontsize=10)
plt.xlabel("N (nodes)",fontsize=10)
plt.ylabel("C = M / N",fontsize=10)
plt.xlim(0,2150)
plt.ylim(0,1)
plt.plot(nodes, C, ".k-", linewidth=0.5, markersize=8,markeredgecolor="crimson")
plt.savefig("../figure/"+"large_scale_test"+".svg", format="svg")
plt.savefig("../figure/"+"large_scale_test"+".png", dpi=600, format="png")
plt.savefig("../figure/"+"large_scale_test"+".eps", format="eps")
plt.show()