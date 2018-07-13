from algorithm_realization.stn_reconstruction_lib import *

x=[100,200,300,400,500,600,700,800,900,1000]
C = [0.426,0.282,0.214,0.174,0.1599,0.12,0.1,0.098,0.08,0.06]
all_font_size = 13
plt.figure(figsize=(6,5))
plt.xlabel("N(number of nodes)",fontsize=16)
plt.ylabel("C(data)",fontsize=16)
plt.xticks(fontsize=all_font_size)
plt.yticks(fontsize=all_font_size)
plt.ylim(0,0.5)
plt.plot(x, C, "bs-", linewidth=1, markersize=7)
plt.subplots_adjust(left=0.15,right=0.95)
plt.savefig("../figure/"+"large_scale_test"+".svg", format="svg")
plt.savefig("../figure/"+"large_scale_test"+".png", dpi=600, format="png")
plt.savefig("../figure/"+"large_scale_test"+".eps", format="eps")
plt.show()