from algorithm_realization.stn_reconstruction_lib import *
import scipy.interpolate as interpolate

mode_list = ["ER","BA","SW"]
distribution_list = ["gaussian","uniform","weibull", "gumbel","exponential","beta"]
mode_select = 0
distribution_select = 0
C=10

axis_font_size = 13
legend_font_size = 13
axis_legend_font_size = 16
curve_line_width = 2
step_line_width = 1

demo_graph, node_number, edge_number = networkx_graph_generator(mode=mode_list[mode_select])

dats, dat_path = dats_generator(demo_graph, mode=distribution_list[distribution_select], dat_number=C*node_number, seed=False)
l_t = 5
delta = 0.01
all_edge_duv_time_list = list()
all_not_edge_duv_time_list = list()
for i in range(node_number):
    for j in range(i + 1, node_number):
        if (i, j) in demo_graph.edges() or (j, i) in demo_graph.edges():
            for dat in dats:
                all_edge_duv_time_list.append(abs(dat[i] - dat[j]))
            break
    break


for i in range(node_number):
    for j in range(i + 1, node_number):
        if (i, j) not in demo_graph.edges() and (j, i) not in demo_graph.edges():
            for dat in dats:
                all_not_edge_duv_time_list.append(abs(dat[i] - dat[j]))
            break
    break
print(node_number)
print(len(all_edge_duv_time_list))
print(len(all_not_edge_duv_time_list))
x_range = [float(i) * delta * 10 for i in range(int((l_t / delta) / 10) + 1)]
x_range_original = [i for i in range(int(l_t/delta))]

pdf_50 = pdf_recover_from_list(all_edge_duv_time_list)
pdf_50 = normalize_pdf(pdf_50)
sf_50 = normalize_sf(pdf2sf_using_density(pdf_50))

pdf_not_50 = pdf_recover_from_list(all_not_edge_duv_time_list)
pdf_not_50 = normalize_pdf(pdf_not_50)
sf_not_50 = normalize_sf(pdf2sf_using_density(pdf_not_50))

original_pdf = pdf_generator(mode=distribution_list[distribution_select])

original_pdf = normalize_pdf(original_pdf)
original_sf = normalize_sf(pdf2sf_using_density(original_pdf))

nx = np.linspace(0, 5, 500)
f_pdf = interpolate.interp1d(x_range, pdf_50)
pdf_500 = f_pdf(nx)
f_sf = interpolate.interp1d(x_range, sf_50)
sf_500 = f_sf(nx)

f_pdf_not = interpolate.interp1d(x_range, pdf_not_50)
pdf_not_500 = f_pdf(nx)
f_sf = interpolate.interp1d(x_range, sf_not_50)
sf_not_500 = f_sf(nx)

pdf_500 = normalize_pdf(pdf_500)
sf_500 = normalize_sf(sf_500)
pdf_not_500 = normalize_pdf(pdf_not_500)
sf_not_500 = normalize_sf(sf_not_500)


plt.figure(figsize=(14, 4))
plt.subplot(1,3,1)
plt.xlabel(r'$\tau$', fontsize=axis_legend_font_size)
plt.ylabel("Probability",fontsize=axis_legend_font_size)
plt.step(x_range,sf_50,'b',linewidth=step_line_width,alpha=0.5)
plt.step(x_range,sf_not_50,'r',linewidth=step_line_width,alpha=0.5)
plt.plot(nx, sf_500, 'b--',linewidth=curve_line_width,label=r'$\hat{\Phi}_{(u,v)\in\mathcal{G}}(\tau)$')
plt.plot(nx, sf_not_500, 'r--', linewidth=curve_line_width, label=r'$\hat{\Phi}_{(u,v)\notin\mathcal{G}}(\tau)$')
plt.plot(nx, original_sf, 'g-', linewidth=curve_line_width,label=r'$\Phi_{(u,v)\in\mathcal{G}}(\tau)$')
plt.legend(fontsize=legend_font_size)
plt.xticks(fontsize=axis_font_size)
plt.yticks(fontsize=axis_font_size)


mode_list = ["ER","BA","SW"]
distribution_list = ["gaussian","uniform","weibull", "gumbel","exponential","beta"]
mode_select = 0
distribution_select = 1
C=10



demo_graph, node_number, edge_number = networkx_graph_generator(mode=mode_list[mode_select])

dats, dat_path = dats_generator(demo_graph, mode=distribution_list[distribution_select], dat_number=C*node_number, seed=False)
l_t = 5
delta = 0.01
all_edge_duv_time_list = list()
all_not_edge_duv_time_list = list()
for i in range(node_number):
    for j in range(i + 1, node_number):
        if (i, j) in demo_graph.edges() or (j, i) in demo_graph.edges():
            for dat in dats:
                all_edge_duv_time_list.append(abs(dat[i] - dat[j]))
            break
    break


for i in range(node_number):
    for j in range(i + 1, node_number):
        if (i, j) not in demo_graph.edges() and (j, i) not in demo_graph.edges():
            for dat in dats:
                all_not_edge_duv_time_list.append(abs(dat[i] - dat[j]))
            break
    break
print(node_number)
print(len(all_edge_duv_time_list))
print(len(all_not_edge_duv_time_list))
x_range = [float(i) * delta * 10 for i in range(int((l_t / delta) / 10) + 1)]
x_range_original = [i for i in range(int(l_t/delta))]

pdf_50 = pdf_recover_from_list(all_edge_duv_time_list)
pdf_50 = normalize_pdf(pdf_50)
sf_50 = normalize_sf(pdf2sf_using_density(pdf_50))

pdf_not_50 = pdf_recover_from_list(all_not_edge_duv_time_list)
pdf_not_50 = normalize_pdf(pdf_not_50)
sf_not_50 = normalize_sf(pdf2sf_using_density(pdf_not_50))

original_pdf = pdf_generator(mode=distribution_list[distribution_select])

original_pdf = normalize_pdf(original_pdf)
original_sf = normalize_sf(pdf2sf_using_density(original_pdf))

nx = np.linspace(0, 5, 500)
f_pdf = interpolate.interp1d(x_range, pdf_50)
pdf_500 = f_pdf(nx)
f_sf = interpolate.interp1d(x_range, sf_50)
sf_500 = f_sf(nx)

f_pdf_not = interpolate.interp1d(x_range, pdf_not_50)
pdf_not_500 = f_pdf(nx)
f_sf = interpolate.interp1d(x_range, sf_not_50)
sf_not_500 = f_sf(nx)

pdf_500 = normalize_pdf(pdf_500)
sf_500 = normalize_sf(sf_500)
pdf_not_500 = normalize_pdf(pdf_not_500)
sf_not_500 = normalize_sf(sf_not_500)


plt.subplot(1,3,2)

plt.xlabel(r'$\tau$', fontsize=axis_legend_font_size)
plt.ylabel("Probability",fontsize=axis_legend_font_size)
plt.step(x_range,sf_50,'b',linewidth=step_line_width,alpha=0.5)
plt.step(x_range,sf_not_50,'r',linewidth=step_line_width,alpha=0.5)
plt.plot(nx, sf_500, 'b--',linewidth=curve_line_width,label=r'$\hat{\Phi}_{(u,v)\in\mathcal{G}}(\tau)$')
plt.plot(nx, sf_not_500, 'r--', linewidth=curve_line_width, label=r'$\hat{\Phi}_{(u,v)\notin\mathcal{G}}(\tau)$')
plt.plot(nx, original_sf, 'g-', linewidth=curve_line_width,label=r'$\Phi_{(u,v)\in\mathcal{G}}(\tau)$')
plt.legend(fontsize=legend_font_size)
plt.xticks(fontsize=axis_font_size)
plt.yticks(fontsize=axis_font_size)



mode_list = ["ER","BA","SW"]
distribution_list = ["gaussian","uniform","weibull", "gumbel","exponential","beta"]
mode_select = 0
distribution_select = 3
C=10


demo_graph, node_number, edge_number = networkx_graph_generator(mode=mode_list[mode_select])

dats, dat_path = dats_generator(demo_graph, mode=distribution_list[distribution_select], dat_number=C*node_number, seed=False)
l_t = 5
delta = 0.01
all_edge_duv_time_list = list()
all_not_edge_duv_time_list = list()
for i in range(node_number):
    for j in range(i + 1, node_number):
        if (i, j) in demo_graph.edges() or (j, i) in demo_graph.edges():
            for dat in dats:
                all_edge_duv_time_list.append(abs(dat[i] - dat[j]))
            break
    break


for i in range(node_number):
    for j in range(i + 1, node_number):
        if (i, j) not in demo_graph.edges() and (j, i) not in demo_graph.edges():
            for dat in dats:
                all_not_edge_duv_time_list.append(abs(dat[i] - dat[j]))
            break
    break
print(node_number)
print(len(all_edge_duv_time_list))
print(len(all_not_edge_duv_time_list))
x_range = [float(i) * delta * 10 for i in range(int((l_t / delta) / 10) + 1)]
x_range_original = [i for i in range(int(l_t/delta))]

pdf_50 = pdf_recover_from_list(all_edge_duv_time_list)
pdf_50 = normalize_pdf(pdf_50)
sf_50 = normalize_sf(pdf2sf_using_density(pdf_50))

pdf_not_50 = pdf_recover_from_list(all_not_edge_duv_time_list)
pdf_not_50 = normalize_pdf(pdf_not_50)
sf_not_50 = normalize_sf(pdf2sf_using_density(pdf_not_50))

original_pdf = pdf_generator(mode=distribution_list[distribution_select])

original_pdf = normalize_pdf(original_pdf)
original_sf = normalize_sf(pdf2sf_using_density(original_pdf))

nx = np.linspace(0, 5, 500)
f_pdf = interpolate.interp1d(x_range, pdf_50)
pdf_500 = f_pdf(nx)
f_sf = interpolate.interp1d(x_range, sf_50)
sf_500 = f_sf(nx)

f_pdf_not = interpolate.interp1d(x_range, pdf_not_50)
pdf_not_500 = f_pdf(nx)
f_sf = interpolate.interp1d(x_range, sf_not_50)
sf_not_500 = f_sf(nx)

pdf_500 = normalize_pdf(pdf_500)
sf_500 = normalize_sf(sf_500)
pdf_not_500 = normalize_pdf(pdf_not_500)
sf_not_500 = normalize_sf(sf_not_500)


plt.subplot(1,3,3)

plt.xlabel(r'$\tau$', fontsize=axis_legend_font_size)
plt.ylabel("Probability",fontsize=axis_legend_font_size)
plt.step(x_range,sf_50,'b',linewidth=step_line_width,alpha=0.5)
plt.step(x_range,sf_not_50,'r',linewidth=step_line_width,alpha=0.5)
plt.plot(nx, sf_500, 'b--',linewidth=curve_line_width,label=r'$\hat{\Phi}_{(u,v)\in\mathcal{G}}(\tau)$')
plt.plot(nx, sf_not_500, 'r--', linewidth=curve_line_width, label=r'$\hat{\Phi}_{(u,v)\notin\mathcal{G}}(\tau)$')
plt.plot(nx, original_sf, 'g-', linewidth=curve_line_width,label=r'$\Phi_{(u,v)\in\mathcal{G}}(\tau)$')
plt.legend(fontsize=legend_font_size)
plt.xticks(fontsize=axis_font_size)
plt.yticks(fontsize=axis_font_size)
plt.subplots_adjust(top=0.98,bottom=0.15,left=0.05,right=0.99,wspace=0.23)

plt.savefig("../figure/"+mode_list[mode_select]+"_"+distribution_list[distribution_select]+ "_sf.svg", format="svg")
plt.savefig("../figure/"+mode_list[mode_select]+"_"+distribution_list[distribution_select]+"_sf.eps", dpi=600, format="eps")
plt.show()