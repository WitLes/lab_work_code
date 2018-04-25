from algorithm_realization.stn_reconstruction_lib import *
import scipy.interpolate as interpolate

mode_list = ["ER","BA","SW"]
distribution_list = ["gaussian","uniform","weibull", "gumbel"]
mode_select = mode_list[2]
distribution_select = distribution_list[3]


demo_graph, node_number, edge_number = networkx_graph_generator(mode=mode_select)

dats, dat_path = dats_generator(demo_graph, mode=distribution_select, dat_number=10*node_number, seed=False)
l_t = 5
delta = 0.01
all_edge_duv_time_list = list()
all_not_edge_duv_time_list = list()
for i in range(node_number):
    for j in range(i + 1, node_number):
        if (i, j) in demo_graph.edges() or (j, i) in demo_graph.edges():
            for dat in dats:
                all_edge_duv_time_list.append(abs(dat[i] - dat[j]))
        else:
            for dat in dats:
                all_not_edge_duv_time_list.append(abs(dat[i] - dat[j]))

x_range = [float(i) * delta * 10 for i in range(int((l_t / delta) / 10) + 1)]
x_range_original = [i for i in range(int(l_t/delta))]
pdf = pdf_recover_from_list(all_edge_duv_time_list)
pdf = normalize_pdf(pdf)
sf = normalize_sf(pdf2sf_using_density(pdf))
pdf_not = pdf_recover_from_list(all_not_edge_duv_time_list)
pdf_not = normalize_pdf(pdf_not)
sf_not = normalize_sf(pdf2sf_using_density(pdf_not))

original_pdf = pdf_generator(mode=distribution_select)

original_pdf = normalize_pdf(original_pdf)
original_sf = normalize_sf(pdf2sf_using_density(original_pdf))

nx = np.linspace(0, 5, 500)
f_pdf = interpolate.interp1d(x_range, pdf, kind="quadratic")
pdf = f_pdf(nx)
f_pdf_not = interpolate.interp1d(x_range, pdf_not, kind='quadratic')
pdf_not = f_pdf_not(nx)

f_sf = interpolate.interp1d(x_range, sf, kind="quadratic")
sf = f_sf(nx)
f_sf_not = interpolate.interp1d(x_range, sf_not, kind="quadratic")
sf_not = f_sf_not(nx)

pdf = normalize_pdf(pdf)
pdf_not = normalize_pdf(pdf_not)
sf = normalize_sf(sf)
sf_not = normalize_sf(sf_not)


plt.figure(figsize=(12, 5))
plt.style.use('ggplot')
# density
plt.subplot(1, 2, 1)
plt.title("PDF", fontsize=10)
plt.xlabel("time interval",fontsize=10)
plt.ylabel("probability",fontsize=10)
plt.plot(nx, pdf, '-', label="topological edge")
plt.plot(nx, pdf_not, '-', label="non-topological edge")
plt.plot(nx, original_pdf, '-', label="distribution")
#plt.legend(fontsize=10)

# survival
plt.subplot(1, 2, 2)
plt.title("SF", fontsize=10)
plt.xlabel("time interval", fontsize=10)
plt.ylabel("probability",fontsize=10)
plt.plot(nx, sf, '-', label="topological edge")
plt.plot(nx, sf_not, '-', label="non-topological edge")
plt.plot(nx, original_sf, '-', label="distribution")
#plt.legend(fontsize=10)

plt.subplots_adjust(top=0.8,wspace=0.3)
plt.savefig("../figure/"+mode_select+"_"+distribution_select+ "_pdf_sf.svg", format="svg")
plt.savefig("../figure/"+mode_select+"_"+distribution_select+"_pdf_sf.jpg", dpi=600, format="jpg")
#plt.suptitle()
plt.show()