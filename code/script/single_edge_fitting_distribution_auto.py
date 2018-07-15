from algorithm_realization.stn_reconstruction_lib import *
import scipy.interpolate as interpolate
C = 1
mode_list = ["ER","BA","SW"]
mode_select = 0
distribution_list = ["gaussian","uniform","weibull", "gumbel","exponential","beta"]
distribution_select = 0
demo_graph, node_number, edge_number = networkx_graph_generator(mode=mode_list[mode_select])
plt.figure(figsize=(12, 8))
plt.style.use("ggplot")
plt.xlabel("time interval", fontsize=10)
plt.ylabel("probability", fontsize=10)
for distribution_select in range(len(distribution_list)):

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

    x_range = [float(i) * delta * 10 for i in range(int((l_t / delta) / 10) + 1)]
    x_range_original = [i for i in range(int(l_t/delta))]
    pdf = pdf_recover_from_list(all_edge_duv_time_list)
    pdf = normalize_pdf(pdf)
    sf = normalize_sf(pdf2sf_using_density(pdf))

    original_pdf = pdf_generator(mode=distribution_list[distribution_select])

    original_pdf = normalize_pdf(original_pdf)
    original_sf = normalize_sf(pdf2sf_using_density(original_pdf))

    nx = np.linspace(0, 5, 500)
    f_pdf = interpolate.interp1d(x_range, pdf, kind="quadratic")
    pdf = f_pdf(nx)

    f_sf = interpolate.interp1d(x_range, sf, kind="quadratic")
    sf = f_sf(nx)


    pdf = normalize_pdf(pdf)
    sf = normalize_sf(sf)

    plt.style.use('ggplot')
    plt.subplot(2,3,distribution_select+1)
    # survival
    #plt.title("SF", fontsize=10)
    plt.xlabel("time interval", fontsize=10)
    plt.ylabel("probability",fontsize=10)
    plt.plot(nx, sf, '-', label="topological edge")
    plt.plot(nx, original_sf, '-', label="distribution")
    #plt.legend(fontsize=10)
plt.subplots_adjust(top=0.8,wspace=0.3)
plt.savefig("../figure/"+mode_list[mode_select]+"_"+distribution_list[distribution_select]+ "_pdf_sf.svg", format="svg")
plt.savefig("../figure/"+mode_list[mode_select]+"_"+distribution_list[distribution_select]+"_pdf_sf.png", dpi=600, format="png")
plt.savefig("../figure/"+mode_list[mode_select]+"_"+distribution_list[distribution_select]+"_pdf_sf.eps", format="eps")
plt.show()