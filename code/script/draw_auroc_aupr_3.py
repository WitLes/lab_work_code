from algorithm_realization.stn_reconstruction_lib import *

record=open("record_data.txt","w+")

file_name = ["data/scalefree.txt","data/polbooks.txt","data/football.txt","data/apollonian.txt","data/dolphins.txt","data/karate.txt","data/lattice2d.txt","data/miserables.txt","data/pseudofractal.txt","data/randomgraph.txt","data/scalefree.txt","data/sierpinski.txt","data/smallworld.txt","data/jazz.txt"]
file_name = ["../"+string for string in file_name]

# graph type and distribution type
mode_list = ["ER","BA","SW"]
distribution_list = ["gaussian","uniform","gumbel"]
mode_select = 1
distribution_select = 2

# x-axis : C=M/N
percent_list = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
color_list=["r","m","b","c","g","r","m","b","c","g"]



#demo_graph, node_number, edge_number = open_file_data_graph(file_name[mode_select])
demo_graph, node_number, edge_number = networkx_graph_generator(mode=mode_list[mode_select])
# demo_graph, node_number, edge_number = demo_graph_generator()
print(mode_list[mode_select], "  graph || nodes:", node_number, "; edges:", edge_number)
print("distribution: ",distribution_select)
print("average shortest path length: ", nx.average_shortest_path_length(demo_graph))
print("average clustering coef: ", nx.average_clustering(demo_graph))
auroc_list = [0 for i in range(10)]
aupr_list = [0 for i in range(10)]
# generate data arrival times(dats), also dat_path.
# set dat_number to control the total number of dat.
TPR_list = [[0 for j in range(10)]for i in range(50)]
FPR_list = [[0 for j in range(10)]for i in range(50)]
Precison_list = [[0 for j in range(10)]for i in range(50)]
Recall_list = [[0 for j in range(10)]for i in range(50)]
best_perform=[0 for i in range(10)]
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.title("ROC Curve", fontsize=10)
plt.xlabel("FPR",fontsize=10)
plt.ylabel("TPR",fontsize=10)
plt.xlim(-0.05, 1.05)
plt.ylim(-0.05, 1.05)
plt.subplot(1, 2, 2)
plt.title("PR Curve", fontsize=10)
plt.xlabel("Recall", fontsize=10)
plt.ylabel("Precision", fontsize=10)
plt.xlim(-0.05, 1.05)
plt.ylim(-0.05, 1.05)
for k in range(10):
    for c in range(10):
        print(str(c),end=" ",flush=True)
        dats, dat_path = dats_generator(demo_graph, mode=distribution_list[distribution_select],dat_number=int(percent_list[k]*node_number), seed=False)
        discrete_wtd = pdf_generator(mode=distribution_list[distribution_select])

        for i in range(50):
            adj_mat = faster_topology_reconstruction_through_dats_based_on_wtd2(dats, discrete_wtd,cutting_index=i)
            tp_fp = count_in_matrix(adj_mat) / 2
            TP = FN = FP = TN = 0
            edge_total = node_number*(node_number-1)/2

            for edge in demo_graph.edges():
                if adj_mat[edge[0]][edge[1]] == 1:
                    TP += 1
                else:
                    FN += 1
            FP = tp_fp - TP
            TN = edge_total - TP - FN - FP
            if TP == 0:
                TPR_list[i][c]=0
                FPR_list[i][c]=0
                Precison_list[i][c]=0
                Recall_list[i][c]=0
            else:
                TPR = TP / (TP + FN)
                FPR = FP / (FP + TN)
                Precison = TP / (TP + FP)
                Recall = TP / (TP + FN)
                TPR_list[i][c] = TPR
                FPR_list[i][c] = FPR
                Precison_list[i][c] = Precison
                Recall_list[i][c] = Recall

    TPR_final = [0 for i in range(50)]
    FPR_final = [0 for i in range(50)]
    Precison_final = [0 for i in range(50)]
    Recall_final = [0 for i in range(50)]

    for i in range(50):
        TPR_final[i] = sum(TPR_list[i])/10
        FPR_final[i] = sum(FPR_list[i])/10
        Precison_final[i] = sum(Precison_list[i])/10
        Recall_final[i] = sum(Recall_list[i])/10

    # use TPR_final FPR_final Precison_final Recall_final to calculate AUPR and AUROC

    f1 = 0
    TPR_best = 0
    FPR_best = 0
    Precison_best = 0
    Recall_best = 0
    for i in range(50):
        if TPR_final[i] != 0:
            if f1 < 2*Precison_final[i]*Recall_final[i]/(Precison_final[i]+Recall_final[i]):
                f1 = 2*Precison_final[i]*Recall_final[i]/(Precison_final[i]+Recall_final[i])
                TPR_best = TPR_final[i]
                FPR_best = FPR_final[i]
                Precison_best = Precison_final[i]
                Recall_best = Recall_final[i]
    best_perform[k]=[f1,TPR_best,FPR_best,Precison_best,Recall_best]


    i = 0
    while i < len(TPR_final):
        if TPR_final[i] <0.1 and FPR_final[i] < 0.2:
            TPR_final.pop(i)
            FPR_final.pop(i)
        else:
            i += 1

    i = 0
    while i < len(Precison_final):
        if Precison_final[i] < 0.95 and Recall_final[i] <0.15:
            Precison_final.pop(i)
            Recall_final.pop(i)
        else:
            i += 1


    temp_auroc_sum = 0
    temp_aupr_sum = 0
    len_auroc = len(TPR_final)
    len_aupr = len(Precison_final)
    for t in range(len_auroc):
        if t == 0:
            temp_auroc_sum += (FPR_final[0]-0)*TPR_final[0]/2
        else:
            temp_auroc_sum += (abs(FPR_final[t] - FPR_final[t-1]))* (TPR_final[t]+TPR_final[t-1])/2
    temp_auroc_sum += (1 - FPR_final[len_auroc-1]) * (1+TPR_final[len_auroc-1])/2

    for t in range(len_aupr):
        if t == 0:
            temp_aupr_sum += (Recall_final[0]) *(1+Precison_final[0]) / 2
        else:
            temp_aupr_sum += (abs(Recall_final[t] - Recall_final[t-1])) * (Precison_final[t] + Precison_final[t - 1]) / 2
    temp_aupr_sum += (1-Recall_final[len_aupr-1]) * (Precison_final[len_aupr-1]) / 2
    print(temp_aupr_sum,temp_auroc_sum)

    auroc_list[k] = temp_auroc_sum
    aupr_list[k] = temp_aupr_sum

    # density
    plt.subplot(1, 2, 1)

    #plt.plot(FPR_final,TPR_final,color_list[k]+"*")
    plt.plot(FPR_final,TPR_final,"o"+color_list[k]+"-",linewidth=0.5,markersize=3,label="C = "+str(percent_list[k]))
    #plt.legend(fontsize=10)
    plt.legend()
    # survival
    plt.subplot(1, 2, 2)
    #plt.plot(Recall_final,Precison_final,color_list[k]+"*")
    plt.plot(Recall_final,Precison_final,"o"+color_list[k]+"-",linewidth=0.5,markersize=3,label="C = "+str(percent_list[k]))
    plt.legend()
#plt.legend(fontsize=10)
print(best_perform)
print(aupr_list)
print(auroc_list)
plt.subplots_adjust(top=0.8,wspace=0.3)
plt.savefig("../figure/"+file_name[mode_select]+distribution_list[distribution_select]+".svg", format="svg")
plt.savefig("../figure/"+file_name[mode_select]+distribution_list[distribution_select]+".png", dpi=600, format="png")
plt.savefig("../figure/"+file_name[mode_select]+distribution_list[distribution_select]+".eps", format="eps")
#plt.suptitle()

plt.show()