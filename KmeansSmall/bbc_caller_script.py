from BBCRead import BBCData
from VanillaKmeans import KmeansCall
from EMCall import EMCall
import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl
import os

#run these three lines the first time to cache the data
#quite_cheers_lorry = BBCData()
#with open("results/bbc_data.pkl", "wb") as f:
#    pkl.dump(obj=quite_cheers_lorry, file=f)
#run these lines every time to recover the data
quite_cheers_lorry = pkl.load(open("results/bbc_data.pkl", "rb"))
print quite_cheers_lorry.freq_mtx.head(5)
for c in range(5):
    print "class " + str(c)
    print "".join([(pair[0] + ":" + str(round(pair[1], 3)) + ", ") \
                   for pair in quite_cheers_lorry.five_best[c]] )

# Cluster BBC via EM

def run_cluster_bbc(method):
    bbc_emcall = EMCall(data=quite_cheers_lorry.get_dense_data(),
                        num_clusters=5, cov_reg=0.2,
                        initialization="specified",
                        init_centers_path=quite_cheers_lorry.path + "/bbc.centers")
    #Run 5 iterations
    accuracy = [0 for j in range(5)]
    def do_one():
        if method == "em":
            bbc_emcall.run_em(maxiter = 1)
        elif  method == "kmeans":
            bbc_emcall.run_kmeans(maxiter = 1)
        else:
            raise Exception("method should be either 'em' or 'kmeans'")
        return
    for j in range(5):
        #It'll pick up where it left off next time.
        print "Done with " + str(j) + " iterations."
        do_one()
        pred_labels = quite_cheers_lorry.class_mtx.class_label
        is_correct = [bbc_emcall.class_label[i] == str(label) for i, label in enumerate(pred_labels)]
        accuracy[j] = np.mean([int(george_boole) for george_boole in is_correct])

    #save results and plots
    with open("results/bbc_"+ method + ".pkl", "wb") as f:
        pkl.dump(obj=bbc_emcall, file=f)
    plt.close()
    plt.plot(range(5), accuracy)
    plt.title(method + " classification accuracy by iteration")
    plt.savefig("results/bbc_" + method + "_accuracy.png")

    plt.close()
    plt.plot(range(5), bbc_emcall.obj_record)
    plt.title(method + " objective by iteration")
    plt.savefig("results/bbc_" + method + "_obj.png")
    return

run_cluster_bbc("kmeans")
run_cluster_bbc("em")