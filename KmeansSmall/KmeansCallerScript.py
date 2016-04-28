from VanillaKmeans import KmeansCall
import pandas as pd
import numpy as np
import os
import ggplot as gg
import datetime
import pickle as pkl

PATH_TO_DATA = "../../data"
synth_data = pd.read_csv(os.path.join(PATH_TO_DATA, "2DGaussianMixture.csv"))
#the 'class' column also gets renamed to 'class_label' since class is a keyword.
synth_data.rename(columns={'class': 'class_label'}, inplace=True)
print "Data looks (ok, 'data look') like:"
print synth_data.head(5)
datestring = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def prob231b():
    cluster_counts = [2,3,5,10,15,20]
    kmcalls = [0 for i in cluster_counts]
    for i, num_clusters in enumerate(cluster_counts):
        kmcalls[i] = KmeansCall(synth_data, num_clusters)
        kmcalls[i].run_kmeans(verbose = False)
        p = gg.ggplot(kmcalls[i].data, gg.aes(x= "x1", y="x2", colour="class_label")) + \
        gg.geom_point() + gg.ggtitle("Synth. data, k=" + str(num_clusters))
        metadata = "k=" + str(num_clusters) + "_" + datestring
        gg.ggsave(filename = "results/" + metadata +".png", plot = p)


def prob231cd(initialization, num_trials = 20):
    num_clusters_231cd = 3
    kmcalls = [0 for i in range(num_trials)]
    prob231c_plot_df = pd.DataFrame({"x1":synth_data.x1.copy(), "x2":synth_data.x2.copy(), "data":"data"})
    for i in range(num_trials):
        print "prob231cd iter " + str(i)
        kmcalls[i] = KmeansCall(synth_data, num_clusters_231cd, initialization)
        kmcalls[i].run_kmeans(verbose = False)
        for k in range(num_clusters_231cd):
            prob231c_plot_df.loc[len(prob231c_plot_df)] = np.concatenate((["centers"], kmcalls[i].cluster_centers[k]))
    filename = "results/prob231cd" + initialization + ".pkl"
    pkl.dump(obj = (prob231c_plot_df, kmcalls, num_trials), file = open(filename, "wb"))
    return

def prob231cd_recover(initialization):
    filename = "results/prob231cd" + initialization
    tuple_in = pkl.load(open(filename + ".pkl", "rb"))
    prob231c_plot_df = tuple_in[0]
    kmcalls = tuple_in[1]
    num_trials = tuple_in[2]
    p = gg.ggplot(prob231c_plot_df, gg.aes(x= "x1", y="x2", colour="data")) + \
        gg.geom_point() + gg.ggtitle(initialization + " initialization")
    print p
    gg.ggsave(filename + ".png", plot = p)
    obj = [kmcalls[i].obj for i in range(num_trials)]
    obj_stats = {"mean":np.mean(obj), "sd":np.std(obj), "min":np.min(obj)}
    return obj_stats


#prob231b()

prob231cd(initialization = "regular", num_trials=20)
obj_stats = prob231cd_recover(initialization = "regular")
print obj_stats

prob231cd(initialization = "km++", num_trials=20)
obj_stats2 = prob231cd_recover(initialization = "km++")
print obj_stats2
