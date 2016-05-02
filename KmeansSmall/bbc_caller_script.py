from BBCRead import BBCData
from VanillaKmeans import KmeansCall
import matplotlib.pyplot as plt

quite_cheers_lorry = BBCData()
print quite_cheers_lorry.freq_mtx.head(5)
five_best = quite_cheers_lorry.get_tfidf_by_class()

for c in range(5):
    print "class " + str(c)
    print "".join([(pair[0] + ":" + str(round(pair[1], 3)) + ", ") for pair in five_best[c]] )


bbc_kmcall = KmeansCall(data = quite_cheers_lorry.get_dense_data(),
                        num_clusters = 5, initialization = "specified",
                        init_centers_path = quite_cheers_lorry.path + "/bbc.centers")

error = [0 for i in range(5)]
for i in range(5):
    #It'll pick up where it left off next time.
    bbc_kmcall.run_kmeans(maxiter = 1)
    is_correct = (bbc_kmcall.data.class_labels == quite_cheers_lorry.class_mtx.class_label)
    error[i] = np.mean([int(george) for george in is_correct])

plt.plot(range(5), error)
plt.title("Classification error by iteration")