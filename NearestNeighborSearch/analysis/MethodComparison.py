import time
import datetime
import os
import warnings
import data.DocumentData as DocumentData
from methods.LocalitySensitiveHash import LocalitySensitiveHash
from util import EvalUtil
from kdtree.KDTree import KDTree
from TestResult import *
import ggplot as gg
import pandas as pd
import pickle as pkl


def test_approx_nn(method, traindata, testdata):
        avg_distance = 0
        if method == "hashing":
            #train
            lsh = LocalitySensitiveHash(traindata, D, m)
            #time test
            t0 = time.time()
            for testdoc_id, testdoc in testdata.iteritems():
                avg_distance += lsh.nearest_neighbor(testdoc, depth = HW2_DEPTH).distance
        if method == "kdtree":
            #train
            kdt = KDTree(D)
            for i, document in traindata.iteritems():
                key = [0 for idx in xrange(0, D)]
                for word_id, count in document.iteritems():
                    #words are indexed from 1
                    key[word_id-1] = count
                kdt.insert(key, i)
            #time test
            t0 = time.time()
            for _, testdoc in testdata.iteritems():
                neighbor = kdt.nearest(key, alpha)
                avg_distance += EvalUtil.distance(testdoc, docdata[neighbor])

        #finish timing, report results
        mean_time = (time.time() - t0) / len(testdata)
        mean_distance = avg_distance   / len(testdata)
        return TestResult(method, n=len(docdata), D=D, alpha = 1.3, avg_time=mean_time, avg_distance=mean_distance)


DATA_PATH = '../../data/'
if __name__ == '__main__':

    #Load data and define tester routine hardwired to these data
    docdata  = DocumentData.read_in_data(os.path.join(DATA_PATH,  "sim_docdata.mtx"), True)
    testdata = DocumentData.read_in_data(os.path.join(DATA_PATH, "test_docdata.mtx"), True)

    train_n, test_n  = (100, 10)
    def first_n(long_dict, n):
        my_keys = long_dict.keys()[0:n]
        return {my_key:long_dict[my_key] for my_key in my_keys}
    _ = "Currently testing on just %d and %d documents, train and test" % (train_n, test_n)
    warnings.warn(_)
    docdata = first_n(docdata, train_n)
    testdata = first_n(testdata, test_n)



    #Parameters
    D = 1000
    HW2_DEPTH = 3
    mvals = [5, 10, 20]
    avals = [1,  5, 10]

    #Testing
    results = []
    for m in mvals:
        results.append(test_approx_nn(method = "hashing", traindata=docdata, testdata = testdata))
    for alpha in avals:
        results.append(test_approx_nn(method = "kdtree" , traindata=docdata, testdata = testdata))

    times =     [r.avg_time     for r in results]
    distances = [r.avg_distance for r in results]
    methods =   [r.method[0:3]  for r in results]
    results_df = pd.DataFrame(data = {"times" : times, "distances" : distances, "methods" : methods})
    p = gg.ggplot(data = results_df, aesthetics = gg.aes(x = "times",
                                                         y = "distances",
                                                         label = "methods")) + \
        gg.geom_text() + \
        gg.ggtitle("LSH and KD trees: tradeoffs") + \
        gg.xlab("Average query time  ") + gg.ylab("Average L2 distance from query point)")
    print os.getcwd()
    nowstr = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    gg.ggsave(filename="lsh_vs_kdt_%s.png" % nowstr, plot = p)
    pkl.pickle(results, filename="lsh_vs_kdt_%s.pkl" % nowstr)