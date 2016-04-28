import time
import datetime
import os
import warnings
import data.DocumentData as DocumentData
from methods.LocalitySensitiveHash import LocalitySensitiveHash
from util import EvalUtil
from kdtree.KDTree import KDTree
from TestResult import *
import math
import ggplot as gg
import pandas as pd
import pickle as pkl


def make_dense(document):
    key = [0 for idx in xrange(0, D)]
    for word_id, count in document.iteritems():
        #words are indexed from 1
        key[word_id-1] = count
    return key


def test_approx_nn(method, traindata, testdata, m, alpha):
        avg_distance = 0
        if method == "hashing":
            #train
            lsh = LocalitySensitiveHash(traindata, D=1000, m=m)
            #time test
            t0 = time.time()
            for testdoc_id, testdoc in testdata.iteritems():
                avg_distance += lsh.nearest_neighbor(testdoc, depth = HW2_DEPTH).distance
        if method == "kdtree":
            #train
            kdt = KDTree(D)
            for i, document in traindata.iteritems():
                key = make_dense(document)
                kdt.insert(key, i)
            #time test
            t0 = time.time()
            for _, testdoc in testdata.iteritems():
                key = make_dense(testdoc)
                neighbor = kdt.nearest(key, alpha)
                avg_distance += EvalUtil.distance(testdoc, docdata[neighbor])

        #finish timing, report results
        mean_time = (time.time() - t0) / len(testdata)
        mean_distance = avg_distance   / len(testdata)
        return TestResult(method, m=m, D=D, alpha = alpha, avg_time=mean_time, avg_distance=mean_distance)


DATA_PATH = '../../data/'
if True: #__name__ == '__main__':

    #Load data and make small subset for debugging
    docdata  = DocumentData.read_in_data(os.path.join(DATA_PATH,  "sim_docdata.mtx"), True)
    testdata = DocumentData.read_in_data(os.path.join(DATA_PATH, "test_docdata.mtx"), True)

    test_mode = True
    if test_mode:
        train_n, test_n  = (100, 50)
        def first_n(long_dict, n):
            my_keys = long_dict.keys()[0:n]
            return {my_key:long_dict[my_key] for my_key in my_keys}
        _ = "Currently testing on just %d and %d documents, train and test" % (train_n, test_n)
        warnings.warn(_)
        _ = "Also testing code with training set equal to test set (solves LSH empty bin problem)"
        warnings.warn(_)
        docdata = first_n(docdata, train_n)
        testdata = first_n(docdata, test_n)

    #Parameters
    D = 1000
    HW2_DEPTH = 3
    mvals = [5, 10, 20]
    avals = [1,  5, 10]

    #Testing
    results = []
    for m in mvals:
        results.append(test_approx_nn(method = "hashing", traindata=docdata, testdata = testdata, m=m, alpha=1))
    for alpha in avals:
        results.append(test_approx_nn(method = "kdtree" , traindata=docdata, testdata = testdata, m=1, alpha=alpha))

    #save results to results folder, with plot and printing to screen.
    metadata = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "test_mode==" + str(test_mode)
    f = open("results/LSH_vs_KDT_%s.pkl" % metadata, mode = 'w')
    pkl.dump(obj=results, file=f)

    logtimes =  [math.log(r.avg_time, 2)     for r in results]
    distances = [r.avg_distance for r in results]
    methods =   [r.method[0:3]  for r in results]
    alpha =     [r.alpha  for r in results]
    m =         [r.m  for r in results]
    results_df = pd.DataFrame(data = {"logtimes" : logtimes,
                                      "distances" : distances,
                                      "methods" : methods,
                                      "m":m,
                                      "alpha": alpha})
    print results_df
    p = gg.ggplot(data = results_df, aesthetics = gg.aes(x = "logtimes",
                                                         y = "distances",
                                                         label = "methods")) + \
        gg.geom_text() + \
        gg.ggtitle("LSH and KD trees: tradeoffs") + \
        gg.xlab("Log2 average query time  ") + gg.ylab("Average L2 distance from query point)")
    gg.ggsave(filename="results/LSH_vs_KDT_%s.png" % metadata, plot = p)
