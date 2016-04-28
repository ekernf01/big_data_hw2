from methods.GaussianRandomProjection import GaussianRandomProjection
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

DOCDIM = 1000
def test_kd_tree(n, D, n_test, alphas):
    """
    Tests the query time and distance for a random data set and test set
    @param n: int - the number of points of the dataset
    @param D: int - the dimension of the data points
    @param n_test: int - the number of points to test
    @param alphas: [float] - a set of alphas to test
    @return [TestResult] array of objects of class TestResult, which has the average time and distance for a single query
    """
    documents = RandomData.random_dataset(n, DOCDIM)
    test_documents = RandomData.random_dataset(n_test, DOCDIM)

    rand_tree = KDTree(DOCDIM)
    for i, document in documents.iteritems():
        key = [document.get(idx) for idx in xrange(0, DOCDIM)]
        rand_tree.insert(key, i)

    times = []
    for alpha in alphas:
        start_time = time.clock()
        cum_dist = 0.0
        for i, test_document in test_documents.iteritems():
            key = [test_document.get(idx) for idx in xrange(0, DOCDIM)]
            doc_id = rand_tree.nearest(key, alpha)
            cum_dist += EvalUtil.distance(test_document, documents[doc_id])
        duration = time.clock() - start_time
        times.append(TestResult("KDTree", n, DOCDIM, alpha, duration / n_test, cum_dist / n_test))
    return times


def test_rptree(traindata, testdata, projdim):
    avg_distance = 0

    #train, start timer, test
    rptree = GaussianRandomProjection(traindata, D=DOCDIM, m=projdim)
    t0 = time.time()
    for _, testdoc in testdata.iteritems():
        neighbor = rptree.nearest_neighbor(testdoc, alpha=1)
        avg_distance += EvalUtil.distance(testdoc, rptree.documents[neighbor.doc_id])

    #finish timing, report results
    mean_time = (time.time() - t0) / len(testdata)
    mean_distance = avg_distance   / len(testdata)
    return TestResult(method = "rpkdt", m=projdim, D=DOCDIM, alpha = 1, avg_time=mean_time, avg_distance=mean_distance)



#Load data and make small subset for debugging
DATA_PATH = '../../data/'
docdata  = DocumentData.read_in_data(os.path.join(DATA_PATH,  "sim_docdata.mtx"), True)
testdata = DocumentData.read_in_data(os.path.join(DATA_PATH, "test_docdata.mtx"), True)

test_mode = False
if test_mode:
    train_n, test_n  = (10000, 5000)
    def first_n(long_dict, n):
        my_keys = long_dict.keys()[0:n]
        return {my_key:long_dict[my_key] for my_key in my_keys}
    _ = "Currently testing on just %d and %d documents, train and test" % (train_n, test_n)
    warnings.warn(_)
    _ = "Also testing code with training set equal to test set (solves no-match problem)"
    warnings.warn(_)
    docdata = first_n(docdata, train_n)
    testdata = first_n(docdata, test_n)

results = [None for i in range(3)]
for i, projdim in enumerate([5, 10, 20]):
    results[i] = test_rptree(projdim=projdim, traindata=docdata, testdata = testdata)

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