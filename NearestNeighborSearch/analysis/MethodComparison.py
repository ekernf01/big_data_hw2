import time

import data.DocumentData as DocumentData
from methods.LocalitySensitiveHash import LocalitySensitiveHash
from methods.GaussianRandomProjection import GaussianRandomProjection
from kdtree.KDTree import KDTree


if __name__ == '__main__':
    docdata = DocumentData.read_in_data("/path/to/sim_docdata.mtx", True)
    testdata = DocumentData.read_in_data("/path/to/test_docdata.mtx", True)
    print "Number of Documents: %d" % len(docdata)
    print "Number of Test Documents: %d" % len(testdata)
    D = 1000
    # TODO: run tests here
