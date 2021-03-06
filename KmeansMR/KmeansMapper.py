#!/usr/bin/env python

import sys
import os

from Document import Document
from Cluster import Cluster
import MathUtil
import HDFSUtil


class KmeansMapper:
    def __init__(self):
        self.K = int(os.environ.get("numClusters", 1))
        self.iteration = int(os.environ.get("kmeansIter", 1))
        self.kmeans_hdfs_path = os.environ.get("kmeansHDFS")
        self.hadoop_prefix = os.environ.get("hadoopPrefix")
        self.clusters = []

    def read_input(self, file):
        for line in file:
            yield line.rstrip()

    def emit(self, key, value, separator="\t"):
        """ Emit (key, value) pair to stdout for hadoop streaming """
        #print >> sys.stderr, "emitting document: %s" % key
        print '%s%s%s' % (key, separator, value)

    def main(self):
        if self.iteration == 1:
            path = self.kmeans_hdfs_path + "/cluster0/cluster0.txt"
        else:
            path = self.kmeans_hdfs_path + "/output/cluster" + str(self.iteration - 1) + "/part-00000"
        for line in HDFSUtil.read_lines(path, hadoop_prefix=self.hadoop_prefix):
            if self.iteration > 1:
                if line.startswith("cluster"):
                    line = line.split("\t", 1)[1]
                else:
                    continue
            c = Cluster()
            c.read(line)
            self.clusters.append(c)
        data = self.read_input(sys.stdin)
        for line in data:
            self.map(line)

    def map(self, line):
        #find cluster assignment by brute force
        doc = Document(line)
        cluster_uid = None
        sqdist_to_nearest = float('inf')
        for cluster_k in self.clusters:
            sqdist_k = MathUtil.compute_distance(map1 = cluster_k.tfidf, map2 = doc.tfidf, squared=True)
            if sqdist_k <= sqdist_to_nearest:
                cluster_uid = cluster_k.uid
        #dutifully emit.
        self.emit(key = cluster_uid, value = doc)
        return


if __name__ == '__main__':
    mapper = KmeansMapper()
    mapper.main()
