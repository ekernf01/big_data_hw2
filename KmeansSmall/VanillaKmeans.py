import pandas as pd
import numpy as np
import ggplot as gg
import os
import datetime
import csv
import warnings

class KmeansCall:
    """
    #data should be put in with a 'class_label' column, which is just a placeholder for output.
    #It needs to be the first column.
    """

    def __init__(self, data, class_label, num_clusters = 20, initialization = "km++", init_centers_path = None):
        self.initialization = initialization
        self.data = data
        self.class_label = class_label
        self.nsample = data.shape[0]
        self.dimension = data.shape[1]
        self.num_clusters = num_clusters
        self.cluster_centers = []
        self.initialize_centers(init_centers_path)
        self.obj = float("inf")

    def initialize_centers(self, init_centers_path):
        if self.initialization == "km++":
            new_center = list(self.data.iloc[np.random.choice(a=self.nsample), :])
            self.cluster_centers.append(new_center)
            for smaller_than_k in range(self.num_clusters - 1):
                dist_sq = self.update_labels(smaller_than_k)
                dist_sq = dist_sq / np.sum(dist_sq)
                new_center = list(self.data.iloc[np.random.choice(a=self.nsample, p=dist_sq), :])
                self.cluster_centers.append(new_center)
        elif self.initialization == "specified":
            center_file = open(init_centers_path, "r")
            def line2list(line):
                return list([float(token.strip()) for token in line[0].split()])
            self.cluster_centers = [line2list(line) for line in csv.reader(center_file)]
        elif self.initialization == "firstpoints":
            self.cluster_centers = [list(self.data.iloc[k, :]) for k in range(self.num_clusters)]
        else:
            rand_idx = np.random.choice(a=self.nsample, size=self.num_clusters, replace=False)
            self.cluster_centers = [list(self.data.iloc[rand_idx[k], :]) for k in range(self.num_clusters)]

    def update_cluster_centers(self):
        """
        Wipes the cluster centers and recalculates them based on the labeled data.
        :return:
        """
        for k in range(self.num_clusters):
            for i in range(self.dimension):
                self.cluster_centers[k][i] = 0
        counts = [0 for k in range(self.num_clusters)]
        #get sum
        for i, row in self.data.iterrows():
            k = self.class_label[i]
            counts[k] += 1
            self.cluster_centers[k] += row
        #Divide sum by count to get average, or if count is zero, reinitialize to a random datapoint.
        for k in range(self.num_clusters):
            if counts[k]==0:
                rand_idx = np.random.choice(self.nsample)
                warnings.warn("Re-initializing an empty cluster to a random datum.")
                self.cluster_centers[k] = list(self.data.iloc[rand_idx, :])
            else:
                self.cluster_centers[k] = self.cluster_centers[k] / counts[k]
        return

    def update_labels(self, smaller_than_k = None):
        """
        Reassigns each point to the closest cluster center.
        :param: smaller_than_k: part of retrofitting to
        also help with KMeans++ initialization, this argument
        lets you assign points using only the first smaller_than_k clusters.
        :return: min_dist_sq_records also helps with kmeans++.
        """
        min_dist_sq_records = np.zeros(self.nsample)
        if smaller_than_k is None:
            smaller_than_k = self.num_clusters
        for i, row in self.data.iterrows():
            min_dist_sq = float("inf")
            for k in range(smaller_than_k):
                cur_dist_sq = np.linalg.norm(self.cluster_centers[k] - row) ** 2
                if cur_dist_sq < min_dist_sq:
                    min_dist_sq = cur_dist_sq
                    self.class_label[i] = k
                min_dist_sq_records[i] = min_dist_sq
        return min_dist_sq_records


    def get_objective(self):
        obj_by_cluster = [0 for k in range(self.num_clusters)]
        for i, row in self.data.iterrows():
            k = self.class_label[i]
            obj_by_cluster[k] += np.linalg.norm(self.cluster_centers[k] - row) ** 2
        return np.sum(obj_by_cluster)


    def run_kmeans(self, maxiter = 100, verbose = False):
        """
        :param data: pd dataframe, one row per datum
        :param num_clusters: positive integer
        :param maxiter: positive integer
        :return: data, cluster_labels, cluster_centers: The first two are the same, but with contents of
        cluster_labels altered. cluster_centers is a num_clusters-by-dimension
        dataframe with one cluster center per row.
        """
        reached_tol = False
        num_points = self.data.shape[0]
        self.class_label = [0 for i in range(num_points)]
        for i in range(maxiter):
            self.update_labels()
            self.update_cluster_centers()
            if verbose and i % 10 == 1:
                print "In iteration " + str(i) + " , objective is " + str(obj)
            #Save time by only checking objective every 5 iterations
            if i % 5 == 0:
                old_obj = self.obj
                self.obj = self.get_objective()
                if abs(self.obj - old_obj) < 0.0000001:
                    reached_tol = True
                    break

        #Warn if necessary, fix a type problem, and exit
        self.class_label = [str(label) for label in self.class_label]
        if not reached_tol:
            warnings.warn("Max iteration limit (%d) exceeded." % maxiter)
        return

