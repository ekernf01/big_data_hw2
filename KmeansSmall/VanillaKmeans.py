import pandas as pd
import numpy as np
import ggplot as gg
import os
import datetime

class KmeansCall:
    """
    #data should be put in with a 'class_label' column, which is just a placeholder for output.
    #It needs to be the first column.
    """

    def __init__(self, data, num_clusters = 20, initialization = "km++"):
        self.initialization = initialization
        self.data = data
        self.nsample = data.shape[0]
        self.dimension = data.shape[1]
        self.num_clusters = num_clusters
        self.cluster_centers = []
        self.initialize_centers()
        self.obj = float("inf")

    def initialize_centers(self):
        if self.initialization == "km++":
            new_center = self.data.iloc[np.random.choice(a=self.nsample), 1:self.dimension]
            self.cluster_centers.append(new_center)
            for smaller_than_k in range(self.num_clusters - 1):
                dist_sq = self.update_labels(smaller_than_k)
                dist_sq = dist_sq / np.sum(dist_sq)
                new_center = self.data.iloc[np.random.choice(a=self.nsample, p=dist_sq), 1:self.dimension]
                self.cluster_centers.append(new_center)
        else:
            rand_idx = np.random.choice(a=self.nsample, size=self.num_clusters, replace=False)
            self.cluster_centers = [self.data.iloc[rand_idx[k], 1:self.dimension] for k in range(self.num_clusters)]

    def update_cluster_centers(self):
        """
        Wipes the cluster centers and recalculates them based on the labeled data.
        :return:
        """
        for k in range(self.num_clusters):
            for i in range(self.dimension):
                self.cluster_centers[i] = 0
        counts = [0 for k in range(self.num_clusters)]
        #get sum
        for i, row in self.data.iterrows():
            k = self.data["class_label"][i]
            counts[k] += 1
            self.cluster_centers[k] += row[1:self.dimension] #everything but the label
        #Divide sum by count to get average, or if count is zero, reinitialize to a random datapoint.
        for k in range(self.num_clusters):
            if counts[k]==0:
                rand_idx = np.random.choice(self.nsample)
                self.cluster_centers[k] = self.data.iloc[rand_idx, 1:self.dimension]
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
                #indexing on row gets everything but the class label
                cur_dist_sq = np.linalg.norm(self.cluster_centers[k] - row[1:len(row)]) ** 2
                if cur_dist_sq < min_dist_sq:
                    min_dist_sq = cur_dist_sq
                    self.data.loc[i, "class_label"] = k
                min_dist_sq_records[i] = min_dist_sq
        return min_dist_sq_records


    def get_objective(self):
        obj_by_cluster = [0 for k in range(self.num_clusters)]
        for i, row in self.data.iterrows():
            k = self.data["class_label"][i]
            #indexing on row gets everything but the class label
            obj_by_cluster[k] += np.linalg.norm(self.cluster_centers[k] - row[1:self.dimension]) ** 2
        return np.sum(obj_by_cluster)


    def run_kmeans(self, maxiter = 100, verbose = False):
        """
        :param data: pd dataframe, one row per datum
        :param num_clusters: positive integer
        :param maxiter: positive integer
        :return: data, cluster_centers: data is the same pd dataframe, but with an
        extra column "cluster_labels". cluster_centers is a num_clusters-by-dimension
        dataframe with one cluster center per row.
        """
        reached_tol = False
        num_points = self.data.shape[0]
        self.data["class_label"] = [np.random.choice(range(self.num_clusters)) for i in range(num_points)]
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

        #Warn if necessary, fit a type problem, and exit
        self.data.class_label = self.data.class_label.astype(str)
        if not reached_tol:
            warnings.warn("Max iteration limit (%d) exceeded." % maxiter)
        return

