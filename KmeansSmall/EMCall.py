import pandas as pd
import numpy as np
import ggplot as gg
import os
import datetime
import warnings
from VanillaKmeans import KmeansCall
from scipy.stats import multivariate_normal as mvn

class EMCall(KmeansCall):
    """
    Inherits from class KmeansCall, but instead does gaussian mixture modeling via EM.
    """
    def __init__(self, data, num_clusters, initialization= "regular", init_centers_path = None, cov_reg = 0, verbose = False):
        KmeansCall.__init__(self, data, num_clusters, initialization, init_centers_path)
        self.membership_probs  = np.zeros(shape = (self.nsample, self.num_clusters))
        self.xgivenz = np.zeros(shape = (self.nsample, self.num_clusters))
        self.cluster_probs = np.ones(self.num_clusters) / float(self.num_clusters)
        self.cluster_covs = [np.identity(self.dimension) for k in range(self.num_clusters)]
        self.cov_reg = cov_reg
        self.verbose = verbose
        return

    def plot(self):
        prob231g_plot_df = self.data.copy()
        for k in range(self.num_clusters):
            n = prob231g_plot_df.shape[0]
            prob231g_plot_df.loc[n] = self.cluster_centers[k]
        prob231g_plot_df["class_label"] = [label for label in self.class_label] + \
                                          self.num_clusters * ["center"]
        p = gg.ggplot(prob231g_plot_df, gg.aes(x= "x1", y="x2", colour="class_label")) + \
            gg.geom_point() + gg.ggtitle("EM cluster assignments")
        print p
        return

    #x is the observed data
    #z is the latent class membership
    def update_xgivenz(self):
        for k in range(self.num_clusters):
            for i, row in self.data.iterrows():
                # (x - \mu)^T \hat \Sigma^{-1} (x - \mu)
                self.xgivenz[i, k] = mvn.pdf(row, mean=self.cluster_centers[k],
                                                  cov= self.cluster_covs[k])
        return

    def update_membership_probs(self):
        self.update_xgivenz()
        for i in range(self.nsample):
            post = [self.xgivenz[i, k] * self.cluster_probs[k] for k in range(self.num_clusters)]
            self.membership_probs[i, :] = post / sum(post)
            #self.class_label[i] = np.random.choice(range(self.num_clusters), size = 1, p = post / sum(post))
            self.class_label[i] =  post.index(max(post))
        return

    def update_cluster_means(self):
        for k in range(self.num_clusters):
            for i in range(self.dimension):
                self.cluster_centers[k][i] = 0
        counts = [0 for k in range(self.num_clusters)]
        #get sum
        for i, row in self.data.iterrows():
            for k in range(self.num_clusters):
                counts[k] += self.membership_probs[i, k]
                assert i < self.membership_probs.shape[0]
                self.cluster_centers[k] += self.membership_probs[i, k] * row
        #Divide sum by count to get average, or if count is zero, reinitialize to a random datapoint.
        for k in range(self.num_clusters):
            self.cluster_centers[k] = self.cluster_centers[k] / counts[k]
        return

    def update_cluster_covs(self):
        #zero out current covar to avoid mem allocation
        for k in range(self.num_clusters):
            for d1 in range(self.dimension):
                for d2 in range(self.dimension):
                    self.cluster_covs[k][d1, d2] = 0

        counts = [0 for k in range(self.num_clusters)]
        #get maximizer of expected loglik given cluster memberships:
        #  (\sum_i r_ik (x_i - \mu_k)(x_i - \mu_k)^T) / \sum_i r_ik
        #get numerator
        for i, row in self.data.iterrows():
            for k in range(self.num_clusters):
                counts[k] += self.membership_probs[i, k]
                centered = row - self.cluster_centers[k]
                self.cluster_covs[k] = self.cluster_covs[k] + np.outer(centered, centered*self.membership_probs[i, k])
        # divide by number of contributions and regularize
        for k in range(self.num_clusters):
            self.cluster_covs[k] = (1 - self.cov_reg) * self.cluster_covs[k] / counts[k]
            for d in range(self.dimension):
                self.cluster_covs[k][d, d] += self.cov_reg
        return

    def update_cluster_probs(self):
        self.cluster_probs = np.sum(self.membership_probs, axis = 0)
        return

    def update_cluster_params(self):
        self.update_cluster_means()
        self.update_cluster_covs ()
        self.update_cluster_probs()
        return

    def update_log_likelihood(self):
        # \prod_i Pr(x)
        # = \prod_i \sum_k P(x_i, z = k)
        # = \prod_i \sum_k P(x_i|z = k)\pi_k
        self.obj = 0
        for i in range(self.nsample):
            lik_i = 0
            for k in range(self.num_clusters):
                lik_i += self.xgivenz[i, k] * self.cluster_probs[k]
            self.obj += np.log(lik_i)

        return self.obj

    def run_em(self, maxiter = 100):
        reached_tol = False
        for i in range(maxiter):
            if self.verbose:
                self.plot()
            self.update_membership_probs()
            self.update_cluster_params()
            old_obj = self.obj
            self.update_log_likelihood()
            self.obj_record.append(self.obj)
            if abs(self.obj - old_obj) < 0.0000001:
                reached_tol = True
                break

        self.class_label = [str(label) for label in self.class_label]
        if not reached_tol:
            warnings.warn("Max iteration limit of (%d) exceeded." % maxiter)
        return