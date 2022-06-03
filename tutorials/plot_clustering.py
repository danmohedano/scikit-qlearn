"""

.. _clusteringtutorial:

Clustering
===========================

This tutorial aims to explain and visualize the use of the clustering
algorithms implemented in the package.
"""
###############################################################################
# Introduction
# -------------
#
# Clustering is a form of unsupervised learning where data samples are grouped
# in clusters by using some metric of similarity between the samples.
#
# In this package, the two algorithms implemented are distance-based. This
# means that the similarity measure being used to group the samples is
# the Euclidean distance or L2-norm.
#
# The distances in the algorithms can be computed classically or estimated
# with a quantum subroutine.


###############################################################################
# In order to represent the cluster assignments, a utility function is defined
# that plots the data assignments as well as the center of each cluster.

from skqlearn.utils import JobHandler
from skqlearn.ml.clustering import *
import matplotlib.pyplot as plt
import numpy as np
from qiskit.providers.aer import AerSimulator
from sklearn.datasets import make_blobs


def plot_cluster(axis, x, labels, centers):
    legend = []
    for y in np.unique(labels):
        members = labels == y
        axis.scatter(x[members, 0], x[members, 1], label='_nolegend_')
        axis.scatter(centers[y, 0], centers[y, 1], s=[100], marker='X')
        legend.append(f'Centroid {y}')

    axis.set(xlabel='X1', ylabel='X2')
    axis.set_aspect('auto', 'box')
    axis.legend(legend)


def plot_comparison(x, clf_classic, clf_quantum, title):
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    fig.suptitle(title)
    ax1.set_title('Classic Distance Calculation')
    ax2.set_title('Quantum Distance Estimation')
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.85, wspace=0.1)

    plot_cluster(ax1, x, clf_classic.labels, clf_classic.cluster_centers)
    plot_cluster(ax2, x, clf_quantum.labels, clf_quantum.cluster_centers)

    plt.show()


###############################################################################
# Because the tutorial will also test the quantum distance estimation
# algorithm, the quantum backend must be configured.

JobHandler().configure(backend=AerSimulator(), shots=50000)

###############################################################################
# First, random data is generated centered around two points.

np.random.seed(1)
centers = np.array([[0, 0], [0, 1]])
x, labels = make_blobs(n_samples=20, centers=centers, cluster_std=0.1)

plot_cluster(plt.gca(), x, labels, centers)
plt.title('Generated data for two blobs')
plt.show()

###############################################################################
# KMeans
# --------
#
# The implementation of the KMeans algorithm is based on an iterative
# refinement technique. The cluster centers are initially chosen at random
# between the trainging data. After that, the algorithm assigns each data
# sample to the closest centroid (cluster center) and recalculates the centroid
# as the mean of all sample data assigned to it. This process is repeated until
# there is no change of assignments between two iterations or the iteration
# limit is reached.

###############################################################################
# The first step will be to train the algorithm with the data samples. During
# the training, the values for the centroids are stored in the object, as well
# as the final labels/assignments for the data samples. This allows us to
# see how the algorithm has decided to cluster the data samples. The distance
# calculation method can be provided in the constructor of the class, in order
# to choose between the classic and quantum calculations.

k_means_classic = KMeans(n_clusters=len(centers),
                         max_iterations=20,
                         random_state=0,
                         distance_calculation_method='classic')
k_means_classic.fit(x)

k_means_quantum = KMeans(n_clusters=len(centers),
                         max_iterations=20,
                         random_state=0,
                         distance_calculation_method='quantum')
k_means_quantum.fit(x)

plot_comparison(x, k_means_classic, k_means_quantum,
                'KMeans Clustering on two blobs')

###############################################################################
# The trained estimators can then be used to assign new data to the current
# clusters by using the `predict` method.

new_data = np.array([[-1, 0], [0, 2]])
print(k_means_classic.predict(new_data))

###############################################################################
# KMedians
# --------
#
# KMedians follows the same structure as KMeans. The only difference between
# the two algorithms is how the centroids are updated after every iteration.
# In KMedians, the centroid is calculated as the median of all data samples
# assigned to it, with the median being defined as the sample that minimizes
# the distance to all other samples:
#
# .. math::
#    arg\,min_{\boldsymbol{x}_j}\sum_{i=1}^m||\boldsymbol{x}_i -
#    \boldsymbol{x}_j||_{2}

k_medians_classic = KMedians(n_clusters=len(centers),
                             max_iterations=20,
                             random_state=0,
                             distance_calculation_method='classic')
k_medians_classic.fit(x)

k_medians_quantum = KMedians(n_clusters=len(centers),
                             max_iterations=20,
                             random_state=0,
                             distance_calculation_method='quantum')
k_medians_quantum.fit(x)

plot_comparison(x, k_medians_classic, k_medians_quantum,
                'KMedians Clustering on two blobs')

###############################################################################
# It is also interesting to see the decisions made by the algorithms with
# less straight forward data.

centers_2 = np.array([[0, 0]])
x_2, labels_2 = make_blobs(n_samples=20, centers=centers_2, cluster_std=0.05)

plot_cluster(plt.gca(), x_2, labels_2, centers_2)
plt.title('Generated Data for one blob')
plt.show()

###############################################################################

k_means_classic_2 = KMeans(n_clusters=2,
                           max_iterations=40,
                           random_state=0,
                           distance_calculation_method='classic')
k_means_classic_2.fit(x_2)

k_means_quantum_2 = KMeans(n_clusters=2,
                           max_iterations=40,
                           random_state=0,
                           distance_calculation_method='quantum')
k_means_quantum_2.fit(x_2)

plot_comparison(x_2, k_means_classic_2, k_means_quantum_2,
                'KMeans Clustering on one blob')

###############################################################################

k_medians_classic_2 = KMedians(n_clusters=2,
                               max_iterations=40,
                               random_state=0,
                               distance_calculation_method='classic')
k_medians_classic_2.fit(x_2)

k_medians_quantum_2 = KMedians(n_clusters=2,
                               max_iterations=40,
                               random_state=0,
                               distance_calculation_method='quantum')
k_medians_quantum_2.fit(x_2)

plot_comparison(x_2, k_medians_classic_2, k_medians_quantum_2,
                'KMedians Clustering on one blob')
