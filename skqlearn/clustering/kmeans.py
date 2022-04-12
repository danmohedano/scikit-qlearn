from __future__ import annotations
from typing import Union
import numpy as np
from skqlearn.algorithms import distance_estimation
from ._kclusters import GenericClustering


class KMeans(GenericClustering):
    """K-Means clustering
    """
    def _centroid_update(
            self,
            x: np.ndarray,
            cluster_assignments: dict
    ) -> np.ndarray:
        """Update function for the centroids.

        Args:
            x (np.ndarray of shape (n_samples, n_features)): Input samples.
            cluster_assignments (dict): Index assignments for each cluster of
                each instance index. The dictionary is of the form
                {cluster_index: [instance_indices]}

        Returns:
            np.ndarray of shape (n_clusters, n_features): Updated cluster
                centroids.
        """
        centroids = np.zeros((self.n_clusters, x.shape[1]))
        for i in range(self.n_clusters):
            centroids[i] = x[cluster_assignments[i], :].mean(axis=0)

        return centroids
