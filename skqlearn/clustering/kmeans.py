import numpy as np
from ._kclusters import GenericClustering


class KMeans(GenericClustering):
    """K-Means clustering
    """

    def _centroid_update(
            self,
            x: np.ndarray,
            x_norms: np.ndarray,
            cluster_assignments: dict,
    ) -> np.ndarray:
        """Update function for the centroids.

        Calculates new cluster centroids as mean of instances contained in each
        cluster.

        Args:
            x (np.ndarray of shape (n_samples, n_features)): Input samples.
            x_norms (np.ndarray of shape (n_samples)): L2-norm of every
                instance. Only needed if quantum estimation is used.
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
