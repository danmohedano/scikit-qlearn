import numpy as np
from ._kclusters import GenericClustering
from skqlearn.algorithms import distance_estimation


class KMedians(GenericClustering):
    """K-Medians clustering
    """

    def _centroid_update(
            self,
            x: np.ndarray,
            x_norms: np.ndarray,
            cluster_assignments: dict,
    ) -> np.ndarray:
        """Update function for the centroids.

        Calculates new cluster centroids as median of instances contained in
        each cluster. The median being defined as the instance with minimum
        distance to all other instances in the cluster (aggregated).

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
        if self.distance_calculation_method == 'classic':
            distance_fn = self._distance
        else:
            distance_fn = distance_estimation

        centroids = np.zeros((self.n_clusters, x.shape[1]))
        for cluster_idx in range(self.n_clusters):
            distances = np.zeros((len(cluster_assignments[cluster_idx]),
                                  len(cluster_assignments[cluster_idx])))

            # Calculate distance between every pair of instances
            for i in range(len(cluster_assignments[cluster_idx])):
                data_i = x[cluster_assignments[cluster_idx][i], :]
                norm_i = x_norms[cluster_assignments[cluster_idx][i]]
                for j in range(len(cluster_assignments[cluster_idx])):
                    if i == j:
                        continue
                    data_j = x[cluster_assignments[cluster_idx][j], :]
                    norm_j = x_norms[cluster_assignments[cluster_idx][j]]
                    distances[i, j] = distance_fn(data_i, norm_i,
                                                  data_j, norm_j)

            # Aggregate distances from each instance to every other distance
            # and obtain the median instance
            dist_aggregate = np.sum(distances, axis=0)
            median_idx = np.argmin(dist_aggregate)
            data_median_idx = cluster_assignments[cluster_idx][median_idx]

            # Obtain new centroid
            centroids[cluster_idx] = x[data_median_idx]

        return centroids
