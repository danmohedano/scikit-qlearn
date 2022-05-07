import numpy as np
from .kclusters import GenericClustering
from skqlearn.utils import distance_estimation


class KMedians(GenericClustering):
    r"""K-Means clustering algorithm based on the generic clustering algorithm
    structure.

    The centroids are updated after each epoch by computing the median of all
    input samples assigned to each centroid.

    The median being defined as the geometric median:

    .. math::

       arg\,min_{\boldsymbol{x}_j}\sum_{i=1}^m||\boldsymbol{x}_i -
       \boldsymbol{x}_j||_{2}

    Attributes:
        cluster_centers (numpy.ndarray of shape (n_clusters, n_features)):
            Coordinates for the cluster centroids.
        labels (numpy.ndarray of shape (n_samples,)): Labels of each input
            sample.
        n_features_in (int): Number of features seen during fit.
        n_iter (int): Number of iterations run.
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
            x (numpy.ndarray of shape (n_samples, n_features)): Input samples.
            x_norms (numpy.ndarray of shape (n_samples)): L2-norm of every
                instance. Only needed if quantum estimation is used.
            cluster_assignments (dict): Index assignments for each cluster of
                each instance index. The dictionary is of the form
                {cluster_index: [instance_indices]}

        Returns:
            numpy.ndarray of shape (n_clusters, n_features):
                Updated cluster centroids.
        """
        if self.distance_calculation_method == 'classic':
            distance_fn = self._distance
        else:
            distance_fn = distance_estimation

        # Calculate distances between instances only in the first execution.
        distances = getattr(self, '_sample_distances', None)
        if distances is None:
            distances = np.zeros([x.shape[0], x.shape[0]])
            for i in range(x.shape[0]):
                for j in range(x.shape[0]):
                    if i == j:
                        continue
                    distances[i, j] = distance_fn(x[i, :], x_norms[i],
                                                  x[j, :], x_norms[j])

            setattr(self, '_sample_distances', distances)

        # Update centroids through the median
        centroids = np.zeros((self.n_clusters, x.shape[1]))
        for cluster_idx in range(self.n_clusters):
            # Aggregate distances from each instance to every other distance
            # and obtain the median instance (only considering those contained
            # in the current centroid)
            instance_idxs = cluster_assignments[cluster_idx]
            dist_aggregate = np.sum(distances[instance_idxs, :]
                                    [:, instance_idxs],
                                    axis=0)
            median_idx = np.argmin(dist_aggregate)
            data_median_idx = cluster_assignments[cluster_idx][median_idx]

            # Obtain new centroid
            centroids[cluster_idx] = x[data_median_idx]

        return centroids
