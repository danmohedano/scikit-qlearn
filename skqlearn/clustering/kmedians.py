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
            labels: np.ndarray,
    ) -> np.ndarray:
        """Update function for the centroids.

        Calculates new cluster centroids as median of instances contained in
        each cluster. The median being defined as the instance with minimum
        distance to all other instances in the cluster (aggregated).

        Args:
            x (numpy.ndarray of shape (n_samples, n_features)): Input samples.
            x_norms (numpy.ndarray of shape (n_samples)): L2-norm of every
                instance. Only needed if quantum estimation is used.
            labels (numpy.ndarray of shape (n_samples)): Assignments of each
                sample to each cluster.

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
            distances = np.ones([x.shape[0], x.shape[0]]) * -1
            np.fill_diagonal(distances, 0)

        # Update centroids through the median
        centroids = np.zeros((self.n_clusters, x.shape[1]))
        for cluster_idx in range(self.n_clusters):
            # Calculate distance between all samples in the cluster only if
            # it has not already been done
            instance_idxs = labels == cluster_idx
            for i in range(instance_idxs.shape[0]):
                if not instance_idxs[i]:
                    continue
                for j in range(i + 1, instance_idxs.shape[0]):
                    if instance_idxs[j] and distances[i, j] < 0:
                        # If the position is marked as not evaluated with -1
                        distances[i, j] = distance_fn(x[i, :], x_norms[i],
                                                      x[j, :], x_norms[j])
                        distances[j, i] = distances[i, j]

            # Aggregate distances from each instance to every other distance
            # and obtain the median instance (only considering those contained
            # in the current centroid)
            dist_aggregate = np.sum(distances[instance_idxs, :]
                                    [:, instance_idxs],
                                    axis=0)

            # Obtain new centroid as the samples that minimizes the distances
            # to all other samples assigned to the centroid
            median_idx = np.argmin(dist_aggregate)
            centroids[cluster_idx] = x[instance_idxs, :][median_idx, :]

        # Store the calculations in case they are needed in the future
        self._sample_distances = distances

        return centroids
