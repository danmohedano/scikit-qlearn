import numpy as np
from .kclusters import GenericClustering


class KMeans(GenericClustering):
    r"""K-Means clustering algorithm based on the generic clustering algorithm
    structure.

    The centroids are updated after each epoch by computing the mean of all
    input samples assigned to each centroid.

    .. math::
       \boldsymbol{C}_i=\frac{1}{|\{\boldsymbol{C}_i\}|}\sum_{\boldsymbol{x}_j
       \in \{\boldsymbol{C}_i\}}\boldsymbol{x}_j

    With :math:`\{\boldsymbol{C}_i\}` being the set of vectors assigned to the cluster
    centroid :math:`\boldsymbol{C}_i`.

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

        Calculates new cluster centroids as mean of instances contained in each
        cluster.

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
        centroids = np.zeros((self.n_clusters, x.shape[1]))
        for i in range(self.n_clusters):
            centroids[i] = x[labels == i, :].mean(axis=0)

        return centroids
