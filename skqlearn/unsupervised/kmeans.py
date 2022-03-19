from __future__ import annotations
from typing import Union
import numpy as np
from skqlearn.utils import distance_estimation


class KMeans:
    """K-Means clustering

    Attributes:
        cluster_centers (np.ndarray of shape (n_clusters, n_features)):
            Coordinates for the cluster centroids.
        labels (np.ndarray of shape (n_samples,)): Labels of each point.
        n_features_in (int): Number of features seen during fit.
        n_iter (int): Number of iterations run.
    """
    def __init__(
            self,
            n_clusters: int,
            max_iterations: int,
            random_state: Union[int, np.random.RandomState, None],
            distance_calculation_method: str,
    ):
        """Inits KMeans object.

        Args:
            n_clusters (int): Number of clusters to form and centroid to
                generate.
            max_iterations (int): Maximum number of iterations of the k-means
                algorithm.
            random_state (int, RandomState or None): Determines random number
                generation for centroid initialization. If provided as an int,
                it will be used as seed to make the randomness deterministic.
            distance_calculation_method ({'classic', 'quantum'}): The distance
                calculation method:
                'classic': Regular euclidean distance will be used.
                'quantum': Quantum distance estimation will be used.
        """
        self.n_clusters = n_clusters
        self.max_iterations = max_iterations
        self.random_state = random_state
        self.distance_calculation_method = distance_calculation_method
        self._check_params()
        self.cluster_centers = None
        self.labels = None
        self.n_features_in = None
        self.n_iter = None
        self._centroid_norms = None

    def _check_params(self):
        """Check for the correctness of the parameters provided and processes
        them.

        Raises:
            ValueError:
        """
        if self.n_clusters <= 0:
            raise ValueError(f'n_clusters should be > 0, got {self.n_clusters}'
                             ' instead.')
        if self.max_iterations <= 0:
            raise ValueError('max_iterations should be > 0, got '
                             f'{self.max_iterations} instead.')
        if not self.random_state:
            self.random_state = np.random.RandomState()
        elif isinstance(self.random_state, int):
            self.random_state = np.random.RandomState(self.random_state)
        elif not isinstance(self.random_state, np.random.RandomState):
            raise ValueError('Invalid type for random_state, got '
                             f'{type(self.random_state)}')
        if self.distance_calculation_method not in ['classic', 'quantum']:
            raise ValueError("distance_calculation method should be "
                             "{'classic', 'quantum'}, got "
                             f"{self.distance_calculation_method} instead.")

    def _init_centroids(
            self,
            x: np.ndarray,
    ) -> np.ndarray:
        """Computation of initial centroids.

        Args:
            x (np.ndarray of shape (n_samples, n_features)): Input samples.

        Returns:
            np.ndarray of shape (n_clusters, n_features): Initial cluster
                centroids.
        """
        indices = self.random_state.permutation(x.shape[0])[:self.n_clusters]
        centroids = x[indices]
        return centroids

    def _distance(
            self,
            a: np.ndarray,
            a_norm: float,
            b: np.ndarray,
            b_norm: float,
    ) -> float:
        """Euclidean distance between two data points.

        Args:
            a: Input a.
            a_norm: L2-norm of input a. Not used, defined to conform to
                distance_estimation's prototype.
            b: Input b.
            b_norm: L2-norm of input b. Not used, defined to conform to
                distance_estimation's prototype.

        Returns:
            float: Euclidean distance between a and b.
        """
        return np.linalg.norm(a - b)

    def _data_labels(
            self,
            x: np.ndarray,
            x_norms: np.ndarray,
            centroids: np.ndarray,
            centroid_norms: np.ndarray,
    ) -> np.ndarray:
        """Calculates the distances between each point and all centroids and
        determines the closest one.

        Args:
            x (np.ndarray of shape (n_samples, n_features)): Input data.
            x_norms (np.ndarray of shape (n_samples)): L2-norm of every
                instance. Only needed if quantum estimation is used.
            centroids (np.ndarray of shape (n_clusters, n_features)): Cluster
                centroids.
            centroid_norms (np.ndarray of shape (n_clusters)): L2-norm of
                centroids. Only needed if quantum estimation is used.

        Returns:
            np.ndarray of shape (n_samples,): Label for each of the instances
                provided

        Raises:
            ValueError: If dimension mismatch between x and centroids, x and
                x_norms or centroids and centroid_norms.
        """
        if x.shape[1] != centroids.shape[1]:
            raise ValueError('n_features should match between x and centroids '
                             f', got {x.shape} and {centroids.shape} instead.')
        if x.shape[0] != x_norms.shape[0]:
            raise ValueError('n_samples should match between x and x_norms '
                             f', got {x.shape} and {x_norms.shape} instead.')
        if centroids.shape[0] != centroid_norms.shape[0]:
            raise ValueError('n_clusters should match between centroids and '
                             f'centroid_norms, got {centroids.shape} and '
                             f'{centroid_norms.shape} instead.')

        if self.distance_calculation_method == 'classic':
            distance_fn = self._distance
        else:
            distance_fn = distance_estimation

        labels = np.array(x.shape[0])

        for i in range(x.shape[0]):
            centroid_distances = []
            for j in range(centroids.shape[0]):
                distance = distance_fn(x[i, :], x_norms[i, :],
                                       centroids[j, :], centroid_norms[j, :])
                centroid_distances.append(distance)

            labels[i] = np.argmin(centroid_distances)

        return labels

    def fit(
            self,
            x: np.ndarray,
    ) -> KMeans:
        """Compute k-means clustering of provided data.

        Args:
            x (np.ndarray of shape (n_samples, n_features)): Training data to
                cluster.

        Returns:
            self (object): Fitted estimator.
        """
        centroids = self._init_centroids(x)
        x_labels = np.ones(x.shape[0]) * -1

        x_norms = [np.linalg.norm(x[i, :]) for i in range(x.shape[0])]
        x_norms = np.array(x_norms)
        centroid_norms = [np.linalg.norm(centroids[i, :])
                          for i in range(centroids.shape[0])]
        centroid_norms = np.array(centroid_norms)

        for iter_ in range(self.max_iterations):
            # Auxiliary variables to check for stopping condition and to keep
            # track of the instances belonging to each cluster
            label_change_flag = False
            cluster_data = [[] * self.n_clusters]

            new_labels = self._data_labels(x, x_norms, centroids, centroid_norms)
            for i in range(x.shape[0]):
                if new_labels[i] != x_labels[i]:
                    label_change_flag = True

                cluster_data[new_labels[i]].append(i)
                x_labels[i] = new_labels[i]

            if label_change_flag:
                # Recalculation of centroids as mean of each instance belonging
                # to the cluster
                for i in range(self.n_clusters):
                    centroids[i] = x[cluster_data[i], :].mean(axis=0)

                # Recalculation of centroid norms after their update
                centroid_norms = [np.linalg.norm(centroids[i, :])
                                  for i in range(centroids.shape[0])]
                centroid_norms = np.array(centroid_norms)
            else:
                break

        self.cluster_centers = centroids
        self._centroid_norms = centroid_norms
        self.labels = x_labels
        self.n_iter = iter_
        self.n_features_in = x.shape[1]
        return self

    def fit_predict(
            self,
            x: np.ndarray,
    ) -> np.ndarray:
        """Compute k-means clustering of provided data and predict cluster
        index for each sample.

        Convenience method: It is equivalent to calling fit(x) followed by
        predict(x).

        Args:
            x (np.ndarray of shape (n_samples, n_features)): Training data to
                cluster and predict.

        Returns:
            ndarray of shape (n_samples,): Index of the cluster each sample
                belongs to.
        """
        return self.fit(x).predict(x)

    def predict(
            self,
            x: np.ndarray,
    ) -> np.ndarray:
        """Predict the closest cluster each sample in x belongs to.

        Args:
            x (np.ndarray of shape (n_samples, n_features)): Data to predict.

        Returns:
            ndarray of shape (n_samples,): Index of the cluster each sample
                belongs to.
        """
        x_norms = [np.linalg.norm(x[i, :]) for i in range(x.shape[0])]
        x_norms = np.array(x_norms)

        x_labels = self._data_labels(x,
                                     x_norms,
                                     self.cluster_centers,
                                     self._centroid_norms)

        return x_labels
