from __future__ import annotations
from typing import Union
import numpy as np
from skqlearn.utils import distance_estimation
from abc import abstractmethod, ABC


class GenericClustering(ABC):
    """Generic clustering algorithm used as base for K-Means and K-Medians.

    Through a series of epochs, the algorithm assigns each input sample/vector
    to the closest centroid. Then, the centroids are recalculated based on the
    metric implemented in each specific algorithm. The distance used is the
    Euclidean or :math:`L^2` norm.

    Attributes:
        cluster_centers (numpy.ndarray of shape (n_clusters, n_features)):
            Coordinates for the cluster centroids.
        labels (numpy.ndarray of shape (n_samples,)): Labels of each input
            sample.
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
        """Construct a GenericClustering object.

        Args:
            n_clusters (int): Number of clusters to form and centroids to
                generate.
            max_iterations (int): Max number of iterations of the algorithm.
            random_state (int, RandomState or None): Determines random number
                generation for centroid initialization. If provided as an int,
                it will be used as seed to make the randomness deterministic.
            distance_calculation_method ({'classic', 'quantum'}): The distance
                calculation method:

                * 'classic': Classically calculated using `np.linalg.norm`.
                * 'quantum': Quantum distance estimation will be used.
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
            ValueError: if any of the class parameters have invalid values.
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
            x (numpy.ndarray of shape (n_samples, n_features)): Input samples.

        Returns:
            numpy.ndarray of shape (n_clusters, n_features):
                Initial cluster centroids.

        Raises:
            ValueError: If there are not enough unique input samples to define
                different centroids.
        """
        unique_values = np.unique(x, axis=0)
        if len(unique_values) < self.n_clusters:
            raise ValueError('Invalid input samples. Number of unique samples'
                             f'must be at least n_clusters, '
                             f'got {len(unique_values)} instead.')
        indices = self.random_state.permutation(len(unique_values)
                                                )[:self.n_clusters]
        centroids = unique_values[indices]
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
            a (numpy.ndarray of shape (n_features,)): Input a.
            a_norm (float): L2-norm of input a. Not used, defined to conform to
                distance_estimation's prototype.
            b (numpy.ndarray of shape (n_features,)): Input b.
            b_norm (float): L2-norm of input b. Not used, defined to conform to
                distance_estimation's prototype.

        Returns:
            float:
                Euclidean distance between a and b.
        """
        return np.linalg.norm(a - b)

    @abstractmethod
    def _centroid_update(
            self,
            x: np.ndarray,
            x_norms: np.ndarray,
            labels: np.ndarray,
    ) -> np.ndarray:
        """Update function for the centroids.

        Args:
            x (numpy.ndarray of shape (n_samples, n_features)): Input samples.
            x_norms (numpy.ndarray of shape (n_samples)): L2-norm of every
                sample. Only needed if quantum estimation is used.
            labels (numpy.ndarray of shape (n_samples)): Assignments of each
                sample to each cluster.

        Returns:
            numpy.ndarray of shape (n_clusters, n_features):
                Updated cluster centroids.
        """
        pass

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
            x (numpy.ndarray of shape (n_samples, n_features)): Input data.
            x_norms (numpy.ndarray of shape (n_samples)): L2-norm of every
                instance. Only needed if quantum estimation is used.
            centroids (numpy.ndarray of shape (n_clusters, n_features)):
                Cluster centroids.
            centroid_norms (numpy.ndarray of shape (n_clusters)): L2-norm of
                centroids. Only needed if quantum estimation is used.

        Returns:
            numpy.ndarray of shape (n_samples,):
                Label for each of the instances provided.

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
            estimation_flag = False
        else:
            distance_fn = distance_estimation
            estimation_flag = True

        labels = np.zeros(x.shape[0])

        for i in range(x.shape[0]):
            centroid_distances = np.zeros(centroids.shape[0])
            for j in range(centroids.shape[0]):
                if estimation_flag and (x[i, :] == centroids[j, :]).all():
                    # Correction when computing with quantum estimation to
                    # avoid centroids being left with no samples assigned
                    # because of estimation errors.
                    labels[i] = -1  # Forces the wanted assignment
                    break

                distance = distance_fn(x[i, :], x_norms[i],
                                       centroids[j, :], centroid_norms[j])
                centroid_distances[j] = distance

            labels[i] = np.argmin(centroid_distances)

        return np.array(labels, dtype=int)

    def fit(
            self,
            x: np.ndarray,
    ) -> GenericClustering:
        """Compute clustering of provided data.

        Args:
            x (numpy.ndarray of shape (n_samples, n_features)): Training data
                to cluster.

        Returns:
            self (GenericClustering):
                Fitted estimator.
        """
        centroids = self._init_centroids(x)
        x_labels = np.ones(x.shape[0], dtype=int) * -1

        x_norms = np.linalg.norm(x, axis=1)
        centroid_norms = np.linalg.norm(centroids, axis=1)

        assignment_flag = False

        for iter_ in range(self.max_iterations):
            # Auxiliary variables to check for stopping condition and to keep
            # track of the instances belonging to each cluster
            label_change_flag = False
            cluster_data = {x: [] for x in range(self.n_clusters)}

            new_labels = self._data_labels(x, x_norms,
                                           centroids, centroid_norms)

            # Check if any of the instances has changed cluster
            if (new_labels != x_labels).any():
                label_change_flag = True

            x_labels = new_labels

            if label_change_flag:
                # Check that all centroids have at least 1 assigned sample.
                # If not, restart process (problem caused by doing quantum
                # estimations instead of real calculations)
                assignment_flag = False
                if len(np.unique(x_labels)) != self.n_clusters:
                    assignment_flag = True
                    break

                # Recalculation of centroids according to implemented
                # abstract method
                centroids = self._centroid_update(x, x_norms, x_labels)

                # Recalculation of centroid norms after their update
                centroid_norms = np.linalg.norm(centroids, axis=1)
            else:
                break

        if assignment_flag:
            # Encountered error provoked by incorrect distance estimation,
            # therefore the process must be repeated.
            return self.fit(x)
        else:
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
        """Compute clustering of provided data and predict cluster
        index for each sample.

        .. note::

           It is equivalent to calling `fit(x)` followed by `predict(x)`.

        Args:
            x (numpy.ndarray of shape (n_samples, n_features)): Training data
                to cluster and predict.

        Returns:
            numpy.ndarray of shape (n_samples,):
                Index of the cluster each sample belongs to.
        """
        return self.fit(x).predict(x)

    def predict(
            self,
            x: np.ndarray,
    ) -> np.ndarray:
        """Assigns a label to each input sample based on the closest centroid.

        Args:
            x (numpy.ndarray of shape (n_samples, n_features)): Data to
                predict.

        Returns:
            numpy.ndarray of shape (n_samples,):
                Index of the cluster each sample belongs to.
        """
        x_norms = np.linalg.norm(x, axis=1)

        x_labels = self._data_labels(x,
                                     x_norms,
                                     self.cluster_centers,
                                     self._centroid_norms)

        return x_labels
