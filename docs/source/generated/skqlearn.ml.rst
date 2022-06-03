Machine Learning
===========================

This module contains submodules related to quantum machine learning.


Quantum-enhanced Clustering Algorithms
---------------------------------------

The `clustering` submodule contains implementations of basic distance-based
clustering algorithms with the objective of illustrating and comparing
classical distance calculation with quantum distance estimation.

.. autosummary::
   :toctree: autosummary

   skqlearn.ml.clustering.GenericClustering
   skqlearn.ml.clustering.KMeans
   skqlearn.ml.clustering.KMedians


Quantum-inspired Kernels
---------------------------

The `kernel` submodule contains kernel functions inspired by quantum properties.
Unlike the kernels defined by the quantum encoding methods, these are only
computed classically.

.. autosummary::
   :toctree: autosummary

   skqlearn.ml.kernels.SqueezingKernel
