Utils
======================

This module contains utility functions and classes that support the rest of the package.


**Job handling**

In order to facilitate the configuration of the quantum backends where the quantum subroutines are execute,
a singleton class has been implemented. This must be configured before trying to perform any quantum computation.

.. autosummary::
   :toctree: autosummary

   skqlearn.utils.JobHandler

**Quantum Subroutines**

.. autosummary::
   :toctree: autosummary

   skqlearn.utils.fidelity_estimation
   skqlearn.utils.distance_estimation
   skqlearn.utils.inner_product_estimation

**Others**

.. autosummary::
   :toctree: autosummary

   skqlearn.utils.InteractiveBlochSphere