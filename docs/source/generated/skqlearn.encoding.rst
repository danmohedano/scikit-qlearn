Data Encoding
=========================
The module contains a series of data encoding methods that allow the user to map real data into quantum states, described by their amplitude vectors. These mappings can be defined as feature maps of the form:

.. math::
    \phi: \mathcal{X} \rightarrow \mathcal{F}

Where the input :math:`x` is mapped from the input space into the feature space.

The inner product of two inputs mapped to feature space defines a kernel via:

.. math::
   k(x,x'):= \left<\phi(x), \phi(x')\right>_\mathcal{F}

Where :math:`\left<.,.\right>_\mathcal{F}` is the inner product defined on :math:`\mathcal{F}`. :cite:`hilbert2019`

In this case, as the inputs are being mapped into quantum states, the kernel defined is of the form:

.. math::
   k(x,x')=\left<\phi(x)|\phi(x')\right>

Encodings
---------

To allow the expansion of the module in the future, an abstract class `Encoding` has been defined. To implement a new encoding, one must simply define a child class that implements the `encoding` method.

.. autosummary::
   :toctree: autosummary

   skqlearn.encoding.Encoding

**Real data encodings**

.. autosummary::
   :toctree: autosummary

   skqlearn.encoding.AmplitudeEncoding
   skqlearn.encoding.ExpandedAmplitudeEncoding
   skqlearn.encoding.AngleEncoding
   skqlearn.encoding.BasisEncoding

**Probability distribution encodings**

.. autosummary::
   :toctree: autosummary

   skqlearn.encoding.QSampleEncoding
