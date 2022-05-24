"""

.. _encodingstutorial:

Data Encodings
===========================

This tutorial aims to explain and visualize the different data encoding methods
implemented in the package.
"""
###############################################################################
# Introduction
# -------------
#
# Machine Learning algorithms rely on sample data in order to train and learn.
# When trying to enhance these algorithms with quantum subroutines, the first
# issue one can encounter is how to translate the classical sample data into
# a format that can be understood by a quantum machine.
#
# While classical computers' most basic unit of information is the bit, quantum
# machines' is the qubit. Classical bits can only hold/represent one of two
# values, 0 or 1, as all information can be represented with combinations of
# enough of these binary values. Qubits, on the other hand, can be thought of
# as a 2-state system such as a spin-half . The state of these systems can then
# be described with quantum states. A system is said to have *n* qubits if it
# has a Hilbert space of :math:`N=2^n` dimensions and, thus, has :math:`2^n`
# mutually orthogonal quantum states, denoted as basis states
# :cite:`steane1998quantum`. The main difference with classical mechanics is
# that the system, by superposition, can be in a linear combination of these
# basis states, represented as unit vectors of the form
# :math:`x_1=[1,0,...,0]^T`.
#
# Therefore, a generic state :math:`\left|\psi\right>` can be defined as a
# combination of the basis states
# :math:`\{\left|x_1\right>,...,\left|x_N\right>\}`.
#
# .. math::
#    \left|\psi\right> = c_1\left|x_1\right>+...+c_N\left|x_N\right>
#
# The weights of the linear combination, :math:`c_i\in\mathbb{C}`, are called
# complex amplitudes or probability amplitudes. The norm square of these
# defines the probability of the system being found in each of the basis states
# after measurement. Because of this, the probability amplitudes need to be
# normalized in order to define a proper quantum state.
# A quantum state can thus be represented as its amplitude
# vector :math:`[c_1,...,c_N]^T`.
#
# The following encoding methods describe how to encode the sample data into
# an amplitude vector in order to represent a quantum state. These methods
# can be defined as feature maps of the form:
#
# .. math::
#    \phi: \mathcal{X} \rightarrow \mathcal{F}
#
# Where the input :math:`x` is mapped from the input space into the feature
# space.
#
# In order to visually represent the transformations, another representation of
# quantum states will be used: the Bloch sphere. This is a geometrical
# representation of a qubit where its quantum state is shown as a point in the
# unit sphere. The antipodal points correspond to the two basis states
# :math:`\left|0\right>` and :math:`\left|1\right>`. The translation into
# spherical coordinates is then made by defining the quantum state as:
#
# .. math::
#    \left|\psi\right> = c_0\left|0\right>+c_1\left|1\right> = \cos{
#    \frac{\theta}{2}}\left|0\right>+e^{i\phi}\sin{\frac{\theta}{2}}
#    \left|1\right>
#
# Basis Encoding
# -------------------
#
# In basis encoding, each classical bit of a value is mapped into a single
# qubit, defining the encoding feature map as:
#
# .. math::
#    \phi:i\rightarrow \left|i\right>
#
# This means that a value :math:`x` would need :math:`\lceil\log_2 x\rceil`
# qubits in order to be mapped.
#
# The value :math:`0` would therefore be mapped to the state
# :math:`\left|0\right>`.

from skqlearn.encoding import *
import matplotlib.pyplot as plt
from qiskit.quantum_info import Statevector
import numpy as np

data = 0
state = Statevector(BasisEncoding().encoding(data))
print(state.draw('text'))

###############################################################################
state.draw('bloch').show()

###############################################################################
# A value like :math:`5` would need of :math:`3` qubits for the mapping, each
# one representing each bit in the binary representation of the number,
# :math:`101`.
#
# .. note::
#    Quantum states defined by the state of each of its qubits, for
#    example :math:`\left|011\right>`, are usually represented in little-endian
#    with the first qubit in the left-most position. Qiskit, the SDK used for
#    the visualization of the states, represents the qubits in big-endian.
#    This means that if the state is supposed to be :math:`\left|011\right>`,
#    the qubit with state :math:`\left|0\right>` will be shown by Qiskit
#    in the right-most position.
#
data = 5
state = Statevector(BasisEncoding().encoding(data))
print(state.draw('text'))

###############################################################################
state.draw('bloch').show()

###############################################################################
# The encoding also permits encoding an entire dataset of binary strings
# :math:`\mathcal{D}=\{\boldsymbol{x}^1,...,\boldsymbol{x}^m\}` together as:
#
# .. math::
#    \left|\mathcal{D}\right> = \frac{1}{\sqrt{M}}\sum_{m=1}^{M}
#    \left|\boldsymbol{x}^m\right>.
#
data = np.array([0, 1, 2, 3])
state = Statevector(BasisEncoding().encoding(data))
print(state.draw('text'))

###############################################################################
state.draw('bloch').show()


###############################################################################
# Amplitude Encoding
# -----------------------
#
# In amplitude encoding, each component of the input vector
# :math:`\boldsymbol{x} \in \mathbb{R}^N` is mapped to an amplitude of the
# quantum state, defining the encoding feature map as:
#
# .. math::
#    \phi:\boldsymbol{x}\rightarrow\left|\psi_\boldsymbol{x}\right>=
#    \sum_{i=1}^{N}x_i\left|i-1\right>
#
# In order to represent a valid quantum state, the amount of amplitudes
# must be a power of 2, :math:`N=2^n`. If they are not,
# they can be padded with zeros at the end.
#
# For this encoding to generate valid quantum states, the input vectors must
# be normalized. If they are not, the method is responsible for normalizing
# them. This should be taken into account when planning on using this encoding.
# The forceful normalization is performed as some subroutines can work around
# the issue.

data = np.array([1.0])
state = Statevector(AmplitudeEncoding().encoding(data))
print(state.draw('text'))

###############################################################################
state.draw('bloch').show()

###############################################################################
# The input vectors can also be mapped to :math:`d` copies of the amplitude
# vectors, which can be specially useful when using the kernel defined by the
# encoding feature map.
#
# .. math::
#    \phi:\boldsymbol{x}\rightarrow\left|\psi_\boldsymbol{x}\right>
#    ^{\bigotimes d}
#
data = np.array([1])
state = Statevector(AmplitudeEncoding(degree=2).encoding(data))
print(state.draw('text'))

###############################################################################
state.draw('bloch').show()

###############################################################################
# A dataset can also be encoded by concatenating all the resulting amplitude
# vectors and normalizing.
#
# In this case, the first 2D vector is mapped to the first qubit (or last
# in Qiskit's representation) and so on.

data = np.array([[1/np.sqrt(2), 1/np.sqrt(2)], [1.0, 0.0]])
state = Statevector(AmplitudeEncoding().encoding(data))
print(state.draw('text'))

###############################################################################
state.draw('bloch').show()

###############################################################################
# Expanded Amplitude Encoding
# --------------------------------
#
# This encoding method tries to solve the normalization problem in regular
# Amplitude Encoding. If non-normalized data is normalized for use on
# Amplitude Encoding, the data will lose one dimension of information. For
# example, if a 2D point is normalized, it will be mapped into the unit
# circle, a 1D shape.

points = np.array([[1, 1],
                   [2, 2],
                   [0.5, 3]])
normalized_points = points / np.linalg.norm(points, axis=1)[:, None]
lines = np.array([[[p[0], n[0]], [p[1], n[1]]]
         for p, n in zip(points, normalized_points)])

# Plot unit circle
x = np.linspace(0, np.pi / 2, 30)
plt.plot(np.cos(x), np.sin(x))
# Plot encodings
for i in range(lines.shape[0]):
    plt.plot(lines[i, 0, :], lines[i, 1, :], '--')
plt.scatter(points[:, 0], points[:, 1], marker='o')
plt.scatter(normalized_points[:, 0], normalized_points[:, 1], marker='x')
plt.xlim([0, 4])
plt.ylim([0, 4])
plt.show()

###############################################################################
# By adding an extra component to
# :math:`\boldsymbol{x}\in\mathbb{R}^N` with a value of :math:`c`,
# :math:`x_{0}=1`, and then normalizing, the information loss is mitigated.
#
# .. math::
#        \phi:\boldsymbol{x}\rightarrow\left|\psi_\boldsymbol{x}\right>=
#        \frac{1}{|\boldsymbol{x}|^2+c^2}\left(c\left|0\right> +
#        \sum_{i=1}^{N}x_i\left|i\right>\right)

c = 1
points = np.array([[1, 1, c],
                   [2, 2, c],
                   [0.5, 3, c]])
normalized_points = points / np.linalg.norm(points, axis=1)[:, None]
lines = np.array([[[p[0], n[0]], [p[1], n[1]]]
         for p, n in zip(points, normalized_points)])

# Plot limits
plt.plot([0, 1], [1, 1], 'b')
plt.plot([1, 1], [0, 1], 'b')
# Plot encodings
for i in range(lines.shape[0]):
    plt.plot(lines[i, 0, :], lines[i, 1, :], '--')
plt.scatter(points[:, 0], points[:, 1], marker='o')
plt.scatter(normalized_points[:, 0], normalized_points[:, 1], marker='x')
plt.xlim([0, 4])
plt.ylim([0, 4])
plt.show()

###############################################################################
# As shown in the figure, now the mapping is injective because of the
# mitigation of information loss. The encoding itself works exactly the same
# as Amplitude Encoding.

data = np.array([1.0, 1.0, 1.0])
state = Statevector(ExpandedAmplitudeEncoding(c=1.0).encoding(data))
print(state.draw('text'))

###############################################################################
state.draw('bloch').show()

###############################################################################
# Angle Encoding
# ----------------
#
# In angle encoding, each component of the input vector
# :math:`\boldsymbol{x} \in \mathbb{R}^N` is mapped to a qubit, defining the
# encoding feature map as:
#
# .. math::
#    \phi:\boldsymbol{x}\rightarrow\left|\psi_\boldsymbol{x}\right>=
#    \bigotimes_{i=1}^{N}\cos{x_i}\left|0\right>+\sin{x_i}\left|1\right>
#
# Because of the encoding feature map, the resulting quantum state is
# correctly normalized and therefore valid, as :math:`\cos{x}^2+\sin{x}^2=1`.

data = np.array([0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4])
state = Statevector(AngleEncoding().encoding(data))
print(state.draw('text'))

###############################################################################
# This encoding method is visualized specially well with the Bloch sphere
# because it follows an almost identical definition of how the quantum states
# are represented in the sphere.

state.draw('bloch').show()

###############################################################################
# QSample Encoding
# -----------------
#
# In QSample encoding, a discrete probability distribution is mapped into the
# amplitude vector of a quantum state, defining the encoding feature map as:
#
# .. math::
#     \phi:p(x)\rightarrow \left|p(x)\right>=\sum_{X} \sqrt{p(x_i)}
#     \left|x_i\right>
#
# Because the amplitudes are defined as :math:`\alpha_i = \sqrt{p(x_i)}`,
# the resulting quantum state is valid:
# :math:`\sum |\alpha_i|^2=\sum p(x_i) = 1`.
#
# This allows for the measurement of the quantum state to be interpreted as
# a sampling of the discrete probability distribution.

data = np.array([0.25, 0.5, 0.25])
state = Statevector(QSampleEncoding().encoding(data))
print(state.draw('text'))
