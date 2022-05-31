"""

.. _kerneltutorial:

Kernels
===========================

This tutorial aims to explain and visualize the kernels obtained from the data
encoding methods implemented in the package.
"""
###############################################################################
# Introduction
# -----------------
#
# As described in the :ref:`encodingstutorial` tutorial, the encoding methods
# can be interpreted as feature maps of the form:
#
# .. math::
#     \phi : \mathcal{X} \rightarrow \mathcal{F}
#
# Where the input :math:`x` is mapped from the input space into the feature
# space.
#
# The inner product of two inputs mapped to feature space defines a kernel via:
#
# .. math::
#    k(x,x'):= \left<\phi(x), \phi(x')\right>_\mathcal{F}
#
# Where :math:`\left<.,.\right>_\mathcal{F}` is the inner product defined
# on :math:`\mathcal{F}`. :cite:`hilbert2019`
#
# In this case, as the inputs are being mapped into quantum states,
# the kernel defined is of the form:
#
# .. math::
#    k(x,x')=\left<\phi(x)|\phi(x')\right>.
#
# Because of this, the encoding methods can be used to define kernels with the
# inner product and use this with machine learning algorithms such as
# Support-Vector Machines (SVMs). On top of that, the inner product can also be
# estimated with quantum subroutines with a time complexity of
# :math:`O(\log N)`, providing an exponential speed-up over
# the classical calculation.
#
# Two methods have been defined in this package, `classic_kernel` and
# `quantum_kernel`, in order to use the kernels defined by the encoding methods
# with implementations of SVMs such as
# `sklearn's <https://scikit-learn.org/stable/modules/svm.html#svm>`_.
# These methods compute the Gram matrix of the input set of vectors.
#
# Throughout this tutorial, both the classical computation and quantum
# estimation of the kernels will be tested, so the quantum backend will
# be initialized with one of the simulators provided by Qiskit.

from skqlearn.utils import JobHandler
from skqlearn.encoding import *
import matplotlib.pyplot as plt
import numpy as np
from qiskit.providers.aer import AerSimulator
from sklearn.svm import SVC

JobHandler().configure(backend=AerSimulator(), shots=10000)

###############################################################################
# Basis Encoding
# -----------------
#
# Basis Encoding's feature map was:
#
# .. math::
#     \phi : i\rightarrow \left|i\right>
#
# Therefore, the kernel defined by the inner product is:
#
# .. math::
#     k(i, j) = \left<\phi(i)|\phi(j)\right> = \left<i|j\right> =
#     \delta_{ij}
#
# With :math:`\delta` being the Kronecker delta, defined as
# :math:`\delta_{ij}=[i=j]`.
#
# When computing the Gram matrix of a set of vectors, the expected
# result would be the identity matrix.

x = np.array([[1], [2], [3], [4]])
print('Gram matrix with classical computation:')
print(BasisEncoding().classic_kernel(x, x))
print('Gram matrix with quantum estimation:')
print(BasisEncoding().quantum_kernel(x, x))

###############################################################################
# Because the next encodings can be applied to generic points in 2D space,
# a couple of utility functions will be defined to help with the visualization
# of the results. Specifically, they will be used to visualize the decision
# boundaries of the SVM.

def make_meshgrid(x, y, h=.02, border=.25):
    x_min, x_max = x.min() - border, x.max() + border
    y_min, y_max = y.min() - border, y.max() + border
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out


def plot_comparison(title, clf_c, clf_q, X0, X1):
    xx, yy = make_meshgrid(X0, X1, 0.1)
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    fig.suptitle(title)
    plot_contours(ax1, clf_c, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
    plot_contours(ax2, clf_q, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
    ax1.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=60, edgecolors='k')
    ax2.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=60, edgecolors='k')
    ax1.set_title('Classic Computation')
    ax1.set(xlabel='X1', ylabel='X2')
    ax1.set_aspect('equal', 'box')
    ax2.set_title('Quantum Estimation')
    ax2.set(xlabel='X1', ylabel='X2')
    ax2.set_aspect('equal', 'box')
    plt.subplots_adjust(left=0.10, bottom=0.01, right=0.95, top=0.99, wspace=0.1)
    plt.show()

###############################################################################
# In order to use a simple example which is not linearly separable, the
# proposed data for the SVM to classify is the XOR problem described with
# bipolar inputs. This is chosen over its binary representation because
# Amplitude Encoding is unable to encode empty vectors (where all components
# equal to 0).

x = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]])
y = np.array([-1, 1, 1, -1])
X1, X2 = x[:, 0], x[:, 1]

plt.scatter(X1, X2, c=y, cmap=plt.cm.coolwarm, s=60, edgecolors='k')
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('XOR problem with bipolar values')
plt.show()

###############################################################################
# Amplitude Encoding
# ---------------------
#
# Amplitude Encoding's feature map was:
#
# .. math::
#    \phi:\boldsymbol{x}\rightarrow\left|\psi_\boldsymbol{x}\right>=
#    \sum_{i=1}^{N}\frac{1}{|\boldsymbol{x}|}x_i\left|i-1\right>
#
# Therefore, the kernel defined by the inner product is:
#
# .. math::
#    k(\boldsymbol{x}, \boldsymbol{x'}) = \left<\psi_{\boldsymbol{x}}|
#    \psi_{\boldsymbol{x'}}\right> = \frac{1}{|\boldsymbol{x}|
#    |\boldsymbol{x'}|}\boldsymbol{x}^T\boldsymbol{x'}
#
# By applying a correction of :math:`|\boldsymbol{x}||\boldsymbol{x'}|` this
# can be used to estimate the linear kernel.

clf_amp_c = SVC(kernel=AmplitudeEncoding(degree=1).classic_kernel)
clf_amp_c.fit(x, y)
clf_amp_q = SVC(kernel=AmplitudeEncoding(degree=1).quantum_kernel)
clf_amp_q.fit(x, y)

plot_comparison('Comparison of results for Amplitude Encoding (Degree=1)',
                clf_amp_c, clf_amp_q, X1, X2)

###############################################################################
# Because the problem is not linearly separable, the linear kernel is not
# capable of solving the problem correctly.
#
# If the vectors are instead mapped to :math:`d` copies of the amplitude
# vectors:
#
# .. math::
#    \phi:\boldsymbol{x}\rightarrow\left|\psi_\boldsymbol{x}\right>
#    ^{\bigotimes d}
#
# Then the kernel defined is:
#
# .. math::
#    k(\boldsymbol{x}, \boldsymbol{x'}) = \left<\psi_{\boldsymbol{x}}|
#    \psi_{\boldsymbol{x'}}\right> \bigotimes ... \bigotimes
#    \left<\psi_{\boldsymbol{x}}|\psi_{\boldsymbol{x'}}\right> =
#    \left(\frac{1}{|\boldsymbol{x}||\boldsymbol{x'}|}\boldsymbol{x}^T
#    \boldsymbol{x'}\right)^d
#
# By applying a correction of :math:`(|\boldsymbol{x}||\boldsymbol{x'}|)^d`
# this can be used to estimate a polynomial kernel.
#
# With a higher degree, the problem at hand becomes linearly separable, as the
# input vectors are being expanded into a higher dimension. This is one of the
# main features of SVMs and kernels, commonly refered to as the kernel trick.

clf_amp_c_2 = SVC(kernel=AmplitudeEncoding(degree=2).classic_kernel)
clf_amp_c_2.fit(x, y)
clf_amp_q_2 = SVC(kernel=AmplitudeEncoding(degree=2).quantum_kernel)
clf_amp_q_2.fit(x, y)

plot_comparison('Comparison of results for Amplitude Encoding (Degree=2)',
                clf_amp_c_2, clf_amp_q_2, X1, X2)

###############################################################################
# Expanded Amplitude Encoding
# ----------------------------
#
# Expanded Amplitude Encoding's feature map was identical to regular Amplitude
# Encoding's. The only difference being that the input vectors were expanded
# with an extra component with value *c*.
#
# .. math::
#        \phi:\boldsymbol{x}\rightarrow\left|\psi_\boldsymbol{x}\right>=
#        \frac{1}{|\boldsymbol{x}|^2+c^2}\left(c\left|0\right> +
#        \sum_{i=1}^{N}x_i\left|i\right>\right)
#
# This defines a more general polynomial kernel when mapping to :math:`d`
# copies of the amplitude vector and applying a correction of
# :math:`\sqrt{|\boldsymbol{x}|^2+c^2}\sqrt{|\boldsymbol{x'}|^2+c^2}`:
#
# .. math::
#    k(\boldsymbol{x}, \boldsymbol{x'}) = \left<\psi_{\boldsymbol{x}}|
#    \psi_{\boldsymbol{x'}}\right> = \left(\frac{1}{\sqrt{|\boldsymbol{x}|^2+
#    c^2}\sqrt{|\boldsymbol{x'}|^2+c^2}}\boldsymbol{x}^T\boldsymbol{x'}\right)
#    ^d
#

expamp = ExpandedAmplitudeEncoding(degree=2, c=1)
clf_expamp_c = SVC(kernel=expamp.classic_kernel)
clf_expamp_c.fit(x, y)
clf_expamp_q = SVC(kernel=expamp.quantum_kernel)
clf_expamp_q.fit(x, y)

plot_comparison('Comparison of results for Expanded Amplitude Encoding '
                '(Degree=2, c=1)',
                clf_expamp_c, clf_expamp_q, X1, X2)

###############################################################################
# It should be noted that increasing the value of :math:`c` can have noticeable
# impacts on the precision of the results. The bigger the value of :math:`c`,
# the smaller the values of the rest of the components in the vector once
# normalized. This in turn increases the imprecision of the quantum estimation
# subroutine.

expamp_50 = ExpandedAmplitudeEncoding(degree=2, c=50)
clf_expamp_c_c50 = SVC(kernel=expamp_50.classic_kernel)
clf_expamp_c_c50.fit(x, y)
clf_expamp_q_c50 = SVC(kernel=expamp_50.quantum_kernel)
clf_expamp_q_c50.fit(x, y)

plot_comparison('Comparison of results for Expanded Amplitude Encoding '
                '(Degree=2, c=50)',
                clf_expamp_c_c50, clf_expamp_q_c50, X1, X2)

###############################################################################
# Angle Encoding
# ----------------------------
#
# Angle Encoding's feature map was:
#
# .. math::
#    \phi:\boldsymbol{x}\rightarrow\left|\psi_\boldsymbol{x}\right>=
#    \bigotimes_{i=1}^{N}\cos{x_i}\left|0\right>+\sin{x_i}\left|1\right>
#
# The kernel defined by the inner product is a cosine kernel:
#
# .. math::
#    k(\boldsymbol{x}, \boldsymbol{x'}) = \left<\psi_{\boldsymbol{x}}|
#    \psi_{\boldsymbol{x'}}\right> = \prod_{i=1}^{N}\sin{x_i}\sin{x'_i} +
#    \cos{x_i}\cos{x'_i}=\prod_{i=1}^{N}\cos{(x_i-x'_i)}.

clf_ang_c = SVC(kernel=AngleEncoding().classic_kernel).fit(x, y)
clf_ang_q = SVC(kernel=AngleEncoding().quantum_kernel).fit(x, y)

plot_comparison('Angle Encoding', clf_ang_c, clf_ang_q, X1, X2)
