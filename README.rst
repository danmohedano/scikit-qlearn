scikit-qlearn: Quantum Enhanced Machine Learning
=================================================

Quantum-enhanced Machine Learning focuses on the improvement of classical machine learning algorithms with the help of
quantum subroutines.

This package offers a variety of functions and classes related to quantum computing and quantum enhanced machine learning algorithms.
From data encoding to clustering algorithms based on quantum subroutines. For further information on the features of the package, refer to the documentation.

The package makes use of the open-source `Qiskit SDK <https://qiskit.org/>`_ for the execution of the quantum subroutines, which gives access
to simulators and real quantum computers.

The package was orginally developed as a proof of concept as part of my Bachellor Thesis for my Computer Engineering degree at UAM.

Documentation
=============

The documentation is available at `danmohedano.github.io/scikit-qlearn/ <https://danmohedano.github.io/scikit-qlearn/>`_.
It includes detaild information of the classes and methods offered by the package, tutorials to guide the user and gathers
the changes made on every version.

Installation
=============
Currently, the package is available for Python versions 3.7-3.10, regardless of platform. Stable versions are available for install via `PyPI <https://pypi.org/project/scikit-qlearn/>`_:

.. code-block:: bash

   pip install scikit-qlearn

The latest version can also be manually installed by cloning the main branch of the repository:

.. code-block:: bash

   git clone https://github.com/danmohedano/scikit-qlearn.git
   pip install ./scikit-qlearn

Requirements
--------------
*scikit-qlearn* depends on the following packages:

* `qiskit <https://github.com/Qiskit>`_ - Open-source SDK for working with quantum computers at the level of pulses, circuits, and algorithms
* `numpy <https://github.com/numpy/numpy>`_ - The fundamental package for scientific computing with Python
