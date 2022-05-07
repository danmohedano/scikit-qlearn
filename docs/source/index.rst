.. Scikit-QLearn documentation master file, created by
   sphinx-quickstart on Thu Apr 28 17:09:40 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to scikit-qlearn's documentation!
=========================================
This package offers a variety of functions and classes related to quantum computing and quantum enhanced machine learning algorithms.
From data encoding to clustering algorithms based on quantum subroutines.

The package makes use of the open-source `Qiskit SDK <https://qiskit.org/>`_ for the execution of the quantum subroutines, which gives access
to simulators and real quantum computers.

The package was orginally developed as a proof of concept as part of my end-of-degree thesis for my Computer Engineering degree at UAM.

You can find more information about the package and development in the `project page <https://github.com/danmohedano/scikit-qlearn>`_ on Github.

Installation
-------------
Currently, the package is available for Python versions 3.7-3.10, regardless of platform. Stable versions are available for install via `PyPI <https://pypi.org/project/scikit-qlearn/>`_:

.. code-block:: bash

   pip install scikit-qlearn

The latest version can also be manually installed by cloning the main branch of the repository:

.. code-block:: bash

   git clone https://github.com/danmohedano/scikit-qlearn.git
   pip install ./scikit-qlearn

Where to start?
----------------

For a quick rundown of the main functionalities of the package, how to use them and a bit of the theoretical background,
a series of tutorials are available in the :doc:`Tutorials <generated/auto_tutorials/index>` section.
For more specific information about the implementation of the modules, take a look at the :doc:`API Reference <generated/modules>`.
To see what changes were made in the last version of the package, go to the :doc:`Changelog <changelog>`.


.. toctree::
   :caption: Usage
   :hidden:
   :titlesonly:

   generated/auto_tutorials/index

.. toctree::
   :caption: Further Information
   :hidden:
   :titlesonly:
   :maxdepth: 2

   generated/modules
   changelog
   references
