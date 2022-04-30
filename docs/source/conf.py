# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))
import sys
import pathlib
sys.path.insert(0, pathlib.Path(__file__).parents[2].resolve().as_posix())
import skqlearn


# -- Project information -----------------------------------------------------

project = 'scikit-qlearn'
copyright = '2022, Daniel Mohedano'
author = 'Daniel Mohedano'

# The full version, including alpha/beta/rc tags
version = skqlearn.__version__


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.mathjax',
    'sphinx.ext.doctest',
    'sphinxcontrib.bibtex',
]

autodoc_default_flags = ['members', 'inherited-members']
autoclass_content = 'both'
doctest_global_setup = '''
import numpy as np
from qiskit import *
from skqlearn.clustering import *
from skqlearn.encoding import *
from skqlearn.gates import *
from skqlearn.jobhandler import *
from skqlearn.utils import *
'''

bibtex_bibfiles = ['refs.bib']

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

