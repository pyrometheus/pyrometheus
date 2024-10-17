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

import pkg_resources


# -- Project information -----------------------------------------------------

project = "pyrometheus"
copyright = "2020, University of Illinois Board of Trustees"
author = (
    "Esteban Cisneros, Andreas Kloeckner, Henry Le Berre, "
    "Center for Exascale-Enabled Scramjet Design"
)
version = pkg_resources.get_distribution("pyrometheus").version
# The full version, including alpha/beta/rc tags.
release = version


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.intersphinx",
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.napoleon",
    "sphinx_copybutton",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "furo"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
intersphinx_mapping = {
        "python": ("https://docs.python.org/3/", None),
        "numpy": ("https://numpy.org/doc/stable/", None),
        "mirgecom": ("https://mirgecom.readthedocs.io/en/latest/", None),
        "pymbolic": ("https://documen.tician.de/pymbolic", None),
        "cantera": ("https://cantera.org/documentation/docs-3.0/sphinx/html/", None),
        }

autoclass_content = "class"

mathjax3_config = {
    "tex2jax": {
        "inlineMath": [["\\(", "\\)"]],
        "displayMath": [["\\[", "\\]"]],
    },
}

nitpicky = True
