# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
from datetime import datetime
from aqua_blue import __version__, __authors__

sys.path.insert(0, os.path.abspath(".."))

project = "aqua-blue"
copyright = f"{datetime.now().year}, Chicago Club Management Company"
author = ", ".join(__authors__)
release = __version__
version = __version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["sphinx.ext.todo", "sphinx.ext.viewcode", "sphinx.ext.autodoc", "sphinx_autodoc_typehints"]
templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
autodoc_member_order = "bysource"
typehints_fully_qualified = True


def typehints_formatter(x, y=None):

    try:
        class_name = x.__name__
    except AttributeError:
        class_name = str(x)

    return class_name. \
        replace("[typing.Any, numpy.dtype[+_ScalarType_co]]", ""). \
        replace("numpy.ndarray", "numpy.typing.NDArray"). \
        replace("ndarray", "numpy.typing.NDArray"). \
        replace("numpy.random._generator.Generator", "numpy.random.Generator")

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output


html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
