# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

sys.path.insert(0, os.path.abspath("../../../python/qant_native_computing_toolkit"))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Q.ANT native computing toolkit"
copyright = "2026, Q.ANT GmbH"
author = "Q.ANT GmbH"
release = "2.3.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# napoleon: google-style autodocs
extensions = ["sphinx.ext.autodoc", "sphinx.ext.napoleon"]

templates_path = ["_templates"]
exclude_patterns: list[str] = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# read-the-docs style
html_theme = "sphinx_rtd_theme"

# init for classes should have a separate doc entry
autoclass_content = "init"

# shared logo, see doc/assets/logo
html_logo = "../../assets/logo/QANT_Logo_600_px.png"
