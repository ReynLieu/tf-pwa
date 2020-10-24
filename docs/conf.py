"""Configuration file for the Sphinx documentation builder.

This file only contains a selection of the most common options. For a full
list see the documentation:
https://www.sphinx-doc.org/en/master/usage/configuration.html
"""

import os
import shutil
import subprocess

from tf_pwa.amp import PARTICLE_MODEL, get_config

# -- Project information -----------------------------------------------------
project = "TFPWA"
copyright = "2020, Yi Jiang"
author = "Yi Jiang"

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.graphviz",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]
exclude_patterns = [
    ".DS_Store",
    "Thumbs.db",
    "_build",
]
source_suffix = [
    ".rst",
]

# -- Options for HTML output -------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_title = "TFPWA"
viewcode_follow_imported_members = True

# -- Options for API ---------------------------------------------------------
add_module_names = False
autodoc_mock_imports = [
    "iminuit",
    "tensorflow",
]

# Cross-referencing configuration
default_role = "py:obj"
primary_domain = "py"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
}

# -- Generate API skeleton ----------------------------------------------------
shutil.rmtree("api", ignore_errors=True)
subprocess.call(
    " ".join(
        [
            "sphinx-apidoc",
            "../tf_pwa/",
            "-o api/",
            "--force",
            "--no-toc",
            "--templatedir _templates",
            "--separate",
        ]
    ),
    shell=True,
)


# -- Generate available resonance models --------------------------------------
def add_indent(s, number=2):
    ret = ""
    for i in s.split("\n"):
        ret += " " * number + i + "\n"
    return ret


def gen_particle_model():
    particle_model_doc = """
--------------------------
Available Resonances Model
--------------------------

"""
    for idx, (k, v) in enumerate(get_config(PARTICLE_MODEL).items(), 1):
        doc_i = v.__doc__
        if v.__doc__ is None and v.get_amp.__doc__ is None:
            continue
        if v.__doc__ is None:
            doc_i = v.get_amp.__doc__

        particle_model_doc += (
            f"\n{idx}. {k} (`~{v.__module__}.{v.__qualname__}`)\n\n"
        )
        idx += 1
        particle_model_doc += add_indent(doc_i) + "\n\n"

    with open(
        os.path.dirname(os.path.abspath(__file__)) + "/particle_model.rst", "w"
    ) as f:
        f.write(particle_model_doc)


gen_particle_model()
