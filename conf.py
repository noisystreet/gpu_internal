# Configuration file for the Sphinx documentation builder.

project = "深入理解 GPU"
copyright = "2026, GPU Internal"
author = "GPU Internal"

release = "0.1.0"

extensions = [
    "sphinx.ext.mathjax",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosectionlabel",
]

autosectionlabel_maxdepth = 1

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

language = "zh_CN"

# -- Options for HTML output -------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "navigation_depth": 2,
}
html_static_path = ["_static"]

# -- Options for todo --------------------------------------------------------
todo_include_todos = True
