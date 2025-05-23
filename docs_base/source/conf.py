#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# GraphRAG-Toolkit documentation build configuration file, created by
# sphinx-quickstart and adapted for graphrag-toolkit.
#
# This file is execfile()d with the current directory set to its containing dir.

import datetime
import os
import sys
import logging

logging.basicConfig(level=logging.INFO)

# Add lexical-graph/src
sys.path.insert(
    0, os.path.abspath(os.path.join(__file__, "..", "lexical-graph", "src"))
)

# Add lexical-graph-contrib/falkordb/src
sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(__file__, "..", "lexical-graph-contrib", "falkordb", "src", "graphrag_toolkit", "lexical_graph", "storage", "graph")
    ),
)

# -- General configuration -----------------------------------------------------

# Sphinx extensions
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx_copybutton',
    'myst_parser',
]

# Templates path relative to the docs directory
templates_path = ['_templates']

# Source file suffix
source_suffix = {
    '.rst': 'restructuredtext',
    '.txt': 'markdown',
    '.md': 'markdown',
}

# Master toctree document
master_doc = 'index'

# Project information
project = 'graphrag-toolkit'
current_year = datetime.date.today().year
copyright = f'{current_year}, Amazon Web Services, Inc'

# Version info (update with your project version)
version = '3.6'
release = '3.6.1'

# Patterns to ignore
exclude_patterns = []

# Pygments style for syntax highlighting
pygments_style = 'default'
pygments_dark_style = 'monokai'

# HTML theme
html_theme = 'furo'

# -- Options for HTML output ---------------------------------------------------

# Theme options
html_theme_options = {
    'footer_icons': [
        {
            'name': 'GitHub',
            'url': 'https://github.com/awslabs/graphrag-toolkit',
            'html': """
                <svg stroke="currentColor" fill="currentColor" stroke-width="0" viewBox="0 0 16 16">
                    <path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8z"></path>
                </svg>
            """,
            'class': '',
        },
    ],
    'light_logo': 'logos/aws_light_theme_logo.svg',
    'dark_logo': 'logos/aws_dark_theme_logo.svg',
}

# Static files path
html_static_path = ['_static']

# Custom CSS and JS files
html_css_files = [
    'css/custom.css',
    'css/dark_light_mode.css',
]
html_js_files = [
    'js/custom.js',
]

# Sidebar configuration
html_show_sourcelink = False
html_sidebars = {
    '**': [
        'sidebar/close-icon.html',
        'sidebar/brand.html',
        'sidebar/search.html',
        'sidebar/scroll-start.html',
        'sidebar/feedback.html',
        'sidebar/navigation.html',
        'sidebar/scroll-end.html',
    ]
}

# Show Sphinx and copyright in footer
html_show_sphinx = True
html_show_copyright = True

# Output file base name for HTML help
htmlhelp_basename = 'GraphRAG-Toolkit-doc'

# -- Options for LaTeX output --------------------------------------------------

latex_elements = {}

# LaTeX documents
latex_documents = [
    (
        'index',
        'GraphRAG-Toolkit.tex',
        'GraphRAG-Toolkit Documentation',
        'Amazon Web Services, Inc.',
        'manual',
    ),
]

# -- Options for manual page output --------------------------------------------

man_pages = [
    (
        'index',
        'graphrag-toolkit',
        'GraphRAG-Toolkit Documentation',
        ['Amazon Web Services, Inc.'],
        1,
    )
]

# -- Options for Texinfo output ------------------------------------------------

texinfo_documents = [
    (
        'index',
        'GraphRAG-Toolkit',
        'GraphRAG-Toolkit Documentation',
        'Amazon Web Services, Inc.',
        'GraphRAG-Toolkit',
        'Graph database and RAG toolkit for AWS.',
        'Miscellaneous',
    ),
]

# Autodoc configuration
autoclass_content = 'both'

print("sys.path:")
for p in sys.path:
    print(" -", p)
