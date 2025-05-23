#!/bin/bash

set -e

# Base project directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Paths to source code
SRC_MAIN="$PROJECT_ROOT/lexical-graph/src"
SRC_CONTRIB="$PROJECT_ROOT/lexical-graph-contrib/falkordb/src"

# Path to Sphinx docs
DOCS_DIR="$PROJECT_ROOT/docs_base"
DOCS_SOURCE="$DOCS_DIR/source"
DOCS_BUILD="$DOCS_DIR/build"

echo "🔧 Ensuring __init__.py files exist..."
find "$SRC_MAIN/graphrag_toolkit" -type d -exec touch {}/__init__.py \;
find "$SRC_CONTRIB/graphrag_toolkit" -type d -exec touch {}/__init__.py \;

echo "🧹 Cleaning old .rst files..."
find "$DOCS_SOURCE" -name "*.rst" ! -name "index.rst" -delete || true

echo "🧠 Generating autodoc .rst files..."
sphinx-apidoc -f -o "$DOCS_SOURCE" "$SRC_MAIN/graphrag_toolkit"
sphinx-apidoc -f -o "$DOCS_SOURCE/falkordb" "$SRC_CONTRIB/graphrag_toolkit/lexical_graph/storage/graph/falkordb"

echo "🌍 Building HTML docs..."
PYTHONPATH="$SRC_MAIN:$SRC_CONTRIB/graphrag_toolkit/lexical_graph/storage/graph" sphinx-build -b html -d "$DOCS_BUILD/doctrees" -j auto -v "$DOCS_SOURCE" "$DOCS_BUILD/html"

echo "✅ Docs built successfully at: $DOCS_BUILD/html"