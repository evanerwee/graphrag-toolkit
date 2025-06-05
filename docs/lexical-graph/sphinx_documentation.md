##Topics

* What Sphinx is
* How your documentation is organized
* How to use the build script
* How to regenerate and view documentation

---

```markdown
# GraphRAG Toolkit Documentation

This project also introdice [Sphinx](https://www.sphinx-doc.org/en/master/) to build and maintain developer and API documentation for the **GraphRAG Toolkit**, including both the core framework and optional FalkorDB extensions.

---

## What is Sphinx?

Sphinx is a documentation generator for Python projects that turns your code’s docstrings and reStructuredText or Markdown files into static HTML documentation. It is widely used in the Python ecosystem (including for Python itself) and integrates easily with IDEs, CI pipelines, and GitHub Pages.

---

## 🗂Documentation Structure

```

lexical-graph/
│
├── src/                                # Core source code
│
lexical-graph-contrib/
│
├── falkordb/src/                       # Contrib module (FalkorDB adapter)
│
sphinx\_docs/
│
├── source/                             # Sphinx .rst/.md source files
│   └── index.rst                       # Root index file
│
├── build/                              # Output directory for HTML, doctrees
│
├── build\_docs.sh                       # 🚀 Script to build full docs
│
└── README.md                           # This file

````

---

## 🔧 How to Use the Build Script

The `build_docs.sh` script performs the following steps:

1. Ensures all Python packages have `__init__.py` files (required by Sphinx).
2. Removes any previously generated `.rst` files (excluding `index.rst`).
3. Generates new `.rst` API reference files using `sphinx-apidoc` for:
   - `lexical-graph/src/graphrag_toolkit`
   - `lexical-graph-contrib/falkordb/src/graphrag_toolkit/.../falkordb`
4. Builds the HTML documentation using `sphinx-build`.

### ✅ Run It Like This

```bash
cd <project-root>
./sphinx_docs/build_docs.sh
````

> The final HTML output will be located at:
> `sphinx_docs/build/html/index.html`

You can open it in a browser to explore the generated documentation.

---

## 🔄 When to Rebuild Docs

Re-run the script whenever you:

* Modify code docstrings
* Add new modules or packages
* Change the `index.rst` or any Markdown files in `source/`

---

## 🧪 Prerequisites

Make sure you have the following installed in your Python environment:

```bash
pip install sphinx sphinx-rtd-theme myst-parser
```

Optional but recommended:

```bash
pip install autodoc-typehints
```

---

## Tips

* You can also preview documentation locally with live reload using tools like [sphinx-autobuild](https://github.com/executablebooks/sphinx-autobuild).
* You can write Markdown using [MyST Parser](https://myst-parser.readthedocs.io/) (enabled in `conf.py`).
* Extend `index.rst` to include custom documentation pages or usage examples.

---

## Continuous Integration (Optional)

You can integrate this build process into a CI/CD pipeline to publish documentation automatically (e.g., GitHub Actions + GitHub Pages).

---

## Additional Resources

* [Sphinx Getting Started](https://www.sphinx-doc.org/en/master/usage/quickstart.html)
* [reStructuredText Guide](https://docutils.sourceforge.io/docs/user/rst/quickref.html)
* [MyST Markdown Guide](https://myst-parser.readthedocs.io/en/latest/syntax/syntax.html)

---

Happy documenting!

