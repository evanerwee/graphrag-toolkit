# Code Property Graph Examples

Example notebooks for `codeproperty-graph` — delta-aware code analysis ingestion into Neptune.

## Prerequisites

- Amazon Neptune (DB or Analytics) for production notebooks
- Python 3.10+
- Joern (for generating real CPG exports; sample data provided for demos)

## Setup

```bash
pip install codeproperty-graph
```

For Neptune integration:
```bash
pip install codeproperty-graph graphrag-toolkit-document-graph[graphrag,neptune]
```

## Notebooks

| Notebook | Description | Neptune Required? |
|----------|-------------|-------------------|
| `01-CPG-Models-and-GraphDiff.ipynb` | Core models, method signatures, and diff comparison | No (local only) |
| `02-Delta-Ingestion-Pipeline.ipynb` | Full pipeline: manifest → diff → skip/ingest decision | Optional (has mock mode) |

## Sample Data

`data/sample_nodes.json` — 8 CPG nodes (3 METHOD, 2 CALL, 2 IDENTIFIER, 1 LITERAL)
`data/sample_edges.json` — 10 edges (AST, CFG, CALL, ARGUMENT)

Represents a minimal Java project with `Main.main()`, `Parser.parse()`, and `Validator.validate()`.

## Generating Real CPG Data

```bash
# Install Joern: https://joern.io
joern-export --repr cpg14 --format json --out cpg-export/ /path/to/source

# Output:
#   cpg-export/nodes.json
#   cpg-export/edges.json
```
