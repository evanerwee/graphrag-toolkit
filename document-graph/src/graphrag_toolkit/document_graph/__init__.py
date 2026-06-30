"""Document Graph — structured data → typed graph nodes.

A transformation library that extends graphrag-toolkit with typed node support
for structured formats (CSV, Excel, JSON, Parquet).

Architecture:
    Toolkit Reader → [ingest → transform → build] → Toolkit GraphStore

Usage:
    from graphrag_toolkit.document_graph import NodeModel, EdgeModel
    from graphrag_toolkit.document_graph.transform.graph_transformers.row_to_node import RowToNode
    from graphrag_toolkit.document_graph.graph_build.cypher_builder import CypherBuilder
"""

__version__ = "3.0.3"

from .model import NodeModel, EdgeModel
from .model_elements import Node, Edge
from .pipeline_executor import PipelineExecutor
from .errors import (
    ModelError,
    DatabaseConnectionError,
    ConfigurationError,
    QueryExecutionError,
)

__all__ = [
    "PipelineExecutor",
    "NodeModel", "EdgeModel",
    "Node", "Edge",
    "ModelError", "DatabaseConnectionError",
    "ConfigurationError", "QueryExecutionError",
]
