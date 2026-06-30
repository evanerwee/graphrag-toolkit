# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Code Property Graph — domain layer for Joern CPG delta ingestion.

Built on document-graph for typed property graph primitives.
Adds CPG-specific models, delta comparison, manifest management,
and tenant lifecycle for incremental code analysis.

Supported Joern frontends: Java, JavaScript/TypeScript, Python, C/C++,
Go, PHP, Ruby, Kotlin, Swift.

Full CPG schema: 20 node types, 14+ edge types.
See schema.py for complete enumeration.
"""

__version__ = "0.2.0"

from .models import CPGNode, CPGEdge, Manifest
from .schema import NodeType, EdgeType, DELTA_RELEVANT_TYPES, SUPPORTED_LANGUAGES, joern_export_command
from .graph_diff import GraphDiff
from .manifest_manager import ManifestManager
from .delta_ingestor import DeltaIngestor
from .tenant_ops import delete_tenant

__all__ = [
    # Models
    "CPGNode", "CPGEdge", "Manifest",
    # Schema
    "NodeType", "EdgeType", "DELTA_RELEVANT_TYPES", "SUPPORTED_LANGUAGES",
    "joern_export_command",
    # Pipeline
    "GraphDiff", "ManifestManager", "DeltaIngestor",
    "delete_tenant",
]
