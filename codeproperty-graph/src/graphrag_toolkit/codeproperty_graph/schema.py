# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPG Schema — Complete Joern Code Property Graph node and edge type definitions.

This module defines the canonical node types, edge types, and their properties
as specified by the Joern CPG schema (https://docs.joern.io/).

codeproperty-graph ingests ALL Joern types — the schema is not restrictive.
These enums and property specs serve as:
1. Documentation — what types exist and what they mean
2. Validation — optional strict mode to reject unknown types
3. Query helpers — typed constants for building Cypher queries

Supported Joern frontends (languages):
- Java (javasrc, jimple)
- JavaScript/TypeScript (jssrc)
- Python (pysrc)
- C/C++ (c2cpg)
- Go, PHP, Ruby, Kotlin, Swift (community frontends)
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set


class NodeType(str, Enum):
    """All CPG node types as defined by Joern's schema.

    Reference: https://docs.joern.io/cpgql/node-type-steps/
    """

    # === Structure ===
    FILE = "FILE"                       # Source file
    NAMESPACE = "NAMESPACE"             # Package/module namespace
    NAMESPACE_BLOCK = "NAMESPACE_BLOCK" # Namespace scope block
    TYPE_DECL = "TYPE_DECL"             # Class/struct/interface declaration
    TYPE = "TYPE"                       # Type reference
    MEMBER = "MEMBER"                   # Class/struct member (field)
    META_DATA = "META_DATA"             # Graph metadata (language, version)

    # === Methods ===
    METHOD = "METHOD"                   # Function/method definition
    METHOD_REF = "METHOD_REF"           # Reference to a method (lambda, callback)
    METHOD_RETURN = "METHOD_RETURN"     # Formal return parameter
    PARAMETER = "PARAMETER"             # Formal parameter
    MODIFIER = "MODIFIER"              # Access modifier (public, private, static)

    # === Statements & Expressions ===
    BLOCK = "BLOCK"                     # Code block (scope)
    CALL = "CALL"                       # Call site (method/function invocation)
    IDENTIFIER = "IDENTIFIER"           # Variable/member reference
    LITERAL = "LITERAL"                 # Constant value (number, string)
    LOCAL = "LOCAL"                     # Local variable declaration
    RETURN = "RETURN"                   # Return statement

    # === Metadata ===
    COMMENT = "COMMENT"                 # Source code comment
    TAG = "TAG"                         # User-defined tag/annotation


class EdgeType(str, Enum):
    """All CPG edge types as defined by Joern's schema.

    Joern produces multiple graph layers overlaid on the same nodes:
    - AST:  Abstract Syntax Tree (parent-child structure)
    - CFG:  Control Flow Graph (execution order)
    - CDG:  Control Dependence Graph (branching influences)
    - PDG:  Program Dependence Graph (CDG + data deps)
    - DDG:  Data Dependence Graph (REACHING_DEF)
    - CALL: Call graph (caller → callee)
    """

    # === AST Layer ===
    AST = "AST"                         # Parent → child in syntax tree
    ARGUMENT = "ARGUMENT"               # Call → argument expression
    CONTAINS = "CONTAINS"               # File/Type/Method → contained elements
    BINDS_TO = "BINDS_TO"               # Type parameter binding

    # === Control Flow Layer ===
    CFG = "CFG"                         # Control flow edge (execution order)
    CONDITION = "CONDITION"             # Conditional branch

    # === Data Flow Layer ===
    REACHING_DEF = "REACHING_DEF"       # Data dependency (definition reaches use)
    REF = "REF"                         # Reference edge (identifier → declaration)

    # === Dependence Layers ===
    CDG = "CDG"                         # Control dependence
    DOMINATE = "DOMINATE"               # Dominance relation
    POST_DOMINATE = "POST_DOMINATE"     # Post-dominance relation

    # === Call Graph ===
    CALL = "CALL"                       # Caller → callee method

    # === Type System ===
    INHERITS_FROM = "INHERITS_FROM"     # Type inheritance/implementation
    ALIAS_OF = "ALIAS_OF"              # Type alias

    # === Annotations ===
    TAGGED_BY = "TAGGED_BY"             # Node → Tag association


@dataclass
class NodePropertySpec:
    """Expected properties for each node type per Joern's schema."""

    common: List[str] = field(default_factory=lambda: [
        "id", "label", "order"
    ])

    by_type: Dict[str, List[str]] = field(default_factory=lambda: {
        "METHOD": ["fullName", "name", "signature", "lineNumber", "lineNumberEnd",
                   "isExternal", "code", "filename", "hash"],
        "CALL": ["name", "code", "lineNumber", "methodFullName", "signature",
                 "dispatchType", "argumentIndex"],
        "IDENTIFIER": ["name", "code", "lineNumber", "order", "argumentIndex"],
        "LITERAL": ["code", "lineNumber", "order", "argumentIndex"],
        "LOCAL": ["name", "code", "lineNumber", "typeFullName"],
        "PARAMETER": ["name", "code", "lineNumber", "order", "typeFullName"],
        "TYPE_DECL": ["fullName", "name", "isExternal", "order"],
        "TYPE": ["fullName", "name"],
        "MEMBER": ["name", "code", "order", "typeFullName"],
        "FILE": ["name", "order"],
        "NAMESPACE": ["name", "order"],
        "NAMESPACE_BLOCK": ["fullName", "name", "order"],
        "BLOCK": ["lineNumber", "order", "argumentIndex"],
        "METHOD_RETURN": ["code", "lineNumber", "order", "typeFullName"],
        "METHOD_REF": ["code", "lineNumber", "order", "methodFullName"],
        "MODIFIER": ["modifierType", "order"],
        "RETURN": ["code", "lineNumber", "order", "argumentIndex"],
        "COMMENT": ["code", "lineNumber"],
        "TAG": ["name", "value"],
        "META_DATA": ["language", "version"],
    })


# Delta-relevant node types: only these participate in change detection.
# Rationale: method body changes are the meaningful code changes.
# Other changes (comments, formatting, metadata) don't affect behavior.
DELTA_RELEVANT_TYPES: Set[str] = {
    NodeType.METHOD,
}

# Structure-relevant types: used for understanding code organization
# (not for delta, but for graph queries about architecture).
STRUCTURE_TYPES: Set[str] = {
    NodeType.FILE,
    NodeType.NAMESPACE,
    NodeType.NAMESPACE_BLOCK,
    NodeType.TYPE_DECL,
    NodeType.TYPE,
    NodeType.MEMBER,
}

# Data-flow-relevant types: used for security analysis, taint tracking.
DATAFLOW_TYPES: Set[str] = {
    NodeType.CALL,
    NodeType.IDENTIFIER,
    NodeType.LITERAL,
    NodeType.PARAMETER,
    NodeType.LOCAL,
    NodeType.RETURN,
}

# Supported Joern frontends (languages)
SUPPORTED_LANGUAGES = {
    "java": "Java (javasrc frontend)",
    "javascript": "JavaScript/TypeScript (jssrc frontend)",
    "python": "Python (pysrc frontend)",
    "c": "C/C++ (c2cpg frontend)",
    "go": "Go (gosrc frontend)",
    "php": "PHP (php2cpg frontend)",
    "ruby": "Ruby (rubysrc frontend)",
    "kotlin": "Kotlin (kotlin2cpg frontend)",
    "swift": "Swift (swiftsrc frontend)",
}


def joern_export_command(source_path: str, output_dir: str = "cpg-export",
                         language: str = None) -> str:
    """Generate the Joern CLI command to export a CPG.

    Args:
        source_path: Path to the source code directory.
        output_dir: Directory for the exported JSON files.
        language: Optional language hint (auto-detected if omitted).

    Returns:
        Shell command string to run Joern export.
    """
    cmd = f"joern-export --repr cpg14 --format json --out {output_dir}"
    if language:
        cmd += f" --language {language}"
    cmd += f" {source_path}"
    return cmd
