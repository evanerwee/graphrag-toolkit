# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import List, Any, Sequence

from graphrag_toolkit.lexical_graph.indexing.build.checkpoint import DoNotCheckpoint

from llama_index.core.node_parser import NodeParser
from llama_index.core.schema import BaseNode, Document
from llama_index.core.node_parser.node_utils import build_nodes_from_splits

logger = logging.getLogger(__name__)

class DocsToNodes(NodeParser, DoNotCheckpoint):
    """Parses documents into nodes.

    This class is responsible for parsing a collection of documents or nodes into
    a corresponding list of nodes. It extends functionality from `NodeParser` and
    `DoNotCheckpoint` to ensure compatibility with inheritable features and avoid
    saving checkpoints during operations.

    Attributes:
        None
    """
    def _parse_nodes(
        self,
        nodes: Sequence[BaseNode],
        show_progress: bool = False,
        **kwargs: Any,
    ) -> List[BaseNode]:
        """
        Parses a sequence of nodes into a list of `BaseNode` objects. If a node is of type
        `Document`, it converts the node into `BaseNode` by splitting the text and
        reconstructing the node. For other node types, it retains the original node.

        Args:
            nodes (Sequence[BaseNode]): A sequence of nodes to be parsed.
            show_progress (bool): A flag to indicate whether to display progress
                during parsing.
            **kwargs (Any): Additional keyword arguments for any future extensibility.

        Returns:
            List[BaseNode]: A list of parsed `BaseNode` objects.
        """
        def to_node(node):
            """
            Parses a sequence of nodes and converts documents to nodes where applicable.

            This method processes a given sequence of nodes. If a node is of type Document,
            it converts the node into one or more BaseNode instances based on text splits.
            For all other node types, it retains the original node. The function also
            allows progress tracking if specified.

            Args:
                nodes (Sequence[BaseNode]): A sequence of nodes to be parsed and processed.
                show_progress (bool): Indicates whether to show progress during parsing.
                **kwargs (Any): Additional keyword arguments for customization.

            Returns:
                List[BaseNode]: A list of processed BaseNode instances formed from the
                input nodes.
            """
            if isinstance(node, Document):
                return build_nodes_from_splits([node.text], node)[0]
            else:
                return node
    
        return [to_node(n) for n in nodes]