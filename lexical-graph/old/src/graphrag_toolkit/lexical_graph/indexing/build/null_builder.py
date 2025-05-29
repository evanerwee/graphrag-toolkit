# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import List, Any

from graphrag_toolkit.lexical_graph.indexing import NodeHandler

from llama_index.core.schema import BaseNode

logger = logging.getLogger(__name__)

class NullBuilder(NodeHandler):
    """Handles the acceptance of nodes without performing any transformations,
    primarily used as a pass-through handler.

    The class is designed to process and yield nodes without altering their state. This
    can be helpful in scenarios where nodes need to be logged or monitored without any
    modification. The class inherits from `NodeHandler`.

    Attributes:
        None
    """
    def accept(self, nodes: List[BaseNode], **kwargs: Any):
        """
        Processes a list of nodes and yields them after logging their IDs.

        :param nodes: A list of nodes to process.
        :type nodes: List[BaseNode]
        :param kwargs: Additional keyword arguments for future extensibility.
        :type kwargs: Any
        :return: A generator yielding nodes from the input list.
        :rtype: Generator[BaseNode, None, None]
        """
        for node in nodes:
            logger.debug(f'Accepted node [node_id: {node.node_id}]')         
            yield node