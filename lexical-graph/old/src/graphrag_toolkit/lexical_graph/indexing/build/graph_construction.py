# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from tqdm import tqdm
from typing import Any, List, Union

from graphrag_toolkit.lexical_graph.indexing.build.graph_builder import GraphBuilder
from graphrag_toolkit.lexical_graph.indexing.node_handler import NodeHandler
from graphrag_toolkit.lexical_graph.indexing.build.graph_batch_client import GraphBatchClient
from graphrag_toolkit.lexical_graph.storage.graph import GraphStore
from graphrag_toolkit.lexical_graph.storage.graph_store_factory import GraphStoreFactory
from graphrag_toolkit.lexical_graph.storage.constants import INDEX_KEY 
from graphrag_toolkit.lexical_graph.indexing.build.source_graph_builder import SourceGraphBuilder
from graphrag_toolkit.lexical_graph.indexing.build.chunk_graph_builder import ChunkGraphBuilder
from graphrag_toolkit.lexical_graph.indexing.build.topic_graph_builder import TopicGraphBuilder
from graphrag_toolkit.lexical_graph.indexing.build.statement_graph_builder import StatementGraphBuilder
from graphrag_toolkit.lexical_graph.indexing.build.fact_graph_builder import FactGraphBuilder
from graphrag_toolkit.lexical_graph.indexing.build.entity_graph_builder import EntityGraphBuilder
from graphrag_toolkit.lexical_graph.indexing.build.entity_relation_graph_builder import EntityRelationGraphBuilder
from graphrag_toolkit.lexical_graph.indexing.build.graph_summary_builder import GraphSummaryBuilder

from llama_index.core.bridge.pydantic import Field
from llama_index.core.schema import BaseNode

logger = logging.getLogger(__name__)

def default_builders() -> List[GraphBuilder]:
    """
    Creates and returns a list of default graph builders used for graph construction. Each graph builder in the
    list is responsible for building a specific type of graph structure, such as source graphs, chunk graphs,
    entity graphs, and more. These graph builders facilitate a modular and systematic approach to building
    complex graphs by separating functionality into distinct components.

    :return: A list of instances of GraphBuilder and its derived types used for various graph construction
             processes.
    :rtype: List[GraphBuilder]
    """
    return [
        SourceGraphBuilder(),
        ChunkGraphBuilder(),
        TopicGraphBuilder(),
        StatementGraphBuilder(),
        EntityGraphBuilder(),
        EntityRelationGraphBuilder(),
        FactGraphBuilder(),
        GraphSummaryBuilder()
    ]

GraphInfoType = Union[str, GraphStore]

class GraphConstruction(NodeHandler):
    """
    Handles the construction of graphs using a given GraphStore instance and
    a set of configurable graph builders for processing nodes. This class
    provides functionality for accepting and processing a set of nodes to
    build a graph, utilizing batch operations and other configurations for
    efficient graph construction.

    :ivar graph_client: The GraphStore instance used for managing graph-related
        operations, such as storing and retrieving graph-related data.
    :type graph_client: GraphStore
    :ivar builders: A list of GraphBuilder instances used to process and
        transform nodes into graph components. Builders are responsible for
        applying specific logic to the nodes based on their configurations.
    :type builders: List[GraphBuilder]
    """
    @staticmethod
    def for_graph_store(graph_info:GraphInfoType=None, **kwargs):
        """
        Creates an instance of the GraphConstruction class for interacting with a graph
        store, either by accepting an existing GraphStore instance or by creating one
        using the GraphStoreFactory.

        This static method provides flexibility in initializing the GraphConstruction
        class. If a GraphStore instance is passed, it is directly utilized. Otherwise,
        the method constructs a GraphStore instance using provided graph information
        and additional parameters, and then initializes GraphConstruction with it.

        :param graph_info: Optional information about the graph that may either be
            used to identify the graph store to interact with or directly represent
            an existing graph store instance.
        :type graph_info: GraphInfoType, optional

        :param kwargs: Additional keyword arguments required for creating or interacting
            with a graph store, forwarded to the GraphStoreFactory or GraphConstruction
            initialization as needed.
        :type kwargs: dict

        :return: An initialized instance of the GraphConstruction class representing
            the graph store ready for operations.
        :rtype: GraphConstruction
        """
        if isinstance(graph_info, GraphStore):
            return GraphConstruction(graph_client=graph_info, **kwargs)
        else:
            return GraphConstruction(graph_client=GraphStoreFactory.for_graph_store(graph_info, **kwargs), **kwargs)

    graph_client: GraphStore 
    builders:List[GraphBuilder] = Field(
        description='Graph builders',
        default_factory=default_builders
    )

    def accept(self, nodes: List[BaseNode], **kwargs: Any):
        """
        Processes a list of nodes within a graph, utilizing various builders for construction,
        and applies batch operations according to the provided configurations. The method
        supports optional progress display and facilitates integration with metadata-based
        builder selection.

        :param nodes: A list of nodes to process, each represented by an instance of BaseNode. Nodes may optionally contain metadata with specific keys for targeted builder processing.
        :type nodes: List[BaseNode]

        :param kwargs: Additional keyword arguments for configuration. Includes: batch_writes_enabled (bool) - Determines whether batch write operations are enabled during the process; batch_write_size (int) - Specifies the size of batch operations if enabled; Any other custom arguments required for the specific builders.

        :return: A generator yielding nodes after processing. Nodes may have been
            modified during the graph construction process or as a result of batch
            operations.
        :rtype: Generator[BaseNode, None, None]
        """
        builders_dict = {}
        for b in self.builders:
            if b.index_key() not in builders_dict:
                builders_dict[b.index_key()] = []
            builders_dict[b.index_key()].append(b)

        batch_writes_enabled = kwargs.pop('batch_writes_enabled')
        batch_write_size = kwargs.pop('batch_write_size')

        logger.debug(f'Batch config: [batch_writes_enabled: {batch_writes_enabled}, batch_write_size: {batch_write_size}]')
        logger.debug(f'Graph construction kwargs: {kwargs}')

        with GraphBatchClient(self.graph_client, batch_writes_enabled=batch_writes_enabled, batch_write_size=batch_write_size) as batch_client:

            node_iterable = nodes if not self.show_progress else tqdm(nodes, desc=f'Building graph [batch_writes_enabled: {batch_writes_enabled}, batch_write_size: {batch_write_size}]')

            for node in node_iterable:

                node_id = node.node_id

                if [key for key in [INDEX_KEY] if key in node.metadata]:

                    try:

                        index = node.metadata[INDEX_KEY]['index']
                        builders = builders_dict.get(index, None)

                        if builders:
                            for builder in builders:
                                builder.build(node, batch_client, **kwargs)
                        else:
                            logger.debug(f'No builders for node [index: {index}]')

                    except Exception as e:
                        logger.exception('An error occurred while building the graph')
                        raise e

                else:
                    logger.debug(f'Ignoring node [node_id: {node_id}]')

                if batch_client.allow_yield(node):
                    yield node

            batch_nodes = batch_client.apply_batch_operations()
            for node in batch_nodes:
                yield node
