# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import List, Any
from graphrag_toolkit.lexical_graph.indexing import IdGenerator
from graphrag_toolkit.lexical_graph.indexing.build.build_filter import BuildFilter
from graphrag_toolkit.lexical_graph.indexing.build.node_builder import NodeBuilder
from graphrag_toolkit.lexical_graph.indexing.build.source_node_builder import SourceNodeBuilder
from graphrag_toolkit.lexical_graph.indexing.build.chunk_node_builder import ChunkNodeBuilder
from graphrag_toolkit.lexical_graph.indexing.build.topic_node_builder import TopicNodeBuilder
from graphrag_toolkit.lexical_graph.indexing.build.statement_node_builder import StatementNodeBuilder

from llama_index.core.schema import BaseNode
from llama_index.core.schema import NodeRelationship

logger = logging.getLogger(__name__)

class MetadataToNodes():
    """MetadataToNodes is a utility class that builds and processes nodes based
    on provided metadata and builders, while applying filters and
    transformations.

    This class facilitates the creation of nodes by applying a series of node builders
    and filters to the input nodes. It supports preprocessing steps such as text cleaning
    and tenant-specific transformations for node identifiers and relationships. The class
    is designed to process metadata and generate new nodes that adhere to user-defined
    node builder configurations.

    Attributes:
        builders (List[NodeBuilder]): A list of node builders used to transform metadata and
            build nodes.
        filter (BuildFilter): A filter applied to evaluate the input nodes before building nodes.
        id_generator (IdGenerator): An instance of IdGenerator used for generating and rewriting
            tenant-specific node IDs.
    """
    def __init__(self, builders:List[NodeBuilder]=[], filter:BuildFilter=None, id_generator:IdGenerator=None):

        id_generator = id_generator or IdGenerator()

        self.builders = builders or self.default_builders(id_generator)
        self.filter = filter or BuildFilter()
        self.id_generator = id_generator

        logger.debug(f'Node builders: {[type(b).__name__ for b in self.builders]}')

    def default_builders(self, id_generator:IdGenerator):
        """Builds and returns a list of default node builders using the
        provided IdGenerator.

        This method instantiates and provides a collection of node builders, where each
        builder is responsible for creating a specific type of node within the system.
        The `id_generator` is utilized to assign unique identifiers to the nodes created
        by these builders.

        Args:
            id_generator (IdGenerator): Instance of IdGenerator used to generate unique
                identifiers for nodes.

        Returns:
            list: A list of node builder instances including SourceNodeBuilder,
                ChunkNodeBuilder, TopicNodeBuilder, and StatementNodeBuilder. Each
                builder is initialized with the provided `id_generator`.
        """
        return [
            SourceNodeBuilder(id_generator=id_generator),
            ChunkNodeBuilder(id_generator=id_generator),
            TopicNodeBuilder(id_generator=id_generator),
            StatementNodeBuilder(id_generator=id_generator)
        ]

    @classmethod
    def class_name(cls) -> str:
        """Provides class-level name identifier for the 'MetadataToNodes'
        class.

        This method returns a string value that represents the name of the
        class. It is a utility useful for identifying instances or constructing
        dynamic processes that involve metadata or hierarchical node mappings.

        Returns:
            str: The name of the class, 'MetadataToNodes'.
        """
        return 'MetadataToNodes'

    def get_nodes_from_metadata(self, input_nodes: List[BaseNode], **kwargs: Any) -> List[BaseNode]:
        """Processes input nodes by applying tenant id rewrites and cleaning
        text, then uses the builders to generate new nodes based on metadata
        and appends the original nodes to the results.

        Args:
            input_nodes (List[BaseNode]): A list of input nodes to be processed and used for generating new nodes.
            **kwargs (Any): Additional keyword arguments that may be required by the builders.

        Returns:
            List[BaseNode]: A list of processed nodes that includes both generated nodes and the original input nodes.

        Raises:
            Exception: If an error occurs during the node-building process by any builder.
        """
        def apply_tenant_rewrites(node):
            """Represents a utility to fetch nodes from metadata and apply
            tenant-specific rewrites using an ID generator.

            Methods:
                get_nodes_from_metadata: Process a list of input nodes, apply ID rewrites
                    to each node and its relationships for a specific tenant, and return
                    the modified list of nodes.

            """
            node.id_ =  self.id_generator.rewrite_id_for_tenant(node.id_)

            node_relationships = {}

            for rel, node_info in node.relationships.items():
                if isinstance(node_info, list):
                    node_info_list = []
                    for n in node_info:
                        n.node_id = self.id_generator.rewrite_id_for_tenant(n.node_id) 
                        node_info_list.append(n)
                    node_relationships[rel] = node_info_list
                else:
                    node_info.node_id = self.id_generator.rewrite_id_for_tenant(node_info.node_id)
                    node_relationships[rel] = node_info

            return node

        def clean_text(node):
            """Cleans the text content of a node by removing null characters.

            Args:
                node: The node whose text needs to be cleaned.

            Returns:
                The node with cleaned text content.
            """
            node.text = node.text.replace('\x00', '')
            return node

        def pre_process(node):
            """Preprocesses a node by cleaning its text and applying tenant-specific rewrites.

            Args:
                node: The node to be preprocessed.

            Returns:
                The preprocessed node with cleaned text and tenant-specific ID rewrites.
            """
            node = clean_text(node)
            node = apply_tenant_rewrites(node)
            return node

        results = []

        for builder in self.builders:
            try:

                filtered_input_nodes = [
                    pre_process(node) 
                    for node in input_nodes 
                    if any(key in builder.metadata_keys() for key in node.metadata)
                ]

                results.extend(builder.build_nodes(filtered_input_nodes, self.filter))
            except Exception as e:
                    logger.exception('An error occurred while building nodes from metadata')
                    raise e

        results.extend(input_nodes) # Always add the original nodes after derived nodes    

        logger.debug(f'Accepted {len(input_nodes)} chunks, emitting {len(results)} nodes')

        return results

    def __call__(self, nodes: List[BaseNode], **kwargs: Any) -> List[BaseNode]:    
        return self.get_nodes_from_metadata(nodes, **kwargs)
