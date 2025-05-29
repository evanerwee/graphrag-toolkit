# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import List, Any, Optional, Callable, Dict
from graphrag_toolkit.lexical_graph.metadata import SourceMetadataFormatter, DefaultSourceMetadataFormatter
from graphrag_toolkit.lexical_graph.indexing import IdGenerator
from graphrag_toolkit.lexical_graph.indexing.build.build_filters import BuildFilters
from graphrag_toolkit.lexical_graph.indexing.build.node_builder import NodeBuilder
from graphrag_toolkit.lexical_graph.indexing.build.source_node_builder import SourceNodeBuilder
from graphrag_toolkit.lexical_graph.indexing.build.chunk_node_builder import ChunkNodeBuilder
from graphrag_toolkit.lexical_graph.indexing.build.topic_node_builder import TopicNodeBuilder
from graphrag_toolkit.lexical_graph.indexing.build.statement_node_builder import StatementNodeBuilder

from llama_index.core.schema import BaseNode, NodeRelationship

logger = logging.getLogger(__name__)

class NodeBuilders():
    """
    Represents a class to manage node building logic with support for filtering,
    metadata formatting, and ID generation. This class provides mechanisms to
    process input nodes, apply filters, preprocess data, and construct new nodes
    from metadata using various registered builders.

    This class allows for extensibility through custom builders, filters, and
    metadata formatters. It processes input data into a structured form suitable
    for use in constructing nodes and manages workflows for transforming raw data
    into meaningful output for further processing.

    :ivar builders: List of NodeBuilder instances that encapsulate logic for building
        nodes of various types. Defaults to a set of predefined node builders.
    :type builders: list
    :ivar build_filters: Instance used for applying filtering logic to input nodes during
        preprocessing and node construction. Expected to conform to the `BuildFilters` protocol.
    :type build_filters: BuildFilters
    :ivar source_metadata_formatter: Component responsible for formatting metadata associated
        with nodes. Conforms to the source metadata formatting interface.
    :type source_metadata_formatter: SourceMetadataFormatter
    :ivar id_generator: Utility for generating unique identifiers for nodes and rewriting
        IDs when applying tenant-specific adjustments.
    :type id_generator: IdGenerator
    """
    def __init__(
            self, 
            builders:List[NodeBuilder]=[], 
            build_filters:BuildFilters=None, 
            source_metadata_formatter:Optional[SourceMetadataFormatter]=None,
            id_generator:IdGenerator=None
        ):
        """
        Initializes an instance of the class, setting up node builders, build filters,
        a source metadata formatter, and an ID generator. This initialization ensures
        that default components are provided if none are explicitly specified.

        :param builders: Optional list of NodeBuilder instances to construct nodes. If not
            provided, default builders will be used.
        :param build_filters: Optional BuildFilters instance to handle filtering
            logic during the build process. Defaults to a new instance of BuildFilters.
        :param source_metadata_formatter: Optional SourceMetadataFormatter instance to
            format metadata for nodes. Defaults to an instance of DefaultSourceMetadataFormatter.
        :param id_generator: Optional IdGenerator instance to generate unique IDs. If not
            provided, a new instance of IdGenerator will be used.
        """
        id_generator = id_generator or IdGenerator()
        build_filters = build_filters or BuildFilters()
        source_metadata_formatter = source_metadata_formatter or DefaultSourceMetadataFormatter()

        self.build_filters = build_filters
        self.id_generator = id_generator
        self.builders = builders or self.default_builders(id_generator, build_filters, source_metadata_formatter)

        logger.debug(f'Node builders: {[type(b).__name__ for b in self.builders]}')

    def default_builders(self, id_generator:IdGenerator, build_filters:BuildFilters, source_metadata_formatter:SourceMetadataFormatter):
        """
        Builds a list of node builders using the provided builders and their configurations.

        This function iterates over a predefined list of node builder classes and constructs
        them using the parameters provided. Each of these node builder classes represents a specific
        type of node (source, chunk, topic, or statement), and they are built with the common
        attributes: `id_generator`, `build_filters`, and `source_metadata_formatter`.

        :param id_generator: Instance used to generate unique identifiers for nodes.
            Must implement the `IdGenerator` protocol.
        :param build_filters: Instance responsible for filtering nodes during the
            build process. Expected to conform to the `BuildFilters` protocol.
        :param source_metadata_formatter: Formatter used to process source metadata.
            Expects an object with the `SourceMetadataFormatter` interface.
        :return: A list containing initialized instances of node builders, one for
            each builder class in the node builder class list.
        :rtype: list
        """
        return [
            node_builder(
                id_generator=id_generator, 
                build_filters=build_filters, 
                source_metadata_formatter=source_metadata_formatter
            )
            for node_builder in [SourceNodeBuilder, ChunkNodeBuilder, TopicNodeBuilder, StatementNodeBuilder]
        ]

    @classmethod
    def class_name(cls) -> str:
        """
        Provides the name of the class. This is useful for debugging, logging,
        or dynamic referencing in certain programming patterns.

        :return: The name of the class as a string.
        :rtype: str
        """
        return 'NodeBuilders'

    def get_nodes_from_metadata(self, input_nodes: List[BaseNode], **kwargs: Any) -> List[BaseNode]:
        """
        Processes input nodes, applies preprocessing, filters, tenant-specific rewrites,
        and builds new nodes from metadata using registered builders.

        This function represents a structured workflow for handling a collection of nodes.
        It applies filtering, text cleaning, ID rewriting, and transformation to generate
        a final list of nodes that includes both derived and original input nodes.

        :param input_nodes: List of BaseNode objects to be processed. Each node may have
            relationships, metadata, and text fields that are manipulated during preprocessing.
        :param kwargs: Additional keyword arguments to customize the node processing workflow.
        :return: List of BaseNode objects. Includes the original input nodes along with
            newly constructed nodes generated by the registered node builders.
        :rtype: List[BaseNode]

        :raises Exception: If an error occurs in the registered builders during the node
            building process, it is logged and then re-raised.

        """
        def apply_tenant_rewrites(node):
            """
            Applies tenant-specific rewrites to node IDs and their relationships.

            :param node: A BaseNode object to be processed.
            :return: The BaseNode object after applying tenant-specific rewrites.
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
            """
            Cleans the text content of a node by removing null characters.

            :param node: A BaseNode object whose text needs to be cleaned.
            :return: The BaseNode object with cleaned text.
            """
            node.text = node.text.replace('\x00', '')
            return node

        def pre_process(node):
            """
            Preprocesses a node by cleaning its text and applying tenant-specific rewrites.

            :param node: A BaseNode object to be preprocessed.
            :return: The preprocessed BaseNode object.
            """
            node = clean_text(node)
            node = apply_tenant_rewrites(node)
            return node

        results = []

        filtered_nodes = [
            node 
            for node in input_nodes 
            if self.build_filters.filter_source_metadata_dictionary(node.relationships[NodeRelationship.SOURCE].metadata) 
        ]

        pre_processed_nodes = [
            pre_process(node) 
            for node in filtered_nodes
        ]

        for builder in self.builders:
            try:

                builder_specific_nodes = [
                    node
                    for node in pre_processed_nodes 
                    if any(key in builder.metadata_keys() for key in node.metadata)
                ]

                results.extend(builder.build_nodes(builder_specific_nodes))
            except Exception as e:
                    logger.exception('An error occurred while building nodes from chunks')
                    raise e

        results.extend(input_nodes) # Always add the original nodes after derived nodes    

        logger.debug(f'Accepted {len(input_nodes)} chunks, emitting {len(results)} nodes')

        return results

    def __call__(self, nodes: List[BaseNode], **kwargs: Any) -> List[BaseNode]:
        """
        Executes the logic for processing a list of nodes to retrieve specific nodes
        based on metadata. The method uses provided nodes and any additional keyword
        arguments to filter or obtain nodes according to some predefined criteria.

        :param nodes: A list of BaseNode instances to be processed for metadata filtering.
        :param kwargs: Additional filters or parameters that refine the metadata search.
        :return: A list of BaseNode instances that match the metadata criteria.
        :rtype: List[BaseNode]
        """
        return self.get_nodes_from_metadata(nodes, **kwargs)
