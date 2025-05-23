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
    Represents a utility class for building or transforming nodes.

    The `NodeBuilders` class is designed to handle the creation, transformation, and
    filtering of nodes based on metadata. This is achieved through various configurations
    like node builders, filtering mechanisms, metadata formatting, and an ID generator.
    It facilitates the processing of custom metadata and ensures nodes are correctly structured.

    :ivar builders: A list of configured node builder instances, each specializing in
        creating nodes based on specific logic or rules.
    :vartype builders: List[NodeBuilder]

    :ivar build_filters: The filter instance defined to control the criteria for filtering
        nodes during their processing.
    :vartype build_filters: BuildFilters

    :ivar source_metadata_formatter: A formatter to process metadata into a specific structure
        during node handling.
    :vartype source_metadata_formatter: SourceMetadataFormatter

    :ivar id_generator: An instance responsible for generating unique IDs for the nodes,
        or rewriting node IDs to conform to specific conventions.
    :vartype id_generator: IdGenerator
    """
    def __init__(
            self, 
            builders:List[NodeBuilder]=[], 
            build_filters:BuildFilters=None, 
            source_metadata_formatter:Optional[SourceMetadataFormatter]=None,
            id_generator:IdGenerator=None
        ):
        """
        Initializes an instance of the class with configured node builders, build filters,
        a custom source metadata formatter, and an ID generator. These configurations dictate
        the behavior of the instance, including how nodes are built, filtered, formatted,
        and uniquely identified.

        :param builders: A list of NodeBuilder instances. Every builder is used to create
            nodes based on specific logic and rules. If not provided, default builders
            are used.
        :type builders: List[NodeBuilder]

        :param build_filters: An instance of BuildFilters to control the filtering of nodes
            during the building process. Defaults to a new BuildFilters instance if not provided.
        :type build_filters: BuildFilters

        :param source_metadata_formatter: A source metadata formatter to shape metadata
            of source nodes. If not provided, a DefaultSourceMetadataFormatter instance
            will be used.
        :type source_metadata_formatter: Optional[SourceMetadataFormatter]

        :param id_generator: An instance of IdGenerator to produce unique identifiers
            for nodes. Will default to a newly created IdGenerator if one is not supplied.
        :type id_generator: IdGenerator
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
        Builds a list of node builders using the provided generator, filters, and formatter.

        This method initializes a list of node builders by iterating over predefined node
        builder classes. Each builder is instantiated with the provided id generator, filter,
        and formatter, which are essential for creating structured and uniquely identified
        nodes in a processing pipeline. The configuration passed to the builders ensures
        consistency and compatibility across different node builders.

        :param id_generator: An instance of IdGenerator used to generate unique identifiers
            for nodes.
        :param build_filters: An instance of BuildFilters used to filter or conditionally
            process nodes during their building process.
        :param source_metadata_formatter: An instance of SourceMetadataFormatter used to
            format metadata for the built nodes.
        :return: A list of instantiated node builders, each configured with the provided
            id generator, filter, and formatter.
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
        Provides a class method to retrieve the name of the class. This method can be used
        to dynamically access the class name at runtime without directly referencing it.

        :returns: The name of the class as a string
        :rtype: str
        """
        return 'NodeBuilders'
    
    def get_nodes_from_metadata(self, input_nodes: List[BaseNode], **kwargs: Any) -> List[BaseNode]:
        """
        Processes a list of input nodes to apply tenant-specific identifier rewrites,
        clean textual data, filter, and prepare node structures based on provided
        metadata and provided builders. It returns a refined list of nodes combining
        both original and derived nodes.

        :param input_nodes: List of input nodes to be processed
        :type input_nodes: List[BaseNode]
        :param kwargs: Additional keyword arguments for optional configurations
        :type kwargs: Any
        :return: A list of nodes that have been filtered, rewritten, cleaned,
            pre-processed, and processed by builder logic
        :rtype: List[BaseNode]
        """
        def apply_tenant_rewrites(node):
            """
            Represents a utility class for building or transforming nodes.

            Attributes:
                id_generator: An instance used for rewriting node IDs according to a tenant's context.
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
            Represents a utility class for building and manipulating nodes.

            This class provides methods used to process and refine nodes, specifically
            deriving node data from provided metadata and cleaning text fields within
            the nodes.
            """
            node.text = node.text.replace('\x00', '')
            return node
        
        def pre_process(node):
            """
            Handles operations related to node processing and construction.

            This class is responsible for managing and building nodes from metadata by
            applying preprocessing operations as defined. It allows for customization
            using additional keyword arguments during node processing.
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
        This method is responsible for processing a list of nodes and returning a filtered or modified
        list based on specified metadata criteria. It uses the `get_nodes_from_metadata` method to
        apply the operations. Additional parameters can be passed to customize the processing behavior.

        :param nodes: A list of `BaseNode` objects that need to be processed.
        :type nodes: List[BaseNode]
        :param kwargs: Additional arguments that control the behavior of metadata filtering or processing.
                       These arguments are passed directly to the `get_nodes_from_metadata` method.
        :type kwargs: Any
        :return: A list of `BaseNode` objects that have been processed based on the given
                 metadata criteria and additional `kwargs`.
        :rtype: List[BaseNode]
        """
        return self.get_nodes_from_metadata(nodes, **kwargs)
                    
