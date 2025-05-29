# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import List

from llama_index.core.schema import TextNode, BaseNode
from llama_index.core.schema import NodeRelationship

from graphrag_toolkit.lexical_graph.indexing.build.node_builder import NodeBuilder
from graphrag_toolkit.lexical_graph.indexing.constants import TOPICS_KEY
from graphrag_toolkit.lexical_graph.storage.constants import INDEX_KEY

class SourceNodeBuilder(NodeBuilder):
    """Handles the construction of source-related nodes with metadata and
    specific configurations.

    This class extends the NodeBuilder and is designed to process a list of nodes to derive and
    construct nodes that represent source information. The class makes use of metadata keys and
    sources-related relationships to prepare well-defined source nodes.

    Attributes:
        TOPICS_KEY (str): Key identifier for topics in metadata.
        INDEX_KEY (str): Key identifier for indexing information.
        source_metadata_formatter (MetadataFormatter): Formatter for the source metadata.
    """
    @classmethod
    def name(cls) -> str:
        """Provides the class method to return the name of the implementing
        builder.

        This method is a utility function typically used in identifying the specific
        implementation of a node builder in a larger system. It ensures that each
        builder can provide a unique identifier or name.

        Returns:
            str: The name of the builder implementation.
        """
        return 'SourceNodeBuilder'
    
    @classmethod
    def metadata_keys(cls) -> List[str]:
        """Returns a list of keys that represent metadata identifiers used
        within the context of this class. Metadata keys are utilized as
        standardized identifiers to retrieve or interact with specific metadata
        attributes governed by the class.

        Returns:
            List[str]: A list of strings where each string is a metadata key.
        """
        return [TOPICS_KEY]
    
    def build_nodes(self, nodes:List[BaseNode]):
        """Builds and returns a list of TextNode objects corresponding to
        source nodes derived from the input `nodes`. The constructed nodes
        contain processed metadata and mapping information required for
        indexing and logical relationships.

        Args:
            nodes (List[BaseNode]): A list of `BaseNode` objects from which the source
                nodes will be derived.

        Returns:
            List[TextNode]: A list of `TextNode` objects created based on the source
                relationships and metadata configurations found in the provided `nodes`.
        """
        source_nodes = {}

        for node in nodes:
            
            source_info = node.relationships.get(NodeRelationship.SOURCE, None)
            source_id = source_info.node_id
            
            if source_id not in source_nodes:
                
                metadata = {
                    'source': {
                        'sourceId': source_id
                    }    
                }
                
                if source_info.metadata:
                    metadata['source']['metadata'] = self.source_metadata_formatter.format(source_info.metadata)
                    
                metadata[INDEX_KEY] = {
                    'index': 'source',
                    'key': self._clean_id(source_id)
                }
                
                source_node = TextNode(
                    id_ = source_id,
                    metadata = metadata,
                    excluded_embed_metadata_keys = [INDEX_KEY],
                    excluded_llm_metadata_keys = [INDEX_KEY]
                )

                source_nodes[source_id] = source_node

        return list(source_nodes.values())


