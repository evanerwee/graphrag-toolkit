# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Any

from graphrag_toolkit.lexical_graph.indexing.model import Topic
from graphrag_toolkit.lexical_graph.storage.graph import GraphStore
from graphrag_toolkit.lexical_graph.indexing.build.graph_builder import GraphBuilder

from llama_index.core.schema import BaseNode

logger = logging.getLogger(__name__)

class TopicGraphBuilder(GraphBuilder):
    """
    Handles the creation and management of topic nodes and their relationships within
    a graph database.

    This class is responsible for extracting topic-related information from nodes,
    validating the extracted data against a defined model, and inserting or updating
    nodes and relationships in the graph database. The primary purpose of this class
    is to ensure all topic nodes are accurately represented and linked to related
    chunks in the graph.

    :ivar some_attribute1: Placeholder description of the attribute if applicable.
    :type some_attribute1: type
    :ivar some_attribute2: Placeholder description of the attribute if applicable.
    :type some_attribute2: type
    """
    @classmethod
    def index_key(cls) -> str:
        """
        Represents a class-level method to retrieve the index key utilized for indexing.

        The `index_key` method is a class-level utility that provides a standardized
        key format for identifying specific instances or categories related to the
        class's functionality.

        :classmethod:

        :return: A string representing the class-level index key.
        :rtype: str
        """
        return 'topic'
    
    def build(self, node:BaseNode, graph_client: GraphStore, **kwargs:Any):
        """
        Constructs and executes a graph database query to insert topic nodes and their relationships
        to chunk nodes based on metadata present in the provided node. If a topic is found in the node's
        metadata, it validates and processes the topic data, builds Cypher queries, and executes them
        to insert topics and their relationships into the graph database.

        If no topic metadata is provided in the node, a warning is logged indicating the absence
        of the topic ID.

        :param node: The `BaseNode` instance which contains metadata information including topic data.
        :param graph_client: The instance of `GraphStore` used to interact with the graph database.
        :param kwargs: Additional arguments that may be used during topic node processing.
        :return: None
        """
        topic_metadata = node.metadata.get('topic', {})

        if topic_metadata:

            topic = Topic.model_validate(topic_metadata)
        
            logger.debug(f'Inserting topic [topic_id: {topic.topicId}]')

            statements = [
                '// insert topics',
                'UNWIND $params AS params'
            ]
            

            chunk_ids =  [ {'chunk_id': chunkId} for chunkId in topic.chunkIds]

            statements.extend([
                f'MERGE (topic:`__Topic__`{{{graph_client.node_id("topicId")}: params.topic_id}})',
                'ON CREATE SET topic.value=params.title',
                'ON MATCH SET topic.value=params.title',
                'WITH topic, params',
                'UNWIND params.chunk_ids as chunkIds',
                f'MERGE (chunk:`__Chunk__`{{{graph_client.node_id("chunkId")}: chunkIds.chunk_id}})',
                'MERGE (topic)-[:`__MENTIONED_IN__`]->(chunk)'
            ])

            properties = {
                'topic_id': topic.topicId,
                'title': topic.value,
                'chunk_ids': chunk_ids
            }

            query = '\n'.join(statements)

            graph_client.execute_query_with_retry(query, self._to_params(properties))

        else:
            logger.warning(f'topic_id missing from topic node [node_id: {node.node_id}]') 
