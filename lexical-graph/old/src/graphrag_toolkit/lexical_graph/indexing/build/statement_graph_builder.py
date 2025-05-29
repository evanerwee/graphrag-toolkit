# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Any

from graphrag_toolkit.lexical_graph.indexing.model import Statement
from graphrag_toolkit.lexical_graph.storage.graph import GraphStore
from graphrag_toolkit.lexical_graph.indexing.build.graph_builder import GraphBuilder

from llama_index.core.schema import BaseNode
from llama_index.core.schema import NodeRelationship

logger = logging.getLogger(__name__)

class StatementGraphBuilder(GraphBuilder):
    """Handles the construction of graph representations specifically for
    `Statement` entities.

    This class extends the `GraphBuilder` and provides functionality to build a graph representation
    from `Statement` nodes, integrating relationships and required metadata into the graph. It
    interacts with the underlying `GraphStore` client to execute queries for creating and linking
    `Statement` entities to other graph components (e.g., chunks, topics). It also manages previous
    statement relationships if they exist.

    Attributes:
        None
    """
    @classmethod
    def index_key(cls) -> str:
        """Obtains the index key associated with the class.

        This method serves as a utility to retrieve a static key defining the
        index or identifying property of the class. It is useful in scenarios
        that require referencing or categorizing data based on a specific key.

        Returns:
            str: A static string 'statement' used as the index key.
        """
        return 'statement'
    
    def build(self, node:BaseNode, graph_client: GraphStore, **kwargs:Any):
        """
        Builds and executes a Cypher query based on the metadata and relationships of
        the provided `node`, inserting or updating a statement node in the graph
        database. The method validates the statement metadata and constructs the query
        to create or update relationships and properties for the statement node.

        :param node: The `BaseNode` object containing metadata and relationships
                     necessary to construct the query.
        :type node: BaseNode
        :param graph_client: The `GraphStore` instance used to interact with the graph
                             database.
        :type graph_client: GraphStore
        :param kwargs: Additional keyword arguments that may be passed for future
                       extensions or functionalities.
        :type kwargs: Any
        :return: None
        """
        statement_metadata = node.metadata.get('statement', {})
        
        if statement_metadata:

            statement = Statement.model_validate(statement_metadata)

            logger.debug(f'Inserting statement [statement_id: {statement.statementId}]')

            prev_statement = None
            prev_info = node.relationships.get(NodeRelationship.PREVIOUS, None)
            if prev_info:
                prev_statement = Statement.model_validate(prev_info.metadata.get('statement', None))

            if statement:

                statements = [
                    '// insert statements',
                    'UNWIND $params AS params'
                ]

                statements.extend([
                    f'MERGE (statement:`__Statement__`{{{graph_client.node_id("statementId")}: params.statement_id}})',
                    'ON CREATE SET statement.value=params.value, statement.details=params.details',
                    'ON MATCH SET statement.value=params.value, statement.details=params.details' 
                ])

                properties = {
                    'statement_id': statement.statementId,
                    'value': statement.value,
                    'details': '\n'.join(s for s in statement.details)
                }

                if statement.chunkId:
                    statements.extend([
                        f'MERGE (chunk:`__Chunk__`{{{graph_client.node_id("chunkId")}: params.chunk_id}})',
                        'MERGE (statement)-[:`__MENTIONED_IN__`]->(chunk)'
                    ])
                    properties['chunk_id'] = statement.chunkId

                if statement.topicId:
                    statements.extend([
                        f'MERGE (topic:`__Topic__`{{{graph_client.node_id("topicId")}: params.topic_id}})',
                        'MERGE (statement)-[:`__BELONGS_TO__`]->(topic)'
                    ])
                    properties['topic_id'] = statement.topicId

                if prev_statement:
                    statements.extend([
                        f'MERGE (prev_statement:`__Statement__`{{{graph_client.node_id("statementId")}: params.prev_statement_id}})',
                        'MERGE (statement)-[:`__PREVIOUS__`]->(prev_statement)'
                    ])
                    properties['prev_statement_id'] = prev_statement.statementId
                
                query = '\n'.join(statements)

                graph_client.execute_query_with_retry(query, self._to_params(properties))

        else:
            logger.warning(f'statement_id missing from statement node [node_id: {node.node_id}]')