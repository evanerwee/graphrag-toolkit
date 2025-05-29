# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Any

from graphrag_toolkit.lexical_graph.indexing.model import Fact
from graphrag_toolkit.lexical_graph.storage.graph import GraphStore
from graphrag_toolkit.lexical_graph.storage.graph.graph_utils import search_string_from, label_from, relationship_name_from
from graphrag_toolkit.lexical_graph.indexing.build.graph_builder import GraphBuilder
from graphrag_toolkit.lexical_graph.indexing.constants import DEFAULT_CLASSIFICATION

from llama_index.core.schema import BaseNode

logger = logging.getLogger(__name__)

class EntityRelationGraphBuilder(GraphBuilder):
    """
    EntityRelationGraphBuilder is a specialized builder class responsible for creating
    entity relationship graphs based on fact data contained within nodes. It processes
    facts, generates corresponding graph relationships, and interacts with the graph store
    to persist these relationships. This class integrates domain-specific metadata and is
    configured to handle optional domain labels for nodes and relationships.

    Attributes:
        DEFAULT_CLASSIFICATION (str): The default classification label used when fact
            classification is not provided.
    """
    @classmethod
    def index_key(cls) -> str:
        """
        Returns the index key associated with this class. The index key is
        a unique identifier for objects of this type, allowing for effective
        lookup or indexing operations.

        Returns:
            str: The index key for the class instance.
        """
        return 'fact'
    
    def build(self, node:BaseNode, graph_client: GraphStore, **kwargs:Any):
        """
        Builds and executes entity-relation creation logic based on the given node and fact metadata. It processes
        the metadata to create or update relationships between entities in a graph database. If `include_domain_labels`
        is specified, additional domain-specific labels will be added to the entities and relationships.

        Args:
            node (BaseNode): The node containing metadata for the fact data to process.
            graph_client (GraphStore): The graph client instance for executing database queries.
            **kwargs (Any): Additional keyword arguments. Must include:
                - include_domain_labels (bool): Indicator for whether domain labels should be included in the query.

        """
        fact_metadata = node.metadata.get('fact', {})
        include_domain_labels = kwargs['include_domain_labels']

        if fact_metadata:

            fact = Fact.model_validate(fact_metadata)

            if fact.subject and fact.object:
        
                logger.debug(f'Inserting entity relations for fact [fact_id: {fact.factId}]')

                statements = [
                    '// insert entity relations',
                    'UNWIND $params AS params'
                ]

                if include_domain_labels:
                    statements.append(f'MERGE (subject:`__Entity__`:{label_from(fact.subject.classification or DEFAULT_CLASSIFICATION)}{{{graph_client.node_id("entityId")}: params.s_id}})')
                    statements.append(f'MERGE (object:`__Entity__`:{label_from(fact.object.classification or DEFAULT_CLASSIFICATION)}{{{graph_client.node_id("entityId")}: params.o_id}})')
                else:
                    statements.append(f'MERGE (subject:`__Entity__`{{{graph_client.node_id("entityId")}: params.s_id}})')
                    statements.append(f'MERGE (object:`__Entity__`{{{graph_client.node_id("entityId")}: params.o_id}})')

                statements.extend([
                    'MERGE (subject)-[r:`__RELATION__`{value: params.p}]->(object)',
                    'ON CREATE SET r.count = 1 ON MATCH SET r.count = r.count + 1'
                ])

                if include_domain_labels:
                    statements.extend([
                        f'MERGE (subject)-[rr:`{relationship_name_from(fact.predicate.value)}`]->(object)',
                        'ON CREATE SET rr.count = 1 ON MATCH SET rr.count = rr.count + 1'
                    ])


                properties = {
                    's_id': fact.subject.entityId,
                    'o_id': fact.object.entityId,
                    'p': fact.predicate.value
                }
            
                query = '\n'.join(statements)
                    
                graph_client.execute_query_with_retry(query, self._to_params(properties), max_attempts=5, max_wait=7)

            else:
                logger.debug(f'SPC fact, so not creating relation [fact_id: {fact.factId}]')
           

        else:
            logger.warning(f'fact_id missing from fact node [node_id: {node.node_id}]')