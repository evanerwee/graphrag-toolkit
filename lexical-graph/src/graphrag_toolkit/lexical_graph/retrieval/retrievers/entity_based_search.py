# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import concurrent.futures
from typing import List, Generator, Tuple, Any, Optional, Type

from graphrag_toolkit.lexical_graph.metadata import FilterConfig
from graphrag_toolkit.lexical_graph.retrieval.model import (
    SearchResultCollection,
    ScoredEntity,
    Entity,
)
from graphrag_toolkit.lexical_graph.storage.graph import GraphStore
from graphrag_toolkit.lexical_graph.storage.vector.vector_store import VectorStore
from graphrag_toolkit.lexical_graph.retrieval.retrievers.keyword_entity_search import (
    KeywordEntitySearch,
)
from graphrag_toolkit.lexical_graph.retrieval.processors import (
    ProcessorBase,
    ProcessorArgs,
)
from graphrag_toolkit.lexical_graph.retrieval.retrievers.traversal_based_base_retriever import (
    TraversalBasedBaseRetriever,
)

from llama_index.core.schema import QueryBundle

logger = logging.getLogger(__name__)


class EntityBasedSearch(TraversalBasedBaseRetriever):
    """
    Handles entity-based search functionality using traversal-based retriever mechanisms.

    This class facilitates searching within a hybrid entity-graph and vector-search
    environment, enabling complex search operations using entities and their relationships.

    Attributes:
        graph_store (GraphStore): Storage for graph data.
        vector_store (VectorStore): Storage for vector data.
        processor_args (ProcessorArgs): Configuration for processors.
        processors (List[Type[ProcessorBase]]): Processing pipeline components.
        filter_config (FilterConfig): Configuration for search filtering.
    """

    def __init__(
        self,
        graph_store: GraphStore,
        vector_store: VectorStore,
        processor_args: Optional[ProcessorArgs] = None,
        processors: Optional[List[Type[ProcessorBase]]] = None,
        filter_config: Optional[FilterConfig] = None,
        **kwargs,
    ):
        """Initializes the class instance with the provided parameters and
        ensures that the base class is properly initialized as well. This
        constructor prepares the object with necessary configurations and data
        stores for its intended purpose.

        Args:
            graph_store: The GraphStore instance responsible for managing graph-
                based data and interactions.
            vector_store: The VectorStore instance used for handling vector-based
                storage and similarity operations.
            processor_args: Optional. A ProcessorArgs instance containing additional
                arguments for configuring processors, if provided.
            processors: Optional. A list of ProcessorBase class types for defining
                the processing pipeline, if applicable.
            filter_config: Optional. A FilterConfig instance used for applying
                filtering rules, if specified.
            **kwargs: Extra keyword arguments passed to the superclass for extended
                configurability.
        """
        super().__init__(
            graph_store=graph_store,
            vector_store=vector_store,
            processor_args=processor_args,
            processors=processors,
            filter_config=filter_config,
            **kwargs,
        )

    def get_start_node_ids(self, query_bundle: QueryBundle) -> List[str]:
        """Retrieve start node IDs for an entity-based search or a keyword-
        entity-based search method.

        If entities are already present within the object, their corresponding IDs are returned.
        Otherwise, the function performs a keyword-entity search using the provided query and returns
        the retrieved entities' IDs.

        Args:
            query_bundle (QueryBundle): The query data used to perform the keyword-entity search
                when entities are not available.

        Returns:
            List[str]: A list of entity IDs as strings.
        """
        logger.debug('Getting start node ids for entity-based search...')

        if self.entities:
            return [entity.entity.entityId for entity in self.entities]

        keyword_entity_search = KeywordEntitySearch(
            graph_store=self.graph_store,
            max_keywords=self.args.max_keywords,
            expand_entities=self.args.expand_entities,
        )

        entity_search_results = keyword_entity_search.retrieve(query_bundle)

        entities = [
            ScoredEntity(
                entity=Entity.model_validate_json(entity_search_result.text),
                score=entity_search_result.score,
            )
            for entity_search_result in entity_search_results
        ]

        return [entity.entity.entityId for entity in entities]

    def _for_each_disjoint(
        self, values: List[Any], others: Optional[List[Any]] = None
    ) -> Generator[Tuple[Any, List[Any]], None, None]:
        """Generates tuples of a value and a list of other values that do not
        include the value from the provided lists. The method operates on a
        list of values, creating a set for quick operations, and optionally
        uses another list to specify the other values. The generator yields a
        tuple containing each value and a corresponding list of other values.

        Args:
            values (List[Any]): A mandatory list of values to process.
            others (Optional[List[Any]]): An optional list specifying other values
                excluding the currently selected value.

        Yields:
            Tuple[Any, List[Any]]: A tuple containing a value from the input list
                and a list of other values excluding the selected value.
        """
        values_as_set = set(values)
        for value in values:
            other_values = others or list(values_as_set.difference({value}))
            yield (value, other_values)

    def _for_each_disjoint_unique(
        self, values: List[Any]
    ) -> Generator[Tuple[Any, List[Any]], None, None]:
        for idx, value in enumerate(values[:-1]):
            other_values = values[idx + 1 :]
            yield (value, other_values)

    def _multiple_entity_based_graph_search(
        self, start_id, end_ids, query: QueryBundle
    ):
        """Executes a multiple-entity-based graph search query for the given
        start ID and a list of end IDs, based on the provided query bundle. The
        function constructs a Cypher query to identify paths between entities
        in a graph, retrieves matching nodes, and queries statements and facts
        related to these entities.

        Args:
            start_id: The starting node ID for the graph search.
            end_ids: A list of ending node IDs to find paths to from the start ID.
            query: The query bundle containing additional parameters and configurations for the
                graph search.

        Returns:
            The result of the executed Cypher query, containing paths and matching entities
            information.

        Raises:
            Any exceptions raised during graph query creation or execution.
        """
        logger.debug(
            f'Starting multiple-entity-based searches for [start_id: {start_id}, end_ids: {end_ids}]'
        )

        cypher = self.create_cypher_query(
            f''' 
        // multiple entity-based graph search                                                                
        MATCH p=(e1:`__Entity__`{{{self.graph_store.node_id("entityId")}:$startId}})-[:`__RELATION__`*1..2]-(e2:`__Entity__`) 
        WHERE {self.graph_store.node_id("e2.entityId")} in $endIds
        UNWIND nodes(p) AS n
        WITH DISTINCT COLLECT(n) AS entities
        MATCH (s:`__Entity__`)-[:`__SUBJECT__`]->(f:`__Fact__`)<-[:`__OBJECT__`]-(o:`__Entity__`),
            (f)-[:`__SUPPORTS__`]->(:`__Statement__`)
            -[:`__PREVIOUS__`*0..1]-(l:`__Statement__`)
            -[:`__BELONGS_TO__`]->(t:`__Topic__`)
        WHERE s in entities and o in entities
        '''
        )

        properties = {
            'startId': start_id,
            'endIds': end_ids,
            'statementLimit': self.args.intermediate_limit,
            'limit': self.args.query_limit,
        }

        return self.graph_store.execute_query(cypher, properties)

    def _single_entity_based_graph_search(self, entity_id, query: QueryBundle):
        """Performs a search in a graph database based on a specific entity ID
        and a query.

        The method constructs a Cypher query to perform a graph search focused on a single
        entity. It navigates through relationships and nodes in the graph, retrieving
        statements, related facts, and associated topics. The search parameters, including
        limits, are applied using the query and properties.

        Args:
            entity_id: The identifier of the entity for which the search is performed.
            query: An instance of QueryBundle that contains query-related details.

        Returns:
            The result of the executed Cypher query as returned by the graph database.
        """
        logger.debug(
            f'Starting single-entity-based search for [entity_id: {entity_id}]'
        )

        cypher = self.create_cypher_query(
            f''' 
        // single entity-based graph search                            
        MATCH (:`__Entity__`{{{self.graph_store.node_id("entityId")}:$startId}})
            -[:`__SUBJECT__`]->(f:`__Fact__`)
            -[:`__SUPPORTS__`]->(:`__Statement__`)
            -[:`__PREVIOUS__`*0..1]-(l:`__Statement__`)
            -[:`__BELONGS_TO__`]->(t:`__Topic__`)'''
        )

        properties = {
            'startId': entity_id,
            'statementLimit': self.args.intermediate_limit,
            'limit': self.args.query_limit,
        }

        return self.graph_store.execute_query(cypher, properties)

    def do_graph_search(
        self, query_bundle: QueryBundle, start_node_ids: List[str]
    ) -> SearchResultCollection:
        """Executes a graph-based search combining multiple and single entity-
        based searches.

        This method performs a graph search using both multi-entity and single-entity
        approaches. It executes the search in parallel across multiple workers for efficiency.
        The results are consolidated into a SearchResultCollection.

        Args:
            query_bundle (QueryBundle): Data structure containing the query details, such as
                metadata and input query text, required for search operations.
            start_node_ids (List[str]): List of node IDs from which the search operation
                begins in the graph.

        Returns:
            SearchResultCollection: A collection of search results obtained from the
                graph-based search operation.
        """
        logger.debug('Running entity-based search...')

        search_results = []

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.args.num_workers
        ) as executor:

            futures = [
                executor.submit(
                    self._multiple_entity_based_graph_search,
                    start_id,
                    end_ids,
                    query_bundle,
                )
                for (start_id, end_ids) in self._for_each_disjoint(start_node_ids)
            ]

            futures.extend(
                [
                    executor.submit(
                        self._single_entity_based_graph_search, entity_id, query_bundle
                    )
                    for entity_id in start_node_ids
                ]
            )

            executor.shutdown()

            for future in futures:
                for result in future.result():
                    search_results.append(result)

        search_results_collection = self._to_search_results_collection(search_results)

        retriever_name = type(self).__name__
        if retriever_name in self.args.debug_results and logger.isEnabledFor(
            logging.DEBUG
        ):
            logger.debug(
                f'''Entity-based results: {search_results_collection.model_dump_json(
                    indent=2, 
                    exclude_unset=True, 
                    exclude_defaults=True, 
                    exclude_none=True, 
                    warnings=False)
                }'''
            )

        return search_results_collection
