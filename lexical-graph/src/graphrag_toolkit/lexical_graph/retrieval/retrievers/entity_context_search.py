# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import List, Optional, Type, Union

from graphrag_toolkit.lexical_graph.metadata import FilterConfig
from graphrag_toolkit.lexical_graph.retrieval.model import SearchResultCollection, ScoredEntity, Entity, SearchResult
from graphrag_toolkit.lexical_graph.storage.graph import GraphStore
from graphrag_toolkit.lexical_graph.storage.vector.vector_store import VectorStore
from graphrag_toolkit.lexical_graph.retrieval.retrievers.keyword_entity_search import KeywordEntitySearch
from graphrag_toolkit.lexical_graph.retrieval.retrievers.chunk_based_search import ChunkBasedSearch
from graphrag_toolkit.lexical_graph.retrieval.retrievers.topic_based_search import TopicBasedSearch
from graphrag_toolkit.lexical_graph.retrieval.processors import ProcessorBase, ProcessorArgs
from graphrag_toolkit.lexical_graph.retrieval.retrievers.traversal_based_base_retriever import TraversalBasedBaseRetriever

from llama_index.core.schema import QueryBundle

logger = logging.getLogger(__name__)

SubRetrieverType = Union[ChunkBasedSearch, TopicBasedSearch, Type[ChunkBasedSearch], Type[TopicBasedSearch]]

class EntityContextSearch(TraversalBasedBaseRetriever):
    """
    A retriever implementation designed to perform entity-context-based search within a graph database.
    It retrieves relevant nodes and their contexts based on the input query and employs advanced filtering
    and scoring mechanisms to refine results.

    The EntityContextSearch class inherits from TraversalBasedBaseRetriever and combines graph traversal
    and vector search for efficient entity context retrieval. It is optimized for use in systems
    requiring complex entity relational data processing.

    Attributes:
        sub_retriever (Optional[SubRetrieverType]): An optional sub-retriever instance or class used for
            deeper result retrieval during the search process.
    """
    def __init__(self,
                 graph_store:GraphStore,
                 vector_store:VectorStore,
                 processor_args:Optional[ProcessorArgs]=None,
                 processors:Optional[List[Type[ProcessorBase]]]=None,
                 sub_retriever:Optional[SubRetrieverType]=None,
                 filter_config:Optional[FilterConfig]=None,
                 **kwargs):
        """
        Initializes an instance of the class with specified parameters. This constructor
        sets up graph and vector stores, optional processing arguments, a list of
        processors, a sub-retriever, and filter configuration to create a specific
        retrieval or searching mechanism.

        Args:
            graph_store: The graph store to be used for managing graph-based data
                structures within the entity.
            vector_store: The vector store to be used for managing vector-based
                representations of data.
            processor_args: Optional arguments for processors, specifying additional
                configuration or parameters required by the processors.
            processors: Optional list of processor classes implementing the
                ProcessorBase type, used to transform or handle data within the entity.
            sub_retriever: Optional specific retriever type to be used for
                sub-retrieval operations. Defaults to ChunkBasedSearch.
            filter_config: Optional configuration object containing filter
                criteria or rules for data refinement.
            **kwargs: Additional keyword arguments that might be required for further
                customization or parameterization.
        """
        self.sub_retriever = sub_retriever or ChunkBasedSearch
        
        super().__init__(
            graph_store=graph_store, 
            vector_store=vector_store,
            processor_args=processor_args,
            processors=processors,
            filter_config=filter_config,
            **kwargs
        )

    def get_start_node_ids(self, query_bundle: QueryBundle) -> List[str]:
        """
        Retrieves the starting node IDs for an entity context search.

        This method processes a given `QueryBundle` by performing an entity search
        using the `KeywordEntitySearch` utility. The search results are used to
        generate a list of entities along with their associated scores, and finally,
        the method extracts and returns the entity IDs.

        Args:
            query_bundle (QueryBundle): The input query bundle containing the query
                text and associated metadata.

        Returns:
            List[str]: A list of entity IDs that serve as the starting nodes.
        """
        logger.debug('Getting start node ids for entity context search...')
        
        keyword_entity_search = KeywordEntitySearch(
            graph_store=self.graph_store, 
            max_keywords=self.args.max_keywords,
            expand_entities=False,
            filter_config=self.filter_config
        )

        entity_search_results = keyword_entity_search.retrieve(query_bundle)

        entities = [
            ScoredEntity(
                entity=Entity.model_validate_json(entity_search_result.text), 
                score=entity_search_result.score
            )
            for entity_search_result in entity_search_results
        ]

        return [entity.entity.entityId for entity in entities]   

    def _get_entity_contexts(self, start_node_ids:List[str]) -> List[str]:
        """
        Fetches and processes the context of entities based on relationships and scoring criteria.

        This method retrieves entity relationship data from a graph database, computes scores for
        the relationships, and organizes the entities into contexts based on configurable thresholds
        for scoring. The resulting entity contexts are limited in number and structured to contain
        related entities grouped by score evaluation and hierarchy.

        Args:
            start_node_ids (List[str]): A list of starting entity node IDs for which the contexts
                need to be retrieved.

        Returns:
            List[str]: A list of entity contexts, where each context is represented by a list of
                entity values.
        """
        if self.args.ecs_max_contexts < 1:
            return []

        cypher = f'''
        // get entity context
        MATCH (s:`__Entity__`)-[:`__RELATION__`*1..2]-(c:`__Entity__`)
        WHERE {self.graph_store.node_id("s.entityId")} in $entityIds
        AND NOT {self.graph_store.node_id("c.entityId")} in $entityIds
        RETURN {self.graph_store.node_id("s.entityId")} as s, collect(distinct {self.graph_store.node_id("c.entityId")}) as c LIMIT $limit
        '''
        
        properties = {
            'entityIds': start_node_ids,
            'limit': self.args.intermediate_limit
        }
        
        results = self.graph_store.execute_query(cypher, properties)
        
        all_entity_ids = set()
        entity_map = {}
        
        for result in results:
            all_entity_ids.add(result['s'])
            all_entity_ids.update(result['c'])
            entity_map[result['s']] = result['c']
            
        cypher = f'''
        // get entity context scores
        MATCH (s:`__Entity__`)-[r:`__RELATION__`]-()
        WHERE {self.graph_store.node_id("s.entityId")} in $entityIds
        RETURN {self.graph_store.node_id("s.entityId")} as s_id, s.value AS value, sum(r.count) AS score
        '''
        
        properties = {
            'entityIds': list(all_entity_ids)
        }
        
        results = self.graph_store.execute_query(cypher, properties)
        
        entity_score_map = {}
        
        for result in results:
            entity_score_map[result['s_id']] = { 'value': result['value'], 'score': result['score']}
            
        scored_entity_contexts = []
        prime_context = []
        
        for parent, children in entity_map.items():

            parent_entity = entity_score_map[parent]
            parent_score = parent_entity['score']

            context_entities = [parent_entity['value']]
            prime_context.append(parent_entity['value'])

            logger.debug(f'parent: {parent_entity}')

            for child in children:

                child_entity = entity_score_map[child]
                child_score = child_entity['score']

                logger.debug(f'child : {child_entity}')

                if child_score <= (self.args.ecs_max_score_factor * parent_score) and child_score >= (self.args.ecs_min_score_factor * parent_score):
                    context_entities.append(child_entity['value'])

            if len(context_entities) > 1:
                scored_entity_contexts.append({
                    'entities': context_entities[:self.args.ecs_max_entities_per_context],
                    'score': parent_score
                })

        scored_entity_contexts = sorted(scored_entity_contexts, key=lambda ec: ec['score'], reverse=True)

        logger.debug(f'scored_entity_contexts: {scored_entity_contexts}')

        all_entity_contexts = [prime_context]

        for scored_entity_context in scored_entity_contexts:
            entities = scored_entity_context['entities']
            all_entity_contexts.extend([
                entities[x:x+3] 
                for x in range(0, max(1, len(entities) - 2))
            ])

        logger.debug(f'all_entity_contexts: {all_entity_contexts}')

        entity_contexts = all_entity_contexts[:self.args.ecs_max_contexts]
                 
        logger.debug(f'entity_contexts: {entity_contexts}')
        
        return entity_contexts
    
    def _get_sub_retriever(self):
        """
        Retrieves or constructs a sub-retriever based on the type of the existing sub-retriever
        or initializes a new one with the provided configuration.

        This private method either returns an already initialized sub-retriever instance
        if it adheres to a specific type or initializes a new sub-retriever object with
        parameters derived from configuration attributes. The initialized or retrieved
        sub-retriever is logged for debugging purposes.

        Returns:
            TraversalBasedBaseRetriever: An instance of a sub-retriever ready for use.
        """
        sub_retriever = (self.sub_retriever if isinstance(self.sub_retriever, TraversalBasedBaseRetriever)
                         else self.sub_retriever(
                            self.graph_store, 
                            self.vector_store, 
                            vss_top_k=2,
                            max_search_results=2,
                            vss_diversity_factor=self.args.vss_diversity_factor,
                            include_facts=self.args.include_facts,
                            filter_config=self.filter_config
                        ))
        logger.debug(f'sub_retriever: {type(sub_retriever).__name__}')
        return sub_retriever
    
    def do_graph_search(self, query_bundle:QueryBundle, start_node_ids:List[str]) -> SearchResultCollection:
        """
        Executes a graph-based search query starting from provided node IDs and processes the results.

        This method performs an entity-context-based search using given starting node IDs, constructs
        entity contexts, retrieves associated search results, and aggregates those results into a
        collection. It also provides debug logs if debugging is enabled for the specific retriever.

        Args:
            query_bundle (QueryBundle): The query bundle containing the search query information.
            start_node_ids (List[str]): A list of starting node IDs to base the search on.

        Returns:
            SearchResultCollection: A collection of search results aggregated from the entity-context-based search.
        """
        logger.debug('Running entity-context-based search...')

        sub_retriever = self._get_sub_retriever()
        entity_contexts = self._get_entity_contexts(start_node_ids)

        search_results = []

        for entity_context in entity_contexts:
            if entity_context:
                results = sub_retriever.retrieve(QueryBundle(query_str=', '.join(entity_context)))
                for result in results:
                    search_results.append(SearchResult.model_validate(result.metadata))
                    
                
        search_results_collection = SearchResultCollection(results=search_results) 
        
        retriever_name = type(self).__name__
        
        if retriever_name in self.args.debug_results and logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'''Entity context results: {search_results_collection.model_dump_json(
                    indent=2, 
                    exclude_unset=True, 
                    exclude_defaults=True, 
                    exclude_none=True, 
                    warnings=False)
                }''')
        
        return search_results_collection