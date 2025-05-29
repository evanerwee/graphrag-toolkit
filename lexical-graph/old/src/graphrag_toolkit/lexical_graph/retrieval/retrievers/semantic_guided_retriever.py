# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import concurrent.futures
import logging
from collections import defaultdict
from typing import List, Optional, Any, Union, Type
from itertools import repeat

from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

from graphrag_toolkit.lexical_graph.metadata import FilterConfig
from graphrag_toolkit.lexical_graph.storage.graph import GraphStore
from graphrag_toolkit.lexical_graph.storage.vector import VectorStore

from graphrag_toolkit.lexical_graph.retrieval.retrievers.semantic_guided_base_retriever import (
    SemanticGuidedBaseRetriever,
)
from graphrag_toolkit.lexical_graph.retrieval.retrievers.keyword_ranking_search import (
    KeywordRankingSearch,
)
from graphrag_toolkit.lexical_graph.retrieval.retrievers.statement_cosine_seach import (
    StatementCosineSimilaritySearch,
)
from graphrag_toolkit.lexical_graph.retrieval.retrievers.semantic_beam_search import (
    SemanticBeamGraphSearch,
)
from graphrag_toolkit.lexical_graph.retrieval.retrievers.rerank_beam_search import (
    RerankingBeamGraphSearch,
)
from graphrag_toolkit.lexical_graph.retrieval.utils.statement_utils import (
    get_statements_query,
    SharedEmbeddingCache,
)

logger = logging.getLogger(__name__)

SemanticGuidedRetrieverType = Union[
    SemanticGuidedBaseRetriever, Type[SemanticGuidedBaseRetriever]
]


class SemanticGuidedRetriever(SemanticGuidedBaseRetriever):
    """
    Defines a semantic-guided retriever that integrates multiple retriever classes
    for performing semantic and keyword-based searches. This composite system is
    built upon shared embedding caches and supports vector and graph-based result
    retrieval. The retriever is designed for flexibility, allowing dynamic
    retriever initialization and advanced search capabilities.

    :ivar share_results: Indicates whether results from different retrievers
        should be shared for additional processing.
    :type share_results: bool
    :ivar shared_embedding_cache: Shared embedding cache to store and reuse
        precomputed embeddings across retrievers for improved efficiency.
    :type shared_embedding_cache: SharedEmbeddingCache
    :ivar initial_retrievers: List of initial retrievers for base-level
        information retrieval.
    :type initial_retrievers: List[Union[SemanticGuidedBaseRetriever, Type[SemanticGuidedBaseRetriever]]]
    :ivar graph_retrievers: List of graph-based retrievers for expanding initial
        retrieved results based on graph relationships.
    :type graph_retrievers: List[SemanticGuidedBaseRetriever]
    """

    def __init__(
        self,
        vector_store: VectorStore,
        graph_store: GraphStore,
        retrievers: Optional[
            List[Union[SemanticGuidedBaseRetriever, Type[SemanticGuidedBaseRetriever]]]
        ] = None,
        share_results: bool = True,
        filter_config: Optional[FilterConfig] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initializes an instance of the class by configuring vector and graph stores, retrievers,
        and shared embedding cache. Additionally, it sets up default retrievers and manages
        retriever-specific configurations including graph retriever instantiation.

        :param vector_store: The vector store instance used for managing embeddings and search.
                             Must be of type `VectorStore`.
        :type vector_store: VectorStore
        :param graph_store: The graph store instance used for managing graph-like structures
                            and connectivity. Must be of type `GraphStore`.
        :type graph_store: GraphStore
        :param retrievers: A list of retriever instances or their types. Retrievers may be
                           constructed dynamically and include optional graph-based retrievers
                           or semantic-guided retrievers. Default is None.
        :type retrievers: Optional[List[Union[SemanticGuidedBaseRetriever, Type[SemanticGuidedBaseRetriever]]]]
        :param share_results: A flag that indicates whether results should be shared across
                              the retrievers. Defaults to True.
        :type share_results: bool
        :param filter_config: Optional filter configuration settings used for initializing the class.
                              Default is None.
        :type filter_config: Optional[FilterConfig]
        :param kwargs: Additional keyword arguments that may be passed to the retriever instances
                       during their initialization or other internal configuration settings.
        :type kwargs: Any
        """
        super().__init__(vector_store, graph_store, filter_config, **kwargs)

        self.share_results = share_results

        # Create shared embedding cache
        self.shared_embedding_cache = SharedEmbeddingCache(vector_store)

        self.initial_retrievers = []
        self.graph_retrievers = []

        # initialize retrievers
        if retrievers:
            for retriever in retrievers:
                if isinstance(retriever, type):
                    instance = retriever(vector_store, graph_store, **kwargs)
                else:
                    instance = retriever

                # Inject shared cache if not already set
                if (
                    hasattr(instance, 'embedding_cache')
                    and instance.embedding_cache is None
                ):
                    instance.embedding_cache = self.shared_embedding_cache

                if isinstance(
                    instance, (SemanticBeamGraphSearch, RerankingBeamGraphSearch)
                ):
                    self.graph_retrievers.append(instance)
                else:
                    self.initial_retrievers.append(instance)
        else:
            # Default configuration
            self.initial_retrievers = [
                StatementCosineSimilaritySearch(
                    vector_store=vector_store,
                    graph_store=graph_store,
                    embedding_cache=self.shared_embedding_cache,
                    **kwargs,
                ),
                KeywordRankingSearch(
                    vector_store=vector_store,
                    graph_store=graph_store,
                    embedding_cache=self.shared_embedding_cache,
                    **kwargs,
                ),
            ]

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """
        Retrieves and processes nodes based on the initial retrieval from retrievers, applies filters, and orders them.
        This method performs a multi-step process which involves retrieving initial results, applying graph-based expansions,
        fetching and updating metadata for nodes, and applying specified filtering and ordering logic.

        :param query_bundle: The input query encapsulated as a QueryBundle instance.
        :type query_bundle: QueryBundle
        :return: A list of NodeWithScore objects containing ordered and filtered nodes.
        :rtype: List[NodeWithScore]
        """
        # 1. Get initial results in parallel
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=len(self.initial_retrievers)
        ) as p:
            initial_results = list(
                p.map(
                    lambda r, query: r.retrieve(query),
                    self.initial_retrievers,
                    repeat(query_bundle),
                )
            )

        if logger.isEnabledFor(logging.DEBUG) and self.debug_results:
            logger.debug(f'initial_results: {initial_results}')
        else:
            logger.debug(f'num initial_results: {len(initial_results)}')

        # 2. Collect unique initial nodes
        seen_statement_ids = set()
        initial_nodes = []
        for nodes in initial_results:
            for node in nodes:
                statement_id = node.node.metadata['statement']['statementId']
                if statement_id not in seen_statement_ids:
                    seen_statement_ids.add(statement_id)
                    initial_nodes.append(node)

        all_nodes = initial_nodes.copy()

        if logger.isEnabledFor(logging.DEBUG) and self.debug_results:
            logger.debug(f'all_nodes (before expansion): {all_nodes}')
        else:
            logger.debug(f'num all_nodes (before expansion): {len(all_nodes)}')

        # 3. Graph expansion if enabled
        if self.share_results and initial_nodes:
            for retriever in self.graph_retrievers:
                try:
                    retriever.shared_nodes = initial_nodes
                    graph_nodes = retriever.retrieve(query_bundle)
                    for node in graph_nodes:
                        statement_id = node.node.metadata['statement']['statementId']
                        if statement_id not in seen_statement_ids:
                            seen_statement_ids.add(statement_id)
                            all_nodes.append(node)
                except Exception as e:
                    logger.error(
                        f"Error in graph retriever {retriever.__class__.__name__}: {e}"
                    )
                    continue

        if logger.isEnabledFor(logging.DEBUG) and self.debug_results:
            logger.debug(f'all_nodes (after expansion): {all_nodes}')
        else:
            logger.debug(f'num all_nodes (after expansion): {len(all_nodes)}')

        # 4. Fetch statements once
        if not all_nodes:
            return []

        statement_ids = [
            node.node.metadata['statement']['statementId'] for node in all_nodes
        ]
        statements = get_statements_query(self.graph_store, statement_ids)

        if logger.isEnabledFor(logging.DEBUG) and self.debug_results:
            logger.debug(f'statements: {statements}')
        else:
            logger.debug(f'num statements: {len(statements)}')

        # 5. Create final nodes with full data
        final_nodes = []
        statements_map = {
            s['result']['statement']['statementId']: s['result'] for s in statements
        }

        for node in all_nodes:
            statement_id = node.node.metadata['statement']['statementId']
            if statement_id in statements_map:
                result = statements_map[statement_id]
                new_node = TextNode(
                    text=result['statement']['value'],
                    metadata={
                        **node.node.metadata,  # Preserve retriever metadata
                        'statement': result['statement'],
                        'chunk': result['chunk'],
                        'source': result['source'],
                    },
                )
                final_nodes.append(NodeWithScore(node=new_node, score=node.score))

        if logger.isEnabledFor(logging.DEBUG) and self.debug_results:
            logger.debug(f'final_nodes: {final_nodes}')
        else:
            logger.debug(f'num final_nodes: {len(final_nodes)}')

        # 6. Apply metadata filters
        filtered_nodes = [
            node
            for node in final_nodes
            if self.filter_config.filter_source_metadata_dictionary(
                node.node.metadata['source']['metadata']
            )
        ]

        if logger.isEnabledFor(logging.DEBUG) and self.debug_results:
            logger.debug(f'filter_nodes: {filtered_nodes}')
        else:
            logger.debug(f'num filter_nodes: {len(filtered_nodes)}')

        # 7. Group by source for better context
        source_nodes = defaultdict(list)
        for node in filtered_nodes:
            source_id = node.node.metadata['source']['sourceId']
            source_nodes[source_id].append(node)

        # 8. Create final ordered list
        ordered_nodes = []
        for source_id, nodes in source_nodes.items():
            nodes.sort(key=lambda x: x.score or 0.0, reverse=True)
            ordered_nodes.extend(nodes)

        if logger.isEnabledFor(logging.DEBUG) and self.debug_results:
            logger.debug(f'ordered_nodes: {ordered_nodes}')
        else:
            logger.debug(f'num ordered_nodes: {len(ordered_nodes)}')

        return ordered_nodes
