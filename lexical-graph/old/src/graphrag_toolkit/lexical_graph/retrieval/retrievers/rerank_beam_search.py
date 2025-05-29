# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from queue import PriorityQueue
from typing import List, Dict, Set, Tuple, Optional, Any, Union, Type

from graphrag_toolkit.lexical_graph.metadata import FilterConfig
from graphrag_toolkit.lexical_graph.storage.graph import GraphStore
from graphrag_toolkit.lexical_graph.storage.vector import VectorStore
from graphrag_toolkit.lexical_graph.retrieval.utils.statement_utils import (
    get_statements_query,
)
from graphrag_toolkit.lexical_graph.retrieval.retrievers.semantic_guided_base_retriever import (
    SemanticGuidedBaseRetriever,
)
from graphrag_toolkit.lexical_graph.retrieval.post_processors import RerankerMixin

from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

logger = logging.getLogger(__name__)


class RerankingBeamGraphSearch(SemanticGuidedBaseRetriever):
    """
    Handles retrieval, graph-based navigation, and reranking with a beam search
    strategy. Employs semantic-guided techniques to maximize the relevance of
    retrieved items. This class allows for a highly-tuned exploration process with
    adjustable depth, beam width, and an optional reranker component for scoring.

    The purpose of this class is to construct a flexible system enabling retrieval
    tasks over structured knowledge graphs, while employing advanced reranking
    techniques to prioritize results. It supports both manual initial retrievers
    for customization and automatic scoring mechanisms. The usage also encompasses
    graph analysis and neighbor exploration during retrieval.

    :ivar reranker: Reranker component to assign relevance scores based on
        specific criteria.
    :type reranker: RerankerMixin
    :ivar max_depth: Maximum allowable depth for graph exploration or retrieval
        operations.
    :type max_depth: int
    :ivar beam_width: Maximum number of concurrent paths retained during
        beam search at each iteration.
    :type beam_width: int
    :ivar shared_nodes: Optional shared nodes utilized across multiple queries
        or retrieval pipelines for consistency or propagation.
    :type shared_nodes: Optional[List[NodeWithScore]]
    :ivar score_cache: Caches relevance scores for previously evaluated
        statements to avoid redundant computations.
    :type score_cache: Dict[str, float]
    :ivar statement_cache: Caches retrieved statement data for efficiency
        across repeated queries.
    :type statement_cache: Dict[str, Dict]
    :ivar initial_retrievers: List of retrievers employed during the initial
        phase of retrieval operations.
    :type initial_retrievers: List[SemanticGuidedBaseRetriever]

    """

    def __init__(
        self,
        vector_store: VectorStore,
        graph_store: GraphStore,
        reranker: RerankerMixin,
        initial_retrievers: Optional[
            List[Union[SemanticGuidedBaseRetriever, Type[SemanticGuidedBaseRetriever]]]
        ] = None,
        shared_nodes: Optional[List[NodeWithScore]] = None,
        max_depth: int = 3,
        beam_width: int = 10,
        filter_config: Optional[FilterConfig] = None,
        **kwargs: Any,
    ) -> None:
        """Initializes an object that handles retrieval, graph navigation, and
        reranking with a beam search approach. It incorporates customization
        options such as initial retrievers, depth constraint for exploration,
        and beam width for limiting concurrent node evaluations. The class also
        supports shared nodes for result aggregation and caches scores and
        statements for efficiency.

        Args:
            vector_store: Underlying storage mechanism for vectorized representations
                used in approximate nearest neighbor searches.
            graph_store: Graph representation database that stores entities,
                relationships, or nodes with interconnections for navigation.
            reranker: Component responsible for ranking nodes or items based on
                customized scoring criteria.
            initial_retrievers: Optional sequence of retriever instances or types that
                guide the initial stage of the retrieval process before using the
                primary retriever. Each retriever in this list must accept vector_store,
                graph_store, and filter_config as construction parameters.
            shared_nodes: Optional sequence of nodes with scores that are used across
                multiple retrievals or beam searches as common input. Allows result
                propagation or consistency.
            max_depth: Maximum allowable depth for search or traversal within the graph.
                Deep graphs may be restricted to this depth limit for computational
                efficiency.
            beam_width: The maximum number of nodes retained after each iteration
                during the beam search; higher values expand search breadth at a cost
                to performance.
            filter_config: Optional configuration that defines the filtering or pruning
                criteria applied during retrieval, ranking, or navigation.
            **kwargs: Arbitrary additional keyword arguments, passed to base class
                initialization or specific retriever construction.
        """
        super().__init__(vector_store, graph_store, filter_config, **kwargs)
        self.reranker = reranker
        self.max_depth = max_depth
        self.beam_width = beam_width
        self.shared_nodes = shared_nodes
        self.score_cache = {}
        self.statement_cache = {}

        # Initialize initial retrievers if provided
        self.initial_retrievers = []
        if initial_retrievers:
            for retriever in initial_retrievers:
                if isinstance(retriever, type):
                    self.initial_retrievers.append(
                        retriever(vector_store, graph_store, filter_config, **kwargs)
                    )
                else:
                    self.initial_retrievers.append(retriever)

    def get_statements(self, statement_ids: List[str]) -> Dict[str, Dict]:
        """
        Retrieves statement details for the specified list of statement IDs. If any statement ID is not
        cached, it queries the graph store for those IDs, updates the cache, and returns all requested
        statements with their details.

        :param statement_ids: List of statement IDs to retrieve. It is a list of strings containing unique
            identifiers for each statement.
        :type statement_ids: List[str]
        :return: A dictionary where keys are the requested statement IDs and the values are dictionaries
            containing statement details.
        :rtype: Dict[str, Dict]
        """
        uncached_ids = [sid for sid in statement_ids if sid not in self.statement_cache]
        if uncached_ids:
            new_results = get_statements_query(self.graph_store, uncached_ids)
            for result in new_results:
                sid = result['result']['statement']['statementId']
                self.statement_cache[sid] = result['result']

        return {sid: self.statement_cache[sid] for sid in statement_ids}

    def get_neighbors(self, statement_id: str) -> List[str]:
        """
        Retrieves a list of neighboring statements connected to a given statement. The neighbors are determined by
        navigating the graph database relationships between entities and their associated statements as defined
        by the provided Cypher query.

        :param statement_id: The unique identifier of the statement whose neighbors are to be retrieved.
        :type statement_id: str
        :return: A list of unique statement IDs that are neighbors of the given statement.
        :rtype: List[str]
        """
        cypher = f"""
        MATCH (e:`__Entity__`)-[:`__SUBJECT__`|`__OBJECT__`]->(:`__Fact__`)-[:`__SUPPORTS__`]->(s:`__Statement__`)
        WHERE {self.graph_store.node_id('s.statementId')} = $statementId
        WITH s, COLLECT(DISTINCT e) AS entities
        UNWIND entities AS entity
        MATCH (entity)-[:`__SUBJECT__`|`__OBJECT__`]->(:`__Fact__`)-[:`__SUPPORTS__`]->(e_neighbors:`__Statement__`)
        RETURN DISTINCT {self.graph_store.node_id('e_neighbors.statementId')} as statementId
        """

        neighbors = self.graph_store.execute_query(
            cypher, {'statementId': statement_id}
        )
        return [n['statementId'] for n in neighbors]

    def rerank_statements(
        self, query: str, statement_ids: List[str], statement_texts: Dict[str, str]
    ) -> List[Tuple[float, str]]:
        """
        Ranks a list of statement IDs and their corresponding texts based on their
        relevance to a given query. The function utilizes a reranker model to calculate
        relevance scores for each statement. Scores for previously unseen statements are
        cached to optimize performance by avoiding repetitive computations. The returned
        list of tuples contains scores and statement IDs sorted in descending order of
        relevance.

        :param query: A string representing the main query for which the statements
            are ranked.
        :type query: str
        :param statement_ids: A list of statement IDs that will be ranked.
        :type statement_ids: List[str]
        :param statement_texts: A dictionary mapping statement IDs to their corresponding
            texts.
        :type statement_texts: Dict[str, str]
        :return: A list of tuples, where each tuple contains a relevance score and
            a statement ID, sorted by score in descending order.
        :rtype: List[Tuple[float, str]]
        """
        uncached_statements = [
            statement_texts[sid]
            for sid in statement_ids
            if statement_texts[sid] not in self.score_cache
        ]

        if uncached_statements:
            pairs = [(query, statement_text) for statement_text in uncached_statements]

            scores = self.reranker.rerank_pairs(
                pairs=pairs, batch_size=self.reranker.batch_size * 2
            )

            for statement_text, score in zip(uncached_statements, scores):
                self.score_cache[statement_text] = score

        scored_pairs = []
        for sid in statement_ids:
            score = self.score_cache[statement_texts[sid]]
            scored_pairs.append((score, sid))

        scored_pairs.sort(reverse=True)
        return scored_pairs

    def beam_search(
        self, query_bundle: QueryBundle, start_statement_ids: List[str]
    ) -> List[Tuple[str, List[str]]]:
        """
        Performs a beam search over a set of statements, starting from initial
        statements and expanding paths through neighboring statements. The method
        uses a priority queue to maintain and evaluate paths based on their scores.
        The scoring is handled by a reranker which ranks statements based on their
        relevance to a query. The results are collected as a list of statement IDs
        and their corresponding paths up to the beam width specified.

        :param query_bundle: The query descriptor containing the input query string
                             used for statement scoring during reranking.
        :type query_bundle: QueryBundle
        :param start_statement_ids: A list of IDs representing the starting statements
                                    for the beam search.
        :type start_statement_ids: List[str]
        :return: A list of tuples, each containing a statement ID and the path of
                 statement IDs leading to it. The number of results is limited by
                 the beam width.
        :rtype: List[Tuple[str, List[str]]]
        """
        visited: Set[str] = set()
        results: List[Tuple[str, List[str]]] = []
        queue: PriorityQueue = PriorityQueue()

        # Get texts for all start statements
        start_statements = self.get_statements(start_statement_ids)
        statement_texts = {
            sid: statement['statement']['value']
            for sid, statement in start_statements.items()
        }

        # Score initial statements using reranker
        start_scores = self.rerank_statements(
            query_bundle.query_str, start_statement_ids, statement_texts
        )

        # Initialize queue with start statements
        for score, statement_id in start_scores:
            queue.put((-score, 0, statement_id, [statement_id]))

        while not queue.empty() and len(results) < self.beam_width:
            neg_score, depth, current_id, path = queue.get()

            if current_id in visited:
                continue

            visited.add(current_id)
            results.append((current_id, path))

            if depth < self.max_depth:
                # Get and score neighbors
                neighbor_ids = self.get_neighbors(current_id)
                if neighbor_ids:
                    # Get texts for neighbors
                    neighbor_statements = self.get_statements(neighbor_ids)
                    neighbor_texts = {
                        sid: str(
                            statement['statement']['value']
                            + '\n'
                            + statement['statement']['details']
                        )
                        for sid, statement in neighbor_statements.items()
                    }

                    # Score neighbors using reranker
                    scored_neighbors = self.rerank_statements(
                        query_bundle.query_str, neighbor_ids, neighbor_texts
                    )

                    # Add top-k neighbors to queue
                    for score, neighbor_id in scored_neighbors[: self.beam_width]:
                        if neighbor_id not in visited:
                            new_path = path + [neighbor_id]
                            queue.put((-score, depth + 1, neighbor_id, new_path))

        return results

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """
        Retrieve a list of relevant nodes based on the given query bundle using an
        initial retrieval process and a subsequent beam search. The method attempts
        to find the most contextually relevant nodes by combining initial filtering
        and iterative refinement through beam search.

        :param query_bundle: The query information which may include query text,
            metadata, or other parameters necessary for retrieval.
        :type query_bundle: QueryBundle

        :return: A list of nodes with their associated scores representing their
            relevance to the query.
        :rtype: List[NodeWithScore]
        """

        # Get initial nodes (either shared or from initial retrievers)
        initial_statement_ids = set()

        if self.shared_nodes is not None:
            # Use shared nodes if available
            for node in self.shared_nodes:
                initial_statement_ids.add(
                    node.node.metadata['statement']['statementId']
                )
        elif self.initial_retrievers:
            # Get nodes from initial retrievers
            for retriever in self.initial_retrievers:
                nodes = retriever.retrieve(query_bundle)
                for node in nodes:
                    initial_statement_ids.add(
                        node.node.metadata['statement']['statementId']
                    )
        else:
            # Fallback to vector similarity if no initial nodes
            results = self.vector_store.get_index('statement').top_k(
                query_bundle,
                top_k=self.beam_width * 2,
                filter_config=self.filter_config,
            )
            initial_statement_ids = {r['statement']['statementId'] for r in results}

        if not initial_statement_ids:
            logger.warning("No initial statements found for the query.")
            return []

        # Perform beam search
        beam_results = self.beam_search(query_bundle, list(initial_statement_ids))

        # Collect all new statement IDs from beam search
        new_statement_ids = [
            statement_id
            for statement_id, _ in beam_results
            if statement_id not in initial_statement_ids
        ]

        if not new_statement_ids:
            logger.info("Beam search did not find any new statements.")
            return []

        # Create nodes from results
        nodes = []
        statement_to_path = {
            statement_id: path
            for statement_id, path in beam_results
            if statement_id not in initial_statement_ids
        }

        for statement_id, path in statement_to_path.items():
            statement_data = self.statement_cache.get(statement_id)
            if statement_data:
                node = TextNode(
                    text=statement_data['statement']['value'],
                    metadata={
                        'statement': statement_data['statement'],
                        'chunk': statement_data['chunk'],
                        'source': statement_data['source'],
                        'search_type': 'beam_search',
                        'depth': len(path),
                        'path': path,
                    },
                )
                score = self.score_cache.get(statement_data['statement']['value'], 0.0)
                nodes.append(NodeWithScore(node=node, score=score))
            else:
                logger.warning(
                    f"Statement data not found in cache for ID: {statement_id}"
                )

        nodes.sort(key=lambda x: x.score or 0.0, reverse=True)

        logger.info(f"Retrieved {len(nodes)} new nodes through beam search.")
        return nodes
