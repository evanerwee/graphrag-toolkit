# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import concurrent.futures
import logging
from functools import reduce
from typing import List, Dict, Set, Any, Optional, Tuple, Iterator

from graphrag_toolkit.lexical_graph.metadata import FilterConfig
from graphrag_toolkit.lexical_graph import GraphRAGConfig
from graphrag_toolkit.lexical_graph.utils import LLMCache, LLMCacheType
from graphrag_toolkit.lexical_graph.storage.graph import GraphStore
from graphrag_toolkit.lexical_graph.storage.vector import VectorStore
from graphrag_toolkit.lexical_graph.retrieval.utils.statement_utils import (
    get_top_k,
    SharedEmbeddingCache,
)
from graphrag_toolkit.lexical_graph.retrieval.prompts import (
    EXTRACT_KEYWORDS_PROMPT,
    EXTRACT_SYNONYMS_PROMPT,
)
from graphrag_toolkit.lexical_graph.retrieval.retrievers.semantic_guided_base_retriever import (
    SemanticGuidedBaseRetriever,
)

from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode
from llama_index.core.prompts import PromptTemplate

logger = logging.getLogger(__name__)


class KeywordRankingSearch(SemanticGuidedBaseRetriever):
    """
    The KeywordRankingSearch class is a specialized retriever extending the
    SemanticGuidedBaseRetriever. It provides functionality to extract and rank
    keywords from a query, enabling more effective search and retrieval processes.
    This involves processing query input using a language model and identifying
    matches in the graph store. The class supports customization with various prompts,
    filter configurations, and embedded caching to optimize performance.

    :ivar embedding_cache: Shared embedding cache for optimizing operations.
    :type embedding_cache: Optional[SharedEmbeddingCache]
    :ivar llm: A large language model cache or instance used for query processing.
    :type llm: LLMCacheType
    :ivar max_keywords: Maximum number of keywords to extract from the query.
    :type max_keywords: int
    :ivar keywords_prompt: A prompt template used specifically for extracting keywords.
    :type keywords_prompt: str
    :ivar synonyms_prompt: A prompt template used for extracting synonyms to aid keyword retrieval.
    :type synonyms_prompt: str
    :ivar top_k: Maximum number of top results to consider after retrieval.
    :type top_k: int
    """

    def __init__(
        self,
        vector_store: VectorStore,
        graph_store: GraphStore,
        embedding_cache: Optional[SharedEmbeddingCache] = None,
        keywords_prompt: str = EXTRACT_KEYWORDS_PROMPT,
        synonyms_prompt: str = EXTRACT_SYNONYMS_PROMPT,
        llm: LLMCacheType = None,
        max_keywords: int = 10,
        top_k: int = 100,
        filter_config: Optional[FilterConfig] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initializes an instance of the class.

        This constructor initializes the object with necessary storage, embedding cache,
        keywords and synonyms prompt, large language model, and filters. It also allows
        additional custom parameters through variable keyword arguments. This configuration
        enables the object to perform tasks associated with keyword extraction and synonym
        generation using the provided LLM and storage backends.

        :param vector_store: A storage backend for managing vectors.
        :type vector_store: VectorStore
        :param graph_store: A storage backend for managing graphs.
        :type graph_store: GraphStore
        :param embedding_cache: Shared embedding cache to optimize retrieval operations. Defaults to None.
        :type embedding_cache: Optional[SharedEmbeddingCache]
        :param keywords_prompt: A string prompt for extracting keywords. Defaults to EXTRACT_KEYWORDS_PROMPT.
        :type keywords_prompt: str
        :param synonyms_prompt: A string prompt for extracting synonyms. Defaults to EXTRACT_SYNONYMS_PROMPT.
        :type synonyms_prompt: str
        :param llm: A large language model cache type. If not provided, it defaults to a configuration object.
        :type llm: LLMCacheType, optional
        :param max_keywords: Maximum number of keywords to extract. Defaults to 10.
        :type max_keywords: int
        :param top_k: Maximum number of top candidates to consider. Defaults to 100.
        :type top_k: int
        :param filter_config: Configuration for applying filters to the data. Defaults to None.
        :type filter_config: Optional[FilterConfig]
        :param kwargs: Additional keyword arguments for customization.
        :type kwargs: Any
        """
        super().__init__(vector_store, graph_store, filter_config, **kwargs)
        self.embedding_cache = embedding_cache
        self.llm = (
            llm
            if llm and isinstance(llm, LLMCache)
            else LLMCache(
                llm=llm or GraphRAGConfig.response_llm,
                enable_cache=GraphRAGConfig.enable_cache,
            )
        )
        self.max_keywords = max_keywords
        self.keywords_prompt = keywords_prompt
        self.synonyms_prompt = synonyms_prompt
        self.top_k = top_k

    def get_keywords(self, query_bundle: QueryBundle) -> Set[str]:
        """
        Extracts a set of keywords from the provided query bundle using an external language
        model. The function processes the query through two prompts: one for keywords and
        another for synonyms, combining the results into a single set of keywords. It uses
        a ThreadPoolExecutor for concurrent processing of prompts and logs the extracted
        keywords. If any error occurs during extraction, an empty set is returned.

        :param query_bundle: The query bundle containing the query string to be processed.
        :type query_bundle: QueryBundle
        :return: A set of keywords extracted from the query.
        :rtype: Set[str]
        """

        def extract(prompt):
            """
            KeywordRankingSearch class is a specialized retriever extending the
            SemanticGuidedBaseRetriever. It provides functionality to extract and rank
            keywords from a query, enabling more effective search and retrieval processes.
            This involves processing query input using a language model and returning a set
            of extracted keywords.

            Attributes
            ----------
            llm : Any
                Language model used for processing queries and extracting keywords.
            max_keywords : int
                The maximum number of keywords to extract from the query.

            Methods
            -------
            get_keywords(query_bundle)
                Extracts keywords from the provided query bundle.
            """
            response = self.llm.predict(
                PromptTemplate(template=prompt),
                text=query_bundle.query_str,
                max_keywords=self.max_keywords,
            )
            return {kw.strip().lower() for kw in response.strip().split('^')}

        try:
            with concurrent.futures.ThreadPoolExecutor() as p:
                keyword_batches: Iterator[Set[str]] = p.map(
                    extract, (self.keywords_prompt, self.synonyms_prompt)
                )
                unique_keywords = reduce(lambda x, y: x.union(y), keyword_batches)
                logger.debug(f"Extracted keywords: {unique_keywords}")
                return unique_keywords
        except Exception as e:
            logger.error(f"Error extracting keywords: {e}")
            return set()

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """
        Retrieves and ranks a list of nodes with scores based on keyword matching and
        similarity to the query.

        The function performs the following steps:
        1. Extracts keywords from the input query bundle.
        2. Executes a database query to find matching statements based on the keywords.
        3. Groups and ranks the results by the number of keyword matches.
        4. Optionally reranks results within groups using similarity scoring.
        5. Returns a limited number of top-ranked nodes.

        :param query_bundle: The input query bundle containing the query text and
            associated metadata.
        :type query_bundle: QueryBundle
        :return: A list of nodes with scores representing ranked results based on
            keyword matches and computed similarity.
        :rtype: List[NodeWithScore]
        """
        # 1. Get keywords
        keywords = self.get_keywords(query_bundle)
        if not keywords:
            logger.warning("No keywords extracted from query")
            return []

        logger.debug(f'keywords: {keywords}')

        # 2. Find statements matching any keyword
        cypher = f"""
        UNWIND $keywords AS keyword
        MATCH (e:`__Entity__`)
        WHERE toLower(e.value) = toLower(keyword)
        WITH e, keyword
        MATCH (e)-[:`__SUBJECT__`|`__OBJECT__`]->(:`__Fact__`)-[:`__SUPPORTS__`]->(statement:`__Statement__`)
        WITH statement, COLLECT(DISTINCT keyword) as matched_keywords
        RETURN {{
            statement: {{
                statementId: {self.graph_store.node_id("statement.statementId")}
            }},
            matched_keywords: matched_keywords
        }} AS result
        """

        results = self.graph_store.execute_query(cypher, {'keywords': list(keywords)})
        if not results:
            logger.debug("No statements found matching keywords")
            return []

        if logger.isEnabledFor(logging.DEBUG) and self.debug_results:
            logger.debug(f'results: {results}')
        else:
            logger.debug(f'num results: {len(results)}')

        # 3. Group statements by number of keyword matches
        statements_by_matches: Dict[int, List[Tuple[str, Set[str]]]] = {}
        for result in results:
            statement_id = result['result']['statement']['statementId']
            matched_keywords = set(result['result']['matched_keywords'])
            num_matches = len(matched_keywords)
            if num_matches not in statements_by_matches:
                statements_by_matches[num_matches] = []
            statements_by_matches[num_matches].append((statement_id, matched_keywords))

        # 4. Process groups in order of most matches
        final_nodes = []
        for num_matches in sorted(statements_by_matches.keys(), reverse=True):
            group = statements_by_matches[num_matches]

            # If there are ties, use similarity to rank within group
            if len(group) > 1:
                statement_ids = [sid for sid, _ in group]
                statement_embeddings = self.embedding_cache.get_embeddings(
                    statement_ids
                )

                scored_statements = get_top_k(
                    query_bundle.embedding, statement_embeddings, len(statement_ids)
                )

                if logger.isEnabledFor(logging.DEBUG) and self.debug_results:
                    logger.debug(f'scored_statements: {scored_statements}')
                else:
                    logger.debug(f'num scored_statements: {len(scored_statements)}')

                # Create nodes with scores and keyword information
                keyword_map = {sid: kw for sid, kw in group}
                for score, statement_id in scored_statements:
                    matched_keywords = keyword_map[statement_id]
                    node = TextNode(
                        text="",  # Placeholder
                        metadata={
                            'statement': {'statementId': statement_id},
                            'search_type': 'keyword_ranking',
                            'keyword_matches': list(matched_keywords),
                            'num_keyword_matches': len(matched_keywords),
                        },
                    )
                    # Normalize score using both keyword matches and similarity
                    combined_score = (num_matches / len(keywords)) * (score + 1) / 2
                    final_nodes.append(NodeWithScore(node=node, score=combined_score))
            else:
                # Single statement in group
                statement_id, matched_keywords = group[0]
                node = TextNode(
                    text="",  # Placeholder
                    metadata={
                        'statement': {'statementId': statement_id},
                        'search_type': 'keyword_ranking',
                        'keyword_matches': list(matched_keywords),
                        'num_keyword_matches': len(matched_keywords),
                    },
                )
                score = num_matches / len(keywords)
                final_nodes.append(NodeWithScore(node=node, score=score))

        # 5. Limit to top_k if specified
        if self.top_k:
            final_nodes.sort(key=lambda x: x.score or 0.0, reverse=True)
            final_nodes = final_nodes[: self.top_k]

        if logger.isEnabledFor(logging.DEBUG) and self.debug_results:
            logger.debug(f'final_nodes: {final_nodes}')
        else:
            logger.debug(f'num final_nodes: {len(final_nodes)}')

        return final_nodes
