# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import time
import tfidf_matcher as tm
from typing import List, Dict
from dateutil.parser import parse

from graphrag_toolkit.lexical_graph.metadata import FilterConfig
from graphrag_toolkit.lexical_graph import GraphRAGConfig
from graphrag_toolkit.lexical_graph.retrieval.model import Source
from graphrag_toolkit.lexical_graph.retrieval.processors import ProcessorBase, ProcessorArgs
from graphrag_toolkit.lexical_graph.retrieval.post_processors import SentenceReranker
from graphrag_toolkit.lexical_graph.retrieval.model import SearchResultCollection, SearchResult, Topic, ScoredEntity

from llama_index.core.schema import QueryBundle, NodeWithScore, TextNode
from llama_index.core.node_parser import TokenTextSplitter

logger = logging.getLogger(__name__)

def default_reranking_source_metadata_fn(source:Source):
    """
    Formats metadata values of a source into a readable string for reranking purposes. Each metadata value
    is processed to check if it can be parsed as a date or if it starts with an HTTP URL. Dates are formatted
    to a "Month Day, Year" format, while valid URLs are ignored. If the metadata value is neither a date
    nor a URL, it is included unchanged in the output. The formatted values are joined together as a
    comma-separated string.

    Args:
        source (Source): An object containing metadata as a dictionary with string keys and values.

    Returns:
        str: A string containing the formatted metadata values, joined by commas.
    """
    def format_value(s):
        try: 
            date = parse(s, fuzzy=False)
            return date.strftime("%B %-d, %Y")
        except ValueError:
            if s.startswith('http'):
                return ''
            else:
                return s
    return ', '.join([format_value(v) for v in source.metadata.values()])

class RerankStatements(ProcessorBase):
    """

    """
    def __init__(self, args:ProcessorArgs, filter_config:FilterConfig, reranking_model=None):
        self.reranking_model = reranking_model or GraphRAGConfig.reranking_model
        super().__init__(args, filter_config)
        self.reranking_source_metadata_fn = self.args.reranking_source_metadata_fn or default_reranking_source_metadata_fn

    def _score_values_with_tfidf(self, values:List[str], query:QueryBundle, entities:List[ScoredEntity]):
        """
        Compute a ranking of provided text values using the TF-IDF (Term Frequency-Inverse Document Frequency) algorithm.
        The method reranks the input values based on their relevance to the provided query and any additional entities.
        It ensures that the scoring is performed with consideration to a pre-defined maximum number of statements if specified
        in the object's configuration. TF-IDF matching is utilized to calculate relevance scores and produce a sorted mapping
        of values by their relevance scores.

        Args:
            values (List[str]): The list of text values to be scored based on their relevance to the query and entities.
            query (QueryBundle): The query object containing the query string to match against the text values.
            entities (List[ScoredEntity]): A collection of entities that provide additional context for scoring the text values.

        Returns:
            dict: A dictionary where keys are the text values and values are their associated relevance scores, sorted
            in descending order of relevance.
        """
        logger.debug('Reranking with tfidf')

        splitter = TokenTextSplitter(chunk_size=25, chunk_overlap=5)
        match_values = splitter.split_text(query.query_str)

        extras = set([
            entity.entity.value
            for entity in entities
        ])

        if extras:
            match_values.append(', '.join(extras))

        logger.debug(f'Match values: {match_values}')
 
        values_to_score = values.copy()
        
        limit =  len(values_to_score)
        if self.args.max_statements:
            limit = min(self.args.max_statements, limit)

        while len(values_to_score) <= limit:
            values_to_score.append('')

        scored_values = {}

        try:
            
            matcher_results = tm.matcher(match_values, values_to_score, limit, 3)

            max_i = len(matcher_results.columns)
        
            for row_index in range(0, len(match_values)):
                for col_index in range(1, max_i, 3) :
                    value = matcher_results.iloc[row_index, col_index]
                    score = matcher_results.iloc[row_index, col_index+1]
                    if value not in scored_values:
                        scored_values[value] = score
                    else:
                        scored_values[value] = max(scored_values[value], score)
        except ValueError:
            scored_values = {v: 0.0 for v in values_to_score if v}
        
                
        sorted_scored_values = dict(sorted(scored_values.items(), key=lambda item: item[1], reverse=True))
        
        return sorted_scored_values

    def _score_values(self, values:List[str], query:QueryBundle, entities:List[ScoredEntity]) -> Dict[str, float]:
        """
        Reranks a list of values based on their relevance to a specified query and associated entities using a
        SentenceReranker model. The method processes the input values, constructs a query bundle considering the
        entities if present, and finally returns a dictionary mapping each value to its computed score.

        Args:
            values (List[str]): A list of strings to be scored and reranked based on relevance.
            query (QueryBundle): The query bundle containing the query string and any metadata for ranking.
            entities (List[ScoredEntity]): A list of scored entities used to refine the query for reranking.

        Returns:
            Dict[str, float]: A dictionary where keys are the original input values and values are their corresponding
                scores after reranking.
        """
        logger.debug('Reranking with SentenceReranker')

        reranker = SentenceReranker(model=self.reranking_model, top_n=self.args.max_statements or len(values))

        rank_query = (
            query 
            if not entities 
            else QueryBundle(query_str=f'{query.query_str} (keywords: {", ".join(set([entity.entity.value for entity in entities]))})')
        )

        reranked_values = reranker.postprocess_nodes(
            [
                NodeWithScore(node=TextNode(text=value), score=0.0)
                for value in values
            ],
            rank_query
        )

        return {
            reranked_value.text : reranked_value.score
            for reranked_value in reranked_values
        }

    def _process_results(self, search_results:SearchResultCollection, query:QueryBundle) -> SearchResultCollection:
        """
        Processes search results by reranking statements within each topic based on their relevance scores.

        This method processes a set of search results, and if a reranking approach is specified,
        it calculates scores for the statements within topics using the supplied query and entities.
        It then reranks statements based on the computed scores and updates the search results accordingly.

        Args:
            search_results (SearchResultCollection): A collection of search results that contain topics
                and their associated statements to be processed.
            query (QueryBundle): A query object that includes the query text and any additional contextual
                entity information required for processing.

        Returns:
            SearchResultCollection: A modified collection of search results where statements within each
            topic are reranked based on the relevance scores computed by the specified reranking method.

        Raises:
            Any exception raised during the reranking process will propagate to the caller.
        """
        if not self.args.reranker or self.args.reranker.lower() == 'none':
            return search_results
       
        values_to_score = []
        
        for search_result in search_results.results:
            source_str = self.reranking_source_metadata_fn(search_result.source)
            for topic in search_result.topics:
                topic_str = topic.topic
                for statement in topic.statements:
                    statement_str = statement.statement_str
                    values_to_score.append(self._format_statement_context(source_str, topic_str, statement_str))
        
        start = time.time()

        scored_values = None
        if self.args.reranker.lower() == 'model':
            scored_values = self._score_values(values_to_score, query, search_results.entities)
        else:
            scored_values = self._score_values_with_tfidf(values_to_score, query, search_results.entities)

        end = time.time()

        rerank_ms = (end-start) * 1000

        logger.debug(f'Rerank duration: {rerank_ms:.2f}ms')

        processor_name = type(self).__name__
        if processor_name in self.args.debug_results and logger.isEnabledFor(logging.DEBUG):
            logger.debug('Scored values:\n' + '\n--------------\n'.join([str(scored_value) for scored_value in scored_values.items()]))

        def rerank_statements(topic:Topic, source_str:str):
            """
            Represents a processor that reranks statements within topics based on their scores.

            This class is used to process and reorder statements in a `SearchResultCollection`
            object using scores associated with the statements. Statements are filtered and
            sorted in descending order by their scores.

            Attributes:
                No attributes are defined directly for this class.

            Methods:
                _process_results: Processes the search results to rerank the statements based
                    on their scores.

            """
            topic_str = topic.topic
            surviving_statements = []
            for statement in topic.statements:
                statement_str = statement.statement_str
                key = self._format_statement_context(source_str, topic_str, statement_str)
                if key in scored_values:
                    statement.score = round(float(scored_values[key]), 4)
                    surviving_statements.append(statement)
            topic.statements = sorted(surviving_statements, key=lambda x: x.score, reverse=True)
            return topic

        def rerank_search_result(index:int, search_result:SearchResult):
            """
            Re-ranks search results based on specific criteria and applies modifications to their
            topics using a metadata source function and a specified reranking method. This
            processor works by overriding the `_process_results` method and provides the
            necessary mechanisms to handle individual search results in a collection.

            Methods:
                _process_results: Processes a collection of search results and applies the reranking
                to individual entries based on the provided query.

            Args:
                index (int): Position index of the search result to rerank within the collection.
                search_result (SearchResult): Individual search result object containing the
                    data to be reranked.
            """
            source_str = self.reranking_source_metadata_fn(search_result.source)
            return self._apply_to_topics(search_result, rerank_statements, source_str=source_str)
        
        return self._apply_to_search_results(search_results, rerank_search_result)


