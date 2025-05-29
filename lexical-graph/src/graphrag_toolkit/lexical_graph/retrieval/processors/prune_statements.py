# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import logging

from graphrag_toolkit.lexical_graph.metadata import FilterConfig
from graphrag_toolkit.lexical_graph.retrieval.processors import ProcessorBase, ProcessorArgs
from graphrag_toolkit.lexical_graph.retrieval.model import SearchResultCollection, SearchResult, Topic

from llama_index.core.schema import QueryBundle

logger = logging.getLogger(__name__)

class PruneStatements(ProcessorBase):
    """
    PruneStatements is responsible for processing search results by pruning statements
    below a specified score threshold.

    This class performs statement pruning based on a score threshold defined in the
    provided configuration. The pruning process is applied to all statements in the
    topics of the search results, ensuring only those statements meeting or exceeding
    the specified threshold are kept. It extends ProcessorBase and leverages its
    infrastructure to perform operations on the given search results.

    Attributes:
        args (ProcessorArgs): Arguments and settings for processing, including the
            statement pruning threshold.
        filter_config (FilterConfig): Configuration parameters for the filtering
            process, defining constraints and additional settings.
    """
    def __init__(self, args:ProcessorArgs, filter_config:FilterConfig):
        """
        Initializes the class instance and sets up the filter configuration and processing
        logic by leveraging the provided arguments.

        Args:
            args (ProcessorArgs): The processing arguments that configure the behavior
                and parameters of the processing tasks.
            filter_config (FilterConfig): The configuration object containing settings
                required to set up the filtering mechanisms appropriately.
        """
        super().__init__(args, filter_config)

    def _process_results(self, search_results:SearchResultCollection, query:QueryBundle) -> SearchResultCollection:
        """
        Processes the search results by applying pruning operations based on a defined statement
        pruning threshold. Specifically, it filters out statements within topics that do not meet the
        required score threshold and applies this pruning to search results.

        Args:
            search_results: A collection of search results that contain topics, statements, and
                associated metadata.
            query: A representation of the user's query for which the search results are being
                processed.

        Returns:
            A processed collection of search results after pruning the statements in each topic
            within the results.
        """
        def prune_statements(topic:Topic):
            """
            Processes the results from a search by filtering out statements based on a defined
            score threshold. This functionality allows only those statements that meet or exceed
            the threshold to be retained in the `Topic.statements`.

            The class is a specialized processor for search results and operates on each `Topic`
            object within a collection of search results.

            Args:
                topic (Topic): A topic containing a collection of statements. Each statement
                    has a score used to determine its eligibility for retention.
            """
            surviving_statements = [
                s 
                for s in topic.statements 
                if s.score >= self.args.statement_pruning_threshold
            ]
            topic.statements = surviving_statements
            return topic

        def prune_search_result(index:int, search_result:SearchResult):
            """
            A processor class to prune search results by applying a specific pruning method
            to topics within each search result.

            This class inherits from ProcessorBase and implements a custom processing
            function that iterates over search results and selectively prunes statements
            by calling the `_apply_to_topics` method.
            """
            return self._apply_to_topics(search_result, prune_statements)
        
        return self._apply_to_search_results(search_results, prune_search_result)


