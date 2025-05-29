# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from graphrag_toolkit.lexical_graph.metadata import FilterConfig
from graphrag_toolkit.lexical_graph.retrieval.processors import (
    ProcessorBase,
    ProcessorArgs,
)
from graphrag_toolkit.lexical_graph.retrieval.model import (
    SearchResultCollection,
    SearchResult,
    Topic,
)

from llama_index.core.schema import QueryBundle


class ZeroScores(ProcessorBase):
    """
    A processor class designed to reset scores associated with search results
    and their related topics.

    This class is used to process `SearchResultCollection` objects by setting
    their scores, as well as the scores within associated topics, to zero.
    It inherits from `ProcessorBase` and extends its functionality to provide
    score-resetting mechanisms. The class ensures uniform score values, which
    helps facilitate predefined filtering or ranking operations.

    :ivar args: The arguments required for processing tasks. This includes
        necessary configurations and settings to perform the processing jobs.
    :type args: ProcessorArgs
    :ivar filter_config: Configuration used to apply filtering mechanisms during
        processing. This object contains the criteria and settings used for data
        filtering.
    :type filter_config: FilterConfig
    """

    def __init__(self, args: ProcessorArgs, filter_config: FilterConfig):
        """Initializes the class inheriting from the parent class and sets up
        processor arguments and filter configuration.

        Args:
            args (ProcessorArgs): The arguments required for processing tasks. This
                includes necessary configurations and settings to perform the
                processing jobs.
            filter_config (FilterConfig): Configuration used to apply filtering
                mechanisms during processing. This object contains the criteria
                and settings used for data filtering.
        """
        super().__init__(args, filter_config)

    def _process_results(
        self, search_results: SearchResultCollection, query: QueryBundle
    ) -> SearchResultCollection:
        """Processes the search results by zeroing out scores associated with
        the statements contained within topics and the search results
        themselves. This method ensures that all scores are reset to a default
        value (0.0) within the given `SearchResultCollection`.

        Args:
            search_results: The collection of search results that needs to be processed.
            query: The query bundle associated with the search results.

        Returns:
            SearchResultCollection: The collection of search results with all scores reset to
            zero, after applying the zeroing operation to the relevant statements and search
            results.
        """

        def zero_statement_scores(topic: Topic):
            """A processor class that sets the scores of all statements within
            topics in the search results to zero.

            This is achieved by applying a helper function to each topic
            in the search results.
            """
            for s in topic.statements:
                s.score = 0.0
            return topic

        def zero_search_result_scores(index: int, search_result: SearchResult):
            """Implements a processor that zeroes out the scores of search
            results.

            This class extends `ProcessorBase` and provides functionality to manipulate
            the scores of search results by setting them to zero. It works on a collection
            of search results and applies a specific transformation to each result. The
            class is designed to process search results while considering topics associated
            with those results.

            Methods:
                _process_results: Processes a collection of search results for a given query
                                  by zeroing out their scores.
            """
            search_result.score = 0.0
            return self._apply_to_topics(search_result, zero_statement_scores)

        return self._apply_to_search_results(search_results, zero_search_result_scores)
