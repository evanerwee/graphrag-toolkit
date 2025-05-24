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
)

from llama_index.core.schema import QueryBundle


class PruneResults(ProcessorBase):
    """
    Handles processing tasks by pruning search results based on scores relative
    to a predefined pruning threshold. This class is used as a base for
    processing search results to filter out less relevant items.

    The purpose of this class is to efficiently manage and refine search
    results by applying a pruning mechanism. It uses a score threshold to
    exclude search results that do not meet the relevance criteria. Users can
    configure the threshold through the associated arguments.

    :ivar args: The arguments required for processing tasks, including the pruning threshold.
    :type args: ProcessorArgs
    :ivar filter_config: Configuration settings for filtering during processing.
    :type filter_config: FilterConfig
    """

    def __init__(self, args: ProcessorArgs, filter_config: FilterConfig):
        """Initializes the base class for processing tasks with specified
        arguments and filter configuration.

        Args:
            args (ProcessorArgs): The arguments required for processing tasks.
            filter_config (FilterConfig): Configuration settings for filtering during processing.
        """
        super().__init__(args, filter_config)

    def _process_results(
        self, search_results: SearchResultCollection, query: QueryBundle
    ) -> SearchResultCollection:
        """Processes the search results by applying a pruning function based on
        the results' scores relative to a predefined threshold. Any search
        result with a score below the threshold is excluded. This method
        modifies the search results collection to retain only those results
        meeting the score criterion.

        Args:
            search_results: The collection of search results to be processed. Each result may either
                be retained or pruned based on its score relative to the pruning threshold.
            query: The query bundle associated with the search results, providing context for processing.

        Returns:
            SearchResultCollection: A new collection of search results with only those results whose
            scores meet the pruning threshold retained.
        """

        def prune_search_result(index: int, search_result: SearchResult):
            return (
                search_result
                if search_result.score >= self.args.results_pruning_threshold
                else None
            )

        return self._apply_to_search_results(search_results, prune_search_result)
