# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from graphrag_toolkit.lexical_graph.metadata import FilterConfig
from graphrag_toolkit.lexical_graph.retrieval.processors import (
    ProcessorBase,
    ProcessorArgs,
)
from graphrag_toolkit.lexical_graph.retrieval.model import SearchResultCollection

from llama_index.core.schema import QueryBundle


class TruncateResults(ProcessorBase):
    """
    Represents a processor that truncates search results to a specified maximum.

    This class is responsible for handling and processing a collection of search
    results by shortening the list to conform to a predefined maximum number of
    results. It leverages the configuration provided during initialization and
    enables efficient handling of search results.

    :ivar args: Arguments used to configure the processor, including settings
        like the maximum number of results.
    :type args: ProcessorArgs
    :ivar filter_config: Configuration details defining the filters or additional
        specifications for processing the results.
    :type filter_config: FilterConfig
    """

    def __init__(self, args: ProcessorArgs, filter_config: FilterConfig):
        """
        Initializes the main processor class with provided arguments and filter configurations.
        This class is a subclass of a base processor class, and its purpose is to handle specific
        processing tasks based on the arguments and filter configurations supplied during initialization.

        :param args: Processor arguments used for configuring behavior and execution flow
         :type args: ProcessorArgs
        :param filter_config: Configuration of the filter settings for processing
         :type filter_config: FilterConfig
        """
        super().__init__(args, filter_config)

    def _process_results(
        self, search_results: SearchResultCollection, query: QueryBundle
    ) -> SearchResultCollection:
        """Processes the search results by truncating the number of results to
        a defined maximum.

        This method modifies a SearchResultCollection object by trimming its results
        based on the `max_search_results` attribute specified in the `args`. It ensures
        that only the top-ranked results up to this maximum limit are retained.

        Args:
            search_results: A collection of search results to process.
            query: The query information associated with the search results.

        Returns:
            A SearchResultCollection object with the results truncated to the specified
            maximum number.
        """
        search_results.results = search_results.results[: self.args.max_search_results]
        return search_results
