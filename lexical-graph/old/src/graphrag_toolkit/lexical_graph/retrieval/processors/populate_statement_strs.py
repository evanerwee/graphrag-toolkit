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


class PopulateStatementStrs(ProcessorBase):
    """
    This class extends ProcessorBase to process search results by populating detailed
    statement strings for each topic within the results.

    The PopulateStatementStrs class is designed to enhance the topics in a collection of search
    results. It processes topics by adding more information into their statement strings, derived
    from associated facts and details. This results in enriched search results with statements
    that better represent the underlying topics.

    :ivar args: Configuration arguments required for the processor.
    :type args: ProcessorArgs
    :ivar filter_config: Configuration specific to the filtering logic for the processor.
    :type filter_config: FilterConfig
    """

    def __init__(self, args: ProcessorArgs, filter_config: FilterConfig):
        """Initializes the instance of the class with the provided
        ProcessorArgs and FilterConfig.

        Args:
            args (ProcessorArgs): The set of arguments that configure the processor.
            filter_config (FilterConfig): The configuration settings for the filter.
        """
        super().__init__(args, filter_config)

    def _process_results(
        self, search_results: SearchResultCollection, query: QueryBundle
    ) -> SearchResultCollection:
        """Processes search results by enriching each search result's topics
        with detailed string information, derived from associated statements,
        facts, and additional details. This method modifies and returns a new
        collection of search results where each result includes the updated
        statement strings for its topics.

        Args:
            search_results (SearchResultCollection): A collection of search results that
                need processing.
            query (QueryBundle): The query associated with the provided search results.

        Returns:
            SearchResultCollection: A processed collection of search results with enriched
            topic statement strings.
        """

        def populate_statement_strs(topic: Topic):
            """This class is a processor that populates statement strings in
            each topic of a search result collection. For each topic, it
            processes their statements by incorporating related facts and
            details into a combined string representation.

            The `_process_results` method iterates through the topics in a collection and
            updates their statements with additional details.

            Args:
                search_results (SearchResultCollection): The collection of search results
                    containing topics to process.
                query (QueryBundle): The query bundle used for the search results.

            Returns:
                SearchResultCollection: A new collection of search results where each topic
                    has updated statements with populated strings.
            """
            for statement in topic.statements:
                statement_details = []
                if statement.facts:
                    statement_details.extend(statement.facts)
                if statement.details:
                    statement_details.extend(statement.details.split('\n'))
                statement.statement_str = (
                    f'{statement.statement} (details: {", ".join(statement_details)})'
                    if statement_details
                    else statement.statement
                )
            return topic

        def populate_search_result_statement_strs(
            index: int, search_result: SearchResult
        ):
            """A processor class to populate statement strings within search
            results for a query.

            This class processes a collection of search results and applies a transformation to
            populate statement strings in each search result based on their topics.

            Attributes:
                None
            """
            return self._apply_to_topics(search_result, populate_statement_strs)

        return self._apply_to_search_results(
            search_results, populate_search_result_statement_strs
        )
