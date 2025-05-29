# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Dict

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


class DedupResults(ProcessorBase):
    """
    Processes search results by deduplicating and aggregating topics, chunks, and
    statements to ensure unique, ordered, and scored data per source.

    This class is derived from ProcessorBase and serves to provide functionality for
    deduplication of search results. It processes a collection of search results by
    merging duplicated structures based on their source IDs and aggregating their
    scores when duplicates are found, maintaining only unique topics, chunks, and
    statements. The processed results are returned in a sorted and consistent
    manner to facilitate further usage.

    :ivar args: The processing arguments required for initialization and further
        execution of tasks.
        Set during class instantiation.
    :type args: ProcessorArgs
    :ivar filter_config: The filter configuration detailing how data should be
        filtered during processing.
        Set during class instantiation.
    :type filter_config: FilterConfig
    """

    def __init__(self, args: ProcessorArgs, filter_config: FilterConfig):
        """Initializes a new instance of the class with the provided arguments.

        This constructor method is used to set up the instance of the class by
        utilizing the arguments provided for processing and configuration.
        It initializes the parent class and prepares necessary configurations
        to execute further operations.

        Args:
            args (ProcessorArgs): The processing arguments encapsulated within
                a ProcessorArgs instance, which contains necessary
                information for processing tasks.
            filter_config (FilterConfig): Filter configuration encapsulated
                within a FilterConfig instance, which specifies the
                configuration related to filters.
        """
        super().__init__(args, filter_config)

    def _process_results(
        self, search_results: SearchResultCollection, query: QueryBundle
    ) -> SearchResultCollection:
        """Processes and deduplicates search results by merging topics, chunks,
        and statements for each source. The deduplication ensures that unique
        topics, chunks, and statements are retained across all results, while
        their scores are aggregated when duplicates occur. Additionally, the
        statements within each topic are sorted in descending order of their
        scores.

        Args:
            search_results: The collection of search results to be processed and deduplicated.
            query: The query bundle associated with the search results.

        Returns:
            SearchResultCollection: A new collection of search results after deduplication
            and processing.
        """
        deduped_results: Dict[str, SearchResult] = {}

        for search_result in search_results.results:
            source_id = search_result.source.sourceId

            if source_id not in deduped_results:
                deduped_results[source_id] = search_result
                continue
            else:
                deduped_result = deduped_results[source_id]
                for topic in search_result.topics:
                    existing_topic = next(
                        (x for x in deduped_result.topics if x.topic == topic.topic),
                        None,
                    )
                    if not existing_topic:
                        deduped_result.topics.append(topic)
                        continue
                    else:
                        for chunk in topic.chunks:
                            existing_chunk = next(
                                (
                                    x
                                    for x in existing_topic.chunks
                                    if x.chunkId == chunk.chunkId
                                ),
                                None,
                            )
                            if not existing_chunk:
                                existing_topic.chunks.append(chunk)
                        for statement in topic.statements:
                            existing_statement = next(
                                (
                                    x
                                    for x in existing_topic.statements
                                    if x.statement == statement.statement
                                ),
                                None,
                            )
                            if not existing_statement:
                                existing_topic.statements.append(statement)
                            else:
                                existing_statement.score += statement.score

        for search_result in search_results.results:
            for topic in search_result.topics:
                topic.statements = sorted(
                    topic.statements, key=lambda x: x.score, reverse=True
                )

        search_results = search_results.with_new_results(
            results=[r for r in deduped_results.values()]
        )

        return search_results
