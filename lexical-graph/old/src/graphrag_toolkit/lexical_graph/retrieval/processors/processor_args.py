# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, Any


class ProcessorArgs:
    """A container class for managing and configuring arguments used in a
    processing pipeline.

    This class serves as a flexible structure for holding and managing configuration settings
    and parameters required for processing tasks. It provides default values for various options
    while allowing customization through keyword arguments. Users can interact with its attributes
    directly or convert its state into a dictionary for easier manipulation and transport.

    Attributes:
        expand_entities (bool): Specifies whether to expand entities during processing.
        include_facts (bool): Determines whether facts are included in the output.
        derive_subqueries (bool): Enables deriving subqueries during processing.
        debug_results (list): A collection of debug information or results.
        reranker (str): Strategy for ranking or reranking results, default is 'tfidf'.
        max_statements (int): Maximum number of statements allowed during processing.
        max_search_results (int): Maximum number of search results to retrieve.
        max_statements_per_topic (int): Limit on the number of statements allowed per topic.
        max_keywords (int): Maximum number of keywords considered in processing.
        max_subqueries (int): Maximum number of subqueries to generate.
        intermediate_limit (int): Interim limit for restricting processing steps.
        query_limit (int): Maximum number of queries to execute.
        vss_top_k (int): Top-k value for vector-space search ranking.
        vss_diversity_factor (int): Factor for introducing diversity in vector search results.
        statement_pruning_threshold (float): Threshold for pruning statements during processing.
        results_pruning_threshold (float): Threshold for pruning search results during processing.
        num_workers (int): Number of worker threads or processes for parallel tasks.
        reranking_source_metadata_fn (Optional[Callable]): A function for reranking based on source metadata.
        source_formatter (Optional[Callable]): A formatter function for handling source data representation.
        ecs_max_score_factor (float): Maximum score factor for entity-context scoring.
        ecs_min_score_factor (float): Minimum score factor for entity-context scoring.
        ecs_max_contexts (int): Limit on the number of entity contexts considered.
        ecs_max_entities_per_context (int): Maximum number of entities allowed per context.
    """

    def __init__(self, **kwargs):

        self.expand_entities = kwargs.get('expand_entities', True)
        self.include_facts = kwargs.get('include_facts', False)
        self.derive_subqueries = kwargs.get('derive_subqueries', False)
        self.debug_results = kwargs.get('debug_results', [])
        self.reranker = kwargs.get('reranker', 'tfidf')
        self.max_statements = kwargs.get('max_statements', 100)
        self.max_search_results = kwargs.get('max_search_results', 5)
        self.max_statements_per_topic = kwargs.get('max_statements_per_topic', 10)
        self.max_keywords = kwargs.get('max_keywords', 10)
        self.max_subqueries = kwargs.get('max_subqueries', 2)
        self.intermediate_limit = kwargs.get('intermediate_limit', 50)
        self.query_limit = kwargs.get('query_limit', 10)
        self.vss_top_k = kwargs.get('vss_top_k', 10)
        self.vss_diversity_factor = kwargs.get('vss_diversity_factor', 5)
        self.statement_pruning_threshold = kwargs.get(
            'statement_pruning_threshold', 0.01
        )
        self.results_pruning_threshold = kwargs.get('results_pruning_threshold', 0.08)
        self.num_workers = kwargs.get('num_workers', 10)
        self.reranking_source_metadata_fn = kwargs.get(
            'reranking_source_metadata_fn', None
        )
        self.source_formatter = kwargs.get('source_formatter', None)
        self.ecs_max_score_factor = kwargs.get('ecs_max_score_factor', 2)
        self.ecs_min_score_factor = kwargs.get('ecs_min_score_factor', 0.25)
        self.ecs_max_contexts = kwargs.get('ecs_max_contexts', 4)
        self.ecs_max_entities_per_context = kwargs.get(
            'ec2_max_entities_per_context', 5
        )

    def to_dict(self, new_args: Dict[str, Any] = {}):
        """Transforms the instance attributes and additional arguments into a
        single dictionary.

        This method combines the instance's `__dict__` attributes, representing the object's
        current state, with any additional dictionary of arguments provided.

        Args:
            new_args (Dict[str, Any], optional): A dictionary of additional arguments to merge
                with the object's instance attributes. Defaults to an empty dictionary.

        Returns:
            Dict[str, Any]: A dictionary containing the merged instance attributes and any
            additional arguments.
        """
        args = self.__dict__
        return args | new_args

    def __repr__(self):
        """Returns a string representation of the object for debugging and
        logging purposes. This method ensures that the object is represented as
        a string in dictionary format, which is particularly useful for
        verifying its state or structure during development.

        Returns:
            str: A string representation of the object in dictionary format.
        """
        return str(self.to_dict())
