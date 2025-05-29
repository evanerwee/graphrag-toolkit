# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, Any

class ProcessorArgs():
    """ProcessorArgs is a configuration class for managing various options and
    parameters used by a processing system or pipeline.

    The class provides an interface to define, store, and retrieve a wide range
    of settings, including options for query expansion, subquery derivation,
    fact inclusion, reranking strategies, thresholds, and limits for processing.
    The settings are initialized through keyword arguments, allowing for
    customizable configurations. This class ensures efficient handling and
    retrieval of these parameters during runtime.

    Attributes:
        expand_entities (bool): Controls whether entity expansion is enabled
            during processing.
        include_facts (bool): Indicates whether to include facts in the output.
        derive_subqueries (bool): Specifies if subqueries should be derived
            from the main query.
        debug_results (list): A list for storing intermediate debug results.
        reranker (str): Defines the reranking strategy to be employed.
        max_statements (int): The maximum number of statements to process.
        max_search_results (int): The maximum number of search results to
            retrieve.
        max_statements_per_topic (int): The cap on statements related to a
            single topic.
        max_keywords (int): The upper limit on the number of keywords used
            in processing.
        max_subqueries (int): The maximum number of subqueries to generate.
        intermediate_limit (int): The threshold for intermediate results during
            processing.
        query_limit (int): The maximum number of queries to handle at once.
        vss_top_k (int): Determines the top K results to include from a
            vector search system.
        vss_diversity_factor (int): Factor for ensuring diversity in vector
            search results.
        statement_pruning_threshold (float): Minimum threshold for pruning
            irrelevant statements.
        results_pruning_threshold (float): Threshold for pruning processed results.
        num_workers (int): Specifies the number of worker threads to utilize.
        reranking_source_metadata_fn (Optional[Callable]): Function for managing
            source metadata during reranking.
        source_formatter (Optional[Callable]): Function for formatting the
            source in the output.
        ecs_max_score_factor (int): Maximum score factor for entity context
            scaling.
        ecs_min_score_factor (float): Minimum score factor for entity context
            scaling.
        ecs_max_contexts (int): Limit on the number of entity contexts to handle.
        ecs_max_entities_per_context (int): Restriction on how many entities
            are considered per context.
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
        self.statement_pruning_threshold = kwargs.get('statement_pruning_threshold', 0.01)
        self.results_pruning_threshold = kwargs.get('results_pruning_threshold', 0.08)
        self.num_workers = kwargs.get('num_workers', 10)
        self.reranking_source_metadata_fn = kwargs.get('reranking_source_metadata_fn', None)
        self.source_formatter = kwargs.get('source_formatter', None)
        self.ecs_max_score_factor = kwargs.get('ecs_max_score_factor', 2)
        self.ecs_min_score_factor = kwargs.get('ecs_min_score_factor', 0.25)
        self.ecs_max_contexts = kwargs.get('ecs_max_contexts', 4)
        self.ecs_max_entities_per_context = kwargs.get('ecs_max_entities_per_context', 5)

  
    def to_dict(self, new_args:Dict[str, Any]={}):
        """
        Transforms the instance attributes and additional arguments into a single dictionary.

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
        """
        Returns a string representation of the object for debugging and logging purposes. This method
        ensures that the object is represented as a string in dictionary format, which is particularly
        useful for verifying its state or structure during development.

        Returns:
            str: A string representation of the object in dictionary format.
        """
        return str(self.to_dict())