# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import concurrent.futures
from typing import List, Optional, Type

from graphrag_toolkit.lexical_graph.metadata import FilterConfig
from graphrag_toolkit.lexical_graph.retrieval.model import SearchResultCollection
from graphrag_toolkit.lexical_graph.storage.vector.vector_store import VectorStore
from graphrag_toolkit.lexical_graph.storage.graph import GraphStore
from graphrag_toolkit.lexical_graph.retrieval.processors import (
    ProcessorBase,
    ProcessorArgs,
)
from graphrag_toolkit.lexical_graph.retrieval.retrievers.traversal_based_base_retriever import (
    TraversalBasedBaseRetriever,
)
from graphrag_toolkit.lexical_graph.retrieval.utils.vector_utils import (
    get_diverse_vss_elements,
)

from llama_index.core.schema import QueryBundle
from llama_index.core.vector_stores.types import MetadataFilters

logger = logging.getLogger(__name__)


class ChunkBasedSearch(TraversalBasedBaseRetriever):
    """
    Handles chunk-based retrieval and search operations.
    """

    def __init__(
        self,
        graph_store: GraphStore,
        vector_store: VectorStore,
        processor_args: Optional[ProcessorArgs] = None,
        processors: Optional[List[Type[ProcessorBase]]] = None,
        filter_config: Optional[FilterConfig] = None,
        **kwargs,
    ):
        """Initializes an instance of a class by setting up graph and vector
        stores, optional processing configurations, and other keyword
        arguments. This constructor is used to setup necessary parameters and
        configurations to enable the functionality of the class instance. It
        manages dependencies, configurations, and extensions that enable
        enhanced data handling and processing.

        Args:
            graph_store: The storage mechanism for graph data.
            vector_store: The storage mechanism for vector data.
            processor_args: Optional set of arguments that configure how
                processors handle incoming operations.
            processors: Optional list of processor types to handle specific operations
                or processing tasks.
            filter_config: Optional configuration data for filtering certain types of
                content or operations.
            **kwargs: Additional keyword arguments to pass into the parent
                initialization.
        """
        super().__init__(
            graph_store=graph_store,
            vector_store=vector_store,
            processor_args=processor_args,
            processors=processors,
            filter_config=filter_config,
            **kwargs,
        )

    def chunk_based_graph_search(self, chunk_id):
        """Performs a graph search query based on a specific chunk ID."""
        cypher = self.create_cypher_query(
            f'''
        // chunk-based graph search                                  
        MATCH (l:`__Statement__`)-[:`__PREVIOUS__`*0..1]-(:`__Statement__`)-[:`__BELONGS_TO__`]->(t:`__Topic__`)-[:`__MENTIONED_IN__`]->(c:`__Chunk__`)
        WHERE {self.graph_store.node_id("c.chunkId")} = $chunkId
        '''
        )

        properties = {
            'chunkId': chunk_id,
            'limit': self.args.query_limit,
            'statementLimit': self.args.intermediate_limit,
        }

        return self.graph_store.execute_query(cypher, properties)

    def get_start_node_ids(self, query_bundle: QueryBundle) -> List[str]:
        """Gets the starting node IDs for chunk-based search."""
        logger.debug('Getting start node ids for chunk-based search...')

        chunks = get_diverse_vss_elements(
            'chunk', query_bundle, self.vector_store, self.args, self.filter_config
        )

        return [chunk['chunk']['chunkId'] for chunk in chunks]

    def do_graph_search(
        self, query_bundle: QueryBundle, start_node_ids: List[str]
    ) -> SearchResultCollection:
        """Performs graph search using chunk-based retrieval strategy."""
        chunk_ids = start_node_ids

        logger.debug('Running chunk-based search...')

        search_results = []

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.args.num_workers
        ) as executor:

            futures = [
                executor.submit(self.chunk_based_graph_search, chunk_id)
                for chunk_id in chunk_ids
            ]

            executor.shutdown()

            for future in futures:
                for result in future.result():
                    search_results.append(result)

        search_results_collection = self._to_search_results_collection(search_results)

        retriever_name = type(self).__name__
        if retriever_name in self.args.debug_results and logger.isEnabledFor(
            logging.DEBUG
        ):
            logger.debug(
                f'''Chunk-based results: {search_results_collection.model_dump_json(
                    indent=2, 
                    exclude_unset=True, 
                    exclude_defaults=True, 
                    exclude_none=True, 
                    warnings=False)
                }'''
            )

        return search_results_collection
