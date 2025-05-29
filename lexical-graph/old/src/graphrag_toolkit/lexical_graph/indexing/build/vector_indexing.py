# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from tqdm import tqdm
from typing import List, Any, Union

from graphrag_toolkit.lexical_graph.storage.vector import VectorStore
from graphrag_toolkit.lexical_graph.storage import VectorStoreFactory
from graphrag_toolkit.lexical_graph.indexing.node_handler import NodeHandler
from graphrag_toolkit.lexical_graph.indexing.build.vector_batch_client import VectorBatchClient
from graphrag_toolkit.lexical_graph.storage.constants import INDEX_KEY, ALL_EMBEDDING_INDEXES, DEFAULT_EMBEDDING_INDEXES

from llama_index.core.schema import BaseNode

logger = logging.getLogger(__name__)

VectorStoreInfoType = Union[str, VectorStore]

class VectorIndexing(NodeHandler):
    """
    Handles vector indexing for nodes, integrating vector stores and managing
    batch operations.

    This class is designed to interface with vector stores, allowing nodes to
    be efficiently indexed based on their metadata. It supports both individual
    and batch indexing operations, with optional progress display. Users can
    configure vector store operations via the provided methods.

    :ivar vector_store: Instance of the vector store being used for indexing.
    :type vector_store: VectorStore
    """
    @staticmethod
    def for_vector_store(vector_store_info:VectorStoreInfoType=None, index_names=DEFAULT_EMBEDDING_INDEXES, **kwargs):
        """
        Constructs and returns a `VectorIndexing` instance for a specified vector store.

        This method facilitates the creation of a `VectorIndexing` object. If an instance
        of `VectorStore` is provided, it directly utilizes that instance. Otherwise, it
        creates a new `VectorStore` using `VectorStoreFactory` and returns the
        corresponding `VectorIndexing` object.

        :param vector_store_info: Vector store instance or the information required to
            construct a vector store.
        :type vector_store_info: VectorStoreInfoType, optional

        :param index_names: List of names for the embedding indexes to be utilized.
            Defaults to `DEFAULT_EMBEDDING_INDEXES`.
        :type index_names: list, optional

        :param kwargs: Additional keyword arguments to be passed to
            `VectorStoreFactory.for_vector_store`.

        :return: A `VectorIndexing` instance created for the given vector store or
            vector store information.
        :rtype: VectorIndexing
        """
        if isinstance(vector_store_info, VectorStore):
            return VectorIndexing(vector_store=vector_store_info)
        else:
            return VectorIndexing(vector_store=VectorStoreFactory.for_vector_store(vector_store_info, index_names, **kwargs))
    
    vector_store:VectorStore

    def accept(self, nodes: List[BaseNode], **kwargs: Any):
        """
        Accepts a list of nodes and performs vector indexing with configurable batch processing and additional keyword
        arguments. Includes optional progress visualization and handles batched operations through the VectorBatchClient.

        This method processes each node, checks metadata for indexing information, and appropriately adds embeddings to
        specified indexes. The batch processing behavior is governed by the batch configuration. Nodes that satisfy specific
        criteria are yielded, and batch operations are applied on remaining nodes before yielding them.

        :param nodes: A list of nodes to be indexed.
        :type nodes: List[BaseNode]
        :param kwargs: Additional keyword arguments for configuring vector indexing and batch processing. Should include:
            'batch_writes_enabled': A boolean flag indicating if batch writes are enabled.
            'batch_write_size': An integer specifying the size of each write batch.
        :type kwargs: Any
        :return: A generator yielding nodes that satisfy specified criteria after applying vector indexing operations.
        :rtype: Generator[BaseNode, None, None]
        """
        batch_writes_enabled = kwargs.pop('batch_writes_enabled')
        batch_write_size = kwargs.pop('batch_write_size')

        logger.debug(f'Batch config: [batch_writes_enabled: {batch_writes_enabled}, batch_write_size: {batch_write_size}]')
        logger.debug(f'Vector indexing kwargs: {kwargs}')
        
        with VectorBatchClient(vector_store=self.vector_store, batch_writes_enabled=batch_writes_enabled, batch_write_size=batch_write_size) as batch_client:

            node_iterable = nodes if not self.show_progress else tqdm(nodes, desc=f'Building vector index [batch_writes_enabled: {batch_writes_enabled}, batch_write_size: {batch_write_size}]')

            for node in node_iterable:
                if [key for key in [INDEX_KEY] if key in node.metadata]:
                    try:
                        index_name = node.metadata[INDEX_KEY]['index']
                        if index_name in ALL_EMBEDDING_INDEXES:
                            index = batch_client.get_index(index_name)
                            index.add_embeddings([node])
                    except Exception as e:
                        logger.exception('An error occurred while indexing vectors')
                        raise e
                if batch_client.allow_yield(node):
                    yield node

            batch_nodes = batch_client.apply_batch_operations()
            for node in batch_nodes:
                yield node
