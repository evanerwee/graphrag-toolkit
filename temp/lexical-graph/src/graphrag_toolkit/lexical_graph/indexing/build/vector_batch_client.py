# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import List
from graphrag_toolkit.lexical_graph.storage.vector import VectorStore, VectorIndex, DummyVectorIndex
from graphrag_toolkit.lexical_graph.storage.constants import ALL_EMBEDDING_INDEXES

class BatchVectorIndex():
    """
    Represents a batch processing wrapper for a VectorIndex to facilitate
    adding embeddings in batches.

    This class is designed to handle batch management and processing of
    embedding nodes to efficiently add them to a VectorIndex. It provides
    functionality to add nodes and write them to the index in predefined
    batch sizes.

    Attributes:
        index_name (str): The name of the underlying vector index.
        index (VectorIndex): The underlying vector index object to which
        embeddings will be added.
        batch_write_size (int): The size of each batch used when writing
        embeddings to the index.
        nodes (list): A list storing nodes to be added to the vector index.
    """
    def __init__(self, idx:VectorIndex, batch_write_size:int):
        """
        Initializes an instance of the class with given index, batch write size, and initializes an empty
        list to store nodes.

        Args:
            idx (VectorIndex): The vector index instance associated with this object.
            batch_write_size (int): The batch size for writing operations.
        """
        self.index_name = idx.index_name
        self.index = idx
        self.batch_write_size = batch_write_size
        self.nodes = []

    def add_embeddings(self, nodes:List):
        """
        Adds embeddings to the current object by extending the existing nodes with the
        given list of nodes.

        Args:
            nodes (List): A list of nodes to be added to the existing nodes.
        """
        self.nodes.extend(nodes)

    def write_embeddings_to_index(self):
        """
        Writes embeddings to the index in batches for improved performance.

        This method processes a list of nodes in smaller chunks defined by the
        batch_write_size attribute and writes embeddings to the index for each
        chunk. The purpose is to handle the processing of potentially large
        datasets without overwhelming memory or computational resources.

        Args:
            None

        Raises:
            None
        """
        node_chunks = [
            self.nodes[x:x+self.batch_write_size] 
            for x in range(0, len(self.nodes), self.batch_write_size)
        ]
        
        for nodes in node_chunks:
            self.index.add_embeddings(nodes)


class VectorBatchClient():
    """Represents a client for managing batch operations in a vector store.

    The `VectorBatchClient` class is designed to streamline batch processing
    operations for a vector store. It allows controlled access to vector indexes,
    enables managing batch writes efficiently, and provides a mechanism for deferring
    node operations until batch updates are applied. This class provides context
    management capabilities for ease of use.

    Attributes:
        indexes (dict): Maps index names (str) to their corresponding
        `BatchVectorIndex` instances, or the base index if batch writes are
        disabled.
        batch_writes_enabled (bool): Indicates whether batch operations for writes
        are enabled.
        all_nodes (list): Stores nodes that are deferred for batch operations.
    """
    def __init__(self, vector_store:VectorStore, batch_writes_enabled:bool, batch_write_size:int):
        """
        Initializes a new instance of the class with the given vector store, batch write settings, and size.

        Args:
            vector_store: The vector store containing all indexes for creating batch vector indexes.
            batch_writes_enabled: A boolean value indicating whether batch writes are enabled.
            batch_write_size: The size of each batch for batch writes.
        """
        self.indexes = {i.index_name: BatchVectorIndex(i, batch_write_size) for i in vector_store.all_indexes()}
        self.batch_writes_enabled = batch_writes_enabled
        self.all_nodes = []

    def get_index(self, index_name):
        """
        Retrieves the index object for the given index name. If the index is not valid,
        an exception is raised. If the index does not exist in the current context,
        it provides a dummy index as a fallback.

        Args:
            index_name: The name of the index to retrieve.

        Returns:
            The index object corresponding to the given index name. Depending on the
            context, this could be either a dummy index or one retrieved from existing
            indexes.

        Raises:
            ValueError: If the provided `index_name` is not among the valid
                `ALL_EMBEDDING_INDEXES`.
        """
        if index_name not in ALL_EMBEDDING_INDEXES:
            raise ValueError(f'Invalid index name ({index_name}): must be one of {ALL_EMBEDDING_INDEXES}')
        if index_name not in self.indexes:
            return DummyVectorIndex(index_name=index_name)
        
        if not self.batch_writes_enabled:          
            return self.indexes[index_name].index
        else:
            return self.indexes[index_name]

    def allow_yield(self, node):
        """
        Determines whether a node should be yielded or added to the internal list based on
        the batch writes configuration.

        If batch writes are enabled, the given node is appended to the internal list of
        nodes and the function returns False, disallowing further yielding. If batch
        writes are not enabled, the function allows yielding by returning True.

        Args:
            node: The node to be evaluated and potentially added to the internal node list
                or allowed for yielding.

        Returns:
            bool: False if batch writes are enabled and the node is added to the internal
            list, otherwise True.
        """
        if self.batch_writes_enabled:
            self.all_nodes.append(node)
            return False
        else:
            return True

    def apply_batch_operations(self):
        """
        Executes batch operations for embedding indexes and returns all nodes.

        This method iterates through the dictionary of indexes and writes embeddings to
        each index. After completing the operations for all indexes, it returns all
        nodes managed by the instance.

        Returns:
            List[Node]: List of all nodes after completing batch operations.
        """
        for index in self.indexes.values():
            index.write_embeddings_to_index()
        return self.all_nodes
    
    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, exception_traceback):
        pass


    