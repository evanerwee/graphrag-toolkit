# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from typing import List
from graphrag_toolkit.lexical_graph import TenantId
from graphrag_toolkit.lexical_graph.storage.vector import VectorStore, VectorIndex


class ReadOnlyVectorStore(VectorStore):
    """
    Represents a read-only wrapper for a VectorStore.

    This class is designed to wrap an existing VectorStore and provide a
    read-only interface to its contents. It ensures that any indexes accessed
    through this wrapper are not writeable. This class can be useful when you
    want to share or use a VectorStore in a context where modifications to it
    are not allowed.

    Attributes:
        inner (VectorStore): The underlying VectorStore being wrapped. All read
            operations are delegated to this inner VectorStore.
    """
    @classmethod
    def wrap(cls, vector_store:VectorStore):
        """
        Wraps the given vector store in a read-only wrapper if it is not already read-only.

        This method ensures that the vector store is encapsulated in a `ReadOnlyVectorStore`
        if it is not yet of that type, providing a read-only interface while preserving the
        original functionality.

        Args:
            vector_store: The vector store instance to wrap.

        Returns:
            ReadOnlyVectorStore: A read-only wrapper around the provided vector store.
            If the input is already a `ReadOnlyVectorStore`, it is returned unchanged.
        """
        if isinstance(vector_store, ReadOnlyVectorStore):
            return vector_store
        return ReadOnlyVectorStore(inner=vector_store)

    inner:VectorStore

    def get_index(self, index_name):
        """
        Retrieves an index by its name, making it non-writeable.

        This method fetches an index from the inner system using the specified
        index name and sets its `writeable` attribute to `False`. The modified
        index is then returned.

        Args:
            index_name: The name of the index to retrieve.

        Returns:
            The requested index with its `writeable` attribute set to `False`.

        Raises:
            KeyError: If the specified index name does not exist in the inner
                system.
        """
        index = self.inner.get_index(index_name=index_name)
        index.writeable = False
        return index
    
    def all_indexes(self) -> List[VectorIndex]:
        """
        Gets all indexes from the inner indexing system.

        This method iterates over the keys of the inner indexing system and retrieves
        the corresponding indexes using the `get_index` method. It consolidates these
        indexes into a list and returns it.

        Returns:
            List[VectorIndex]: A list of all retrieved indexes from the inner indexing
            system.
        """
        return [self.get_index(i) for i in self.inner.indexes.keys()]
        