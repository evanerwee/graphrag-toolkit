# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from typing import List
from graphrag_toolkit.lexical_graph import TenantId
from graphrag_toolkit.lexical_graph.storage.vector import VectorStore, VectorIndex


class MultiTenantVectorStore(VectorStore):
    """Provides a multi-tenant wrapper for VectorStore.

    This class allows creating a wrapper around a `VectorStore` object to
    support multi-tenancy by associating a specific tenant ID with operations.
    It ensures that all indexes retrieved or processed are identified and
    associated with the correct tenant context.

    Attributes:
        inner (VectorStore): The underlying vector store being wrapped.
        tenant_id (TenantId): The tenant ID associated with the operations
            performed on the vector store.
    """
    @classmethod
    def wrap(cls, vector_store:VectorStore, tenant_id:TenantId):
        """
        Wraps the given vector_store with a MultiTenantVectorStore if necessary, based on the
        tenant_id provided. The method ensures that the given vector_store is returned as-is
        if it corresponds to the default tenant or is already an instance of
        MultiTenantVectorStore. Otherwise, it wraps the vector_store inside a
        MultiTenantVectorStore with the given tenant_id.

        Args:
            vector_store: The vector_store to wrap if required.
            tenant_id: The tenant identifier used to decide whether wrapping is necessary.

        Returns:
            The provided vector_store, wrapped in a MultiTenantVectorStore if necessary, or
            the vector_store itself if no wrapping is required.
        """
        
        if isinstance(vector_store, MultiTenantVectorStore):
            return vector_store
        return MultiTenantVectorStore(inner=vector_store, tenant_id=tenant_id)

    inner:VectorStore
    tenant_id:TenantId

    def get_index(self, index_name):
        """
        Retrieves an index from the inner object and associates it with the tenant ID.

        Args:
            index_name: Name of the index to retrieve.

        Returns:
            The index retrieved, with the tenant_id attribute set to the tenant ID.
        """
        index = self.inner.get_index(index_name=index_name)
        index.tenant_id = self.tenant_id
        return index
    
    def all_indexes(self) -> List[VectorIndex]:
        """
        Returns a list of all VectorIndex instances stored in the inner indexes.

        This method iterates through the keys of the `inner.indexes` dictionary and
        retrieves the corresponding VectorIndex object for each key.

        Returns:
            List[VectorIndex]: A list containing all VectorIndex objects within the
            inner indexes of the current instance.
        """
        return [self.get_index(i) for i in self.inner.indexes.keys()]
        