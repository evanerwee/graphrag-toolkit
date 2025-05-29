# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import List, Dict, Any, Optional, Callable

from graphrag_toolkit.lexical_graph import TenantId
from graphrag_toolkit.lexical_graph.storage.constants import LEXICAL_GRAPH_LABELS
from graphrag_toolkit.lexical_graph.storage.graph import GraphStore, NodeId


class MultiTenantGraphStore(GraphStore):
    """
    Represents a multi-tenant graph store that serves as a wrapper around a `GraphStore`
    to add tenant-specific functionality. This class enables handling of graph data
    segregation for multiple tenants by introducing tenant-specific labels, query rewriting,
    and additional tenant context. It delegates underlying operations to the contained
    `GraphStore` instance while applying tenant-awareness on top of it.

    This class is particularly useful in applications where multiple tenants share the same
    underlying graph database infrastructure but need data isolation and customized operations
    per tenant.

    :ivar inner: The underlying graph store instance being wrapped by this multi-tenant wrapper.
    :type inner: GraphStore
    :ivar labels: List of graph labels that are tenant-specific and used in query rewriting.
    :type labels: List[str]
    """

    @classmethod
    def wrap(
        cls,
        graph_store: GraphStore,
        tenant_id: TenantId,
        labels: List[str] = LEXICAL_GRAPH_LABELS,
    ):
        """
        Wraps a ``GraphStore`` instance into a ``MultiTenantGraphStore`` if necessary. This
        method checks the type of the provided ``graph_store`` and the status of the
        ``tenant_id`` to either return the given ``graph_store`` directly or wrap it
        in a ``MultiTenantGraphStore``.

        :param graph_store: The graph store instance to potentially wrap, which may be a
            subclass of ``GraphStore`` or already a ``MultiTenantGraphStore``.
        :type graph_store: GraphStore
        :param tenant_id: The identifier of the tenant information for the
            graph store. Each tenant is identified uniquely.
        :type tenant_id: TenantId
        :param labels: A list of graph labels used for multi-tenancy context.
            Defaults to ``LEXICAL_GRAPH_LABELS``.
        :type labels: List[str]
        :return: Returns the same instance if it's already a multi-tenant graph store or the
            tenant is default. Otherwise, wraps the existing graph store in a
            ``MultiTenantGraphStore`` and returns it.
        :rtype: Union[GraphStore, MultiTenantGraphStore]
        """
        if tenant_id.is_default_tenant():
            return graph_store
        if isinstance(graph_store, MultiTenantGraphStore):
            return graph_store
        return MultiTenantGraphStore(
            inner=graph_store, tenant_id=tenant_id, labels=labels
        )

    inner: GraphStore
    labels: List[str] = []

    def execute_query_with_retry(
        self,
        query: str,
        parameters: Dict[str, Any],
        max_attempts=3,
        max_wait=5,
        **kwargs,
    ):
        """
        Executes a database query with retry capability. The method attempts to execute the
        provided query using a specified maximum number of attempts and wait time between
        each retry. It also allows additional parameters and arguments to modify the query
        execution behavior. This function internally rewrites the query before execution and
        delegates the retry mechanism to another method.

        :param query: The SQL query string to be executed.
        :param parameters: Mapping of query parameters to their respective values.
        :param max_attempts: The maximum number of retry attempts allowed in case of failure
            (default is 3).
        :param max_wait: The maximum wait time between retry attempts, in seconds (default is 5).
        :param kwargs: Additional arguments to customize query execution as needed.
        :return: None.
        """
        self.inner.execute_query_with_retry(
            query=self._rewrite_query(query),
            parameters=parameters,
            max_attempts=max_attempts,
            max_wait=max_wait,
        )

    def _logging_prefix(self, query_id: str, correlation_id: Optional[str] = None):
        """
        Generates a logging prefix string based on the provided query ID and optional
        correlation ID. The function utilizes an internal method to construct the
        logging prefix for identifying specific queries and their related operations.

        :param query_id: A unique identifier for the query.
        :type query_id: str
        :param correlation_id: An optional identifier used for correlating related queries
            or operations. If not provided, the prefix is generated using only the query_id.
        :type correlation_id: Optional[str]
        :return: A string used as the logging prefix, constructed using the query_id and
            optionally the correlation_id.
        :rtype: str
        """
        return self.inner._logging_prefix(
            query_id=query_id, correlation_id=correlation_id
        )

    def property_assigment_fn(self, key: str, value: Any) -> Callable[[str], str]:
        """
        Assigns a property by using a key-value pair. This function acts as a wrapper
        around an inner method and delegates execution to it.

        :param key: The key identifying the property to be assigned.
        :type key: str
        :param value: The value to be assigned to the given key.
        :type value: Any
        :return: A callable function that accepts a key as a string and returns a string.
        :rtype: Callable[[str], str
        """
        return self.inner.property_assigment_fn(key, value)

    def node_id(self, id_name: str) -> NodeId:
        """
        Retrieves a NodeId corresponding to the provided node name.

        This method utilizes the `node_id` method from an underlying
        implementation to fetch the unique identifier (NodeId) for the
        given node name.

        :param id_name: The name of the node for which the unique identifier
            (NodeId) is required.
        :type id_name: str
        :return: A NodeId instance corresponding to the provided node name.
        :rtype: NodeId
        """
        return self.inner.node_id(id_name=id_name)

    def execute_query(
        self, cypher: str, parameters={}, correlation_id=None
    ) -> Dict[str, Any]:
        """
        Executes a query using the provided cypher string, parameters, and optional correlation ID.
        This method interacts with an underlying reference to execute the final translated query.

        :param cypher: The Cypher query string that specifies the operation to be executed.
        :type cypher: str
        :param parameters: Optional dictionary containing the parameters for the Cypher query.
        :type parameters: dict
        :param correlation_id: Optional identifier to correlate the execution of this query.
        :type correlation_id: Any
        :return: A dictionary containing the result of the query execution.
        :rtype: Dict[str, Any]
        """
        return self.inner.execute_query(
            cypher=self._rewrite_query(cypher),
            parameters=parameters,
            correlation_id=correlation_id,
        )

    def _rewrite_query(self, cypher: str):
        """
        Rewrites a Cypher query to ensure that all labels are formatted according
        to the tenant's identifier. If the default tenant is active, the query
        remains unchanged. Otherwise, it replaces each label in the query format
        with the tenant-specific label.

        :param cypher: The Cypher query string to be rewritten.
        :type cypher: str
        :return: The rewritten Cypher query string with tenant-specific labels, or
            the original query if executed under the default tenant.
        :rtype: str
        """
        if self.tenant_id.is_default_tenant():
            return cypher
        for label in self.labels:
            original_label = f'`{label}`'
            new_label = self.tenant_id.format_label(label)
            cypher = cypher.replace(original_label, new_label)
        return cypher
