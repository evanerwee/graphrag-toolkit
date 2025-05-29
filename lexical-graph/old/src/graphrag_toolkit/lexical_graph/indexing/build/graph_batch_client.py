# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, Any, List, Callable
from graphrag_toolkit.lexical_graph.storage.graph import GraphStore

class GraphBatchClient():
    """
    Manages the graph client and facilitates batch processing of operations, such
    as writes.

    This class is designed to streamline and optimize operations with a graph
    client by enabling batch processing. It allows configuration for batch write
    functionality and maintains data structures for batched operations. It also
    keeps track of all nodes processed. The class is intended to enhance
    efficiency in working with the graph store by aggregating and managing
    operations effectively, including retries for query execution.

    :ivar graph_client: The graph client instance used for interacting with the
        graph store.
    :ivar batch_writes_enabled: Boolean flag indicating whether batch writing is
        enabled or not.
    :ivar batch_write_size: The number of items included in a single batch during
        batch operations.
    :ivar batches: Dictionary to store batched query operations, where keys
        represent query strings and values are lists of parameters.
    :ivar all_nodes: List to track all nodes processed during operations.
    """
    def __init__(self, graph_client:GraphStore, batch_writes_enabled:bool, batch_write_size:int):
        """
        Initializes the instance with the required parameters and sets up attributes for managing
        graph-related operations, potential batch processing, and tracking nodes.

        :param graph_client: The client instance used to interact with the graph database.
        :param batch_writes_enabled: Flag indicating whether batch writing is enabled.
        :param batch_write_size: The size limit for a single batch in batch writing.
        """
        self.graph_client = graph_client
        self.batch_writes_enabled = batch_writes_enabled
        self.batch_write_size = batch_write_size
        self.batches = {}
        self.all_nodes = []

    @property
    def tenant_id(self):
        """
        This property retrieves the tenant identifier associated with the
        graph client instance. The tenant identifier is a unique value that
        is used within the context of a multi-tenant system for identifying
        a specific tenant or organization.

        :return: The tenant identifier linked to the graph client.
        :rtype: str
        """
        return self.graph_client.tenant_id

    def node_id(self, id_name:str):
        """
        Returns the unique identifier for a node based on the provided name.

        This method takes a name identifier and retrieves the unique node identifier
        from the internal graph client. It is useful for associating external entities
        with their corresponding internal node representation.

        :param id_name: Name identifier of the node
        :type id_name: str
        :return: Unique identifier for the node
        :rtype: object
        """
        return self.graph_client.node_id(id_name)
    
    def property_assigment_fn(self, key:str, value:Any) -> Callable[[str], str]:
        """
        Assigns a property to a specified key and value pair, utilizing the provided
        graph client to handle the operation. The function ensures a callable result
        is returned upon execution.

        :param key: The identifier for which the property is being assigned.
        :type key: str
        :param value: The data or information to associate with the given key in
                      the property assignment.
        :type value: Any
        :return: A callable function that takes a string as input and returns a
                 string output after performing the property assignment.
        :rtype: Callable[[str], str
        """
        return self.graph_client.property_assigment_fn(key, value)
    
    def execute_query_with_retry(self, query:str, properties:Dict[str, Any], **kwargs):
        """
        Executes a query with retry logic. If batch writes are enabled, the query and
        its parameters are added to a batch for later execution. Otherwise, the query
        is immediately executed using the configured graph client.

        :param query: The Cypher query to execute.
        :type query: str
        :param properties: A dictionary of properties associated with the query,
                           typically including parameters to bind during the query
                           execution.
        :type properties: Dict[str, Any]
        :param kwargs: Additional optional parameters passed to control the query
                       execution behavior.
        :return: None
        """
        if not self.batch_writes_enabled:
            self.graph_client.execute_query_with_retry(query, properties, **kwargs)
        else:
            if query not in self.batches:
                self.batches[query] = []
            self.batches[query].extend(properties['params'])

    def allow_yield(self, node):
        """
        Determines if the computational process should yield or continue without yielding
        based on the batch writes configuration and the given node.

        :param node: The node being processed in the computational operation.
        :type node: Any
        :return: A boolean value indicating whether the process is allowed to yield
                 (`True`) or not (`False`).
        :rtype: bool
        """
        if self.batch_writes_enabled:
            self.all_nodes.append(node)
            return False
        else:
            return True
        
    def apply_batch_operations(self):
        """
        Apply batch operations on stored queries and parameters. This method processes batched query
        parameters, removes duplicates, divides them into chunks of a predefined size, and executes each
        chunk using the configured graph client. It ensures retries with limited attempts and wait time
        in case of execution failures. Returns all processed nodes.

        :return: All nodes processed after executing the batch operations
        :rtype: list
        """
        for query, parameters in self.batches.items():

            deduped_parameters = self._dedup(parameters)
            parameter_chunks = [
                deduped_parameters[x:x+self.batch_write_size] 
                for x in range(0, len(deduped_parameters), self.batch_write_size)
            ]

            for p in parameter_chunks:
                params = {
                    'params': p
                }
                self.graph_client.execute_query_with_retry(query, params, max_attempts=5, max_wait=7)

        return self.all_nodes
  
    def _dedup(self, parameters:List):
        """
        Removes duplicate parameters from the provided list while preserving case sensitivity
        and retains one unique entry based on a lowercased string representation as a key.

        :param parameters: List of parameters of any type to be deduplicated
        :return: A list containing unique parameters based on case-insensitive comparisons
        """
        params_map = {}
        for p in parameters:
            params_map[str(p).lower()] = p
        return list(params_map.values())
    
    def __enter__(self):
        """
        Manages context entering actions for a class or object. When used in a
        with-statement, this method is automatically invoked to allow the
        class or object to define the set-up behavior for the context.

        :return: Returns the context-managed object.
        :rtype: Same class as the object implementing this method
        """
        return self

    def __exit__(self, exception_type, exception_value, exception_traceback):
        """
        Handles the exit from a runtime context, such as that created by the
        `with` statement. It allows cleanup actions or resource finalization,
        and provides a mechanism to suppress exceptions if necessary.

        :param exception_type: The class of the exception, if an exception was raised.
            Otherwise, `None` if no exception occurred.
        :type exception_type: Optional[Type[BaseException]]
        :param exception_value: The instance of the exception raised, or
            `None` if no exception occurred.
        :type exception_value: Optional[BaseException]
        :param exception_traceback: A traceback object providing details of
            where the exception was raised, or `None` if no exception occurred.
        :type exception_traceback: Optional[TracebackType]
        :return: Indicates whether the exception should be suppressed or propagated.
            `True` suppresses the exception, while `False` propagates it outward.
        :rtype: bool
        """
        pass

