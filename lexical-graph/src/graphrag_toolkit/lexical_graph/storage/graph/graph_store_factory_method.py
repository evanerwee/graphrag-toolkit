# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import abc

from graphrag_toolkit.lexical_graph.storage.graph.graph_store import GraphStore


class GraphStoreFactoryMethod:
    """GraphStoreFactoryMethod provides an abstraction for creating GraphStore
    objects.

    This class defines a factory method pattern that serves as the template for creating
    instances of `GraphStore`. It provides a contract that must be implemented by
    subclasses, specifying how `GraphStore` objects are instantiated based on
    graph configuration details.

    Methods:
        try_create(graph_info, \\*\\*kwargs): Abstract method to attempt the creation
            of a `GraphStore` instance based on provided graph configuration
            and optional parameters.
    """

    @abc.abstractmethod
    def try_create(self, graph_info: str, **kwargs) -> GraphStore:
        """Abstract method for creating a graph store from provided graph
        information.

        This method must be implemented by subclasses to create a graph store
        based on the provided information.

        Attributes:
            graph_info (str): Information or details about the graph structure, which
                will be used for constructing the graph store.
            \\*\\*kwargs: Arbitrary keyword arguments that might be required to configure
                the creation process.
        """
        raise NotImplementedError
