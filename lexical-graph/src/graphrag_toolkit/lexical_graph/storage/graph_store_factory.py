# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Union, Type, Dict

from graphrag_toolkit.lexical_graph.storage.graph import (
    GraphStore,
    GraphStoreFactoryMethod,
)
from graphrag_toolkit.lexical_graph.storage.graph.dummy_graph_store import (
    DummyGraphStoreFactory,
)
from graphrag_toolkit.lexical_graph.storage.graph.neptune_graph_stores import (
    NeptuneAnalyticsGraphStoreFactory,
    NeptuneDatabaseGraphStoreFactory,
)

logger = logging.getLogger(__name__)

GraphStoreType = Union[str, GraphStore]
GraphStoreFactoryMethodType = Union[
    GraphStoreFactoryMethod, Type[GraphStoreFactoryMethod]
]

_graph_store_factories: Dict[str, GraphStoreFactoryMethod] = {
    c.__name__: c()
    for c in [
        NeptuneAnalyticsGraphStoreFactory,
        NeptuneDatabaseGraphStoreFactory,
        DummyGraphStoreFactory,
    ]
}


class GraphStoreFactory:
    """Factory class for registering and creating GraphStore objects.

    Provides methods to register custom factory implementations and retrieve
    instances of GraphStore based on input parameters. Handles dynamic
    factory method registration and graph store instance creation.

    Attributes:
        _graph_store_factories (dict): Internal storage of registered
            factory methods, mapping factory names to their respective
            instances.
    """

    @staticmethod
    def register(factory_type: GraphStoreFactoryMethodType):
        """Registers a factory type instance or class to the internal factory
        registry.

        This method allows registration of classes inheriting from GraphStoreFactoryMethod
        or their instances. The input factory_type is checked for its validity as a subclass
        or instance of GraphStoreFactoryMethod. If this condition is met, the factory
        type is added to the registry with its class name as the key.

        Args:
            factory_type (GraphStoreFactoryMethod | type): The factory type, which can either
                be a subclass of GraphStoreFactoryMethod or an instance of that subclass.

        Raises:
            ValueError: If factory_type is neither a subclass of GraphStoreFactoryMethod
                nor an instance of that class. The error message will indicate the exact
                nature of the issue.
        """
        if isinstance(factory_type, type):
            if not issubclass(factory_type, GraphStoreFactoryMethod):
                raise ValueError(
                    f'Invalid factory_type argument: {factory_type.__name__} must inherit from GraphStoreFactoryMethod.'
                )
            _graph_store_factories[factory_type.__name__] = factory_type()
        else:
            factory_type_name = type(factory_type).__name__
            if not isinstance(factory_type, GraphStoreFactoryMethod):
                raise ValueError(
                    f'Invalid factory_type argument: {factory_type_name} must inherit from GraphStoreFactoryMethod.'
                )
            _graph_store_factories[factory_type_name] = factory_type

    @staticmethod
    def for_graph_store(graph_info: GraphStoreType = None, **kwargs) -> GraphStore:
        """
        Creates or retrieves a graph store instance based on the provided graph
        information or configuration. If an existing `GraphStore` instance is
        given, it is returned directly. Otherwise, iterates through registered
        graph store factories to attempt creating a `GraphStore` instance.
        Raises an error if no suitable factory can process the given graph store
        information.

        :param graph_info: An existing `GraphStore` instance or configuration
            information for the graph store.
        :type graph_info: GraphStoreType
        :param kwargs: Additional keyword arguments passed to factory methods
            when creating a `GraphStore` instance.
        :return: A `GraphStore` instance that matches the provided information.
        :rtype: GraphStore
        :raises ValueError: If the graph store information is invalid or no
            registered factory can process the given configuration.
        """
        if graph_info and isinstance(graph_info, GraphStore):
            return graph_info

        for factory in _graph_store_factories.values():
            graph_store = factory.try_create(graph_info, **kwargs)
            if graph_store:
                return graph_store

        raise ValueError(
            f'Unrecognized graph store info: {graph_info}. Check that the graph store connection info is formatted correctly, and that an appropriate graph store factory method is registered with GraphStoreFactory.'
        )
