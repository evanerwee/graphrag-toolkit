# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import abc
from typing import List, Any, Generator
from llama_index.core.schema import BaseNode
from llama_index.core.schema import TransformComponent
from llama_index.core.bridge.pydantic import Field


class NodeHandler(TransformComponent):
    """
    Represents a component for processing and filtering nodes. The class serves as
    a base for handling `BaseNode` objects by applying specific logic defined in
    a concrete implementation. It provides an abstract method `accept` that must
    be implemented by subclasses to define the node processing behavior.

    This class includes functionality to process a list of `BaseNode` objects and
    demonstrates flexible handling through additional keyword arguments, enabling
    various processing needs.

    :ivar show_progress: Indicates whether progress should be displayed during
        processing.
    :type show_progress: bool
    """

    show_progress: bool = Field(default=True, description='Whether to show progress.')

    def __call__(self, nodes: List[BaseNode], **kwargs: Any) -> List[BaseNode]:
        """
        Filters and processes a list of nodes using a specified acceptance logic.

        This method leverages the `accept` method to process the input `nodes` list,
        applying the defined criteria or transformation. It returns only those nodes
        that meet the acceptance criteria.

        :param nodes: List of nodes to be processed. Each node should be an instance
            of `BaseNode`, adhering to the required structure and format.
        :param kwargs: Additional keyword arguments that may be passed to the internal
            `accept` method for customized processing.
        :return: A filtered list of nodes that meet the acceptance criteria, as defined
            by the implementation of the `accept` method.
        :rtype: List[BaseNode]
        """
        return [n for n in self.accept(nodes, **kwargs)]

    @abc.abstractmethod
    def accept(
        self, nodes: List[BaseNode], **kwargs: Any
    ) -> Generator[BaseNode, None, None]:
        """
        Processes a list of nodes and applies specific logic as defined by
        the implementing method. This abstract method must be implemented
        by subclasses, defining the exact behavior for processing nodes.
        This method is expected to yield instances of `BaseNode` as it processes.

        :param nodes: A list of `BaseNode` instances to be processed.
        :type nodes: List[BaseNode]
        :param kwargs: Additional keyword arguments to support extensions
            or optional processing behaviors. May vary depending on
            subclass implementation.
        :type kwargs: Any
        :return: A generator yielding processed `BaseNode` instances.
        :rtype: Generator[BaseNode, None, None]
        """
        raise NotImplementedError()
