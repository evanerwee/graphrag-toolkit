# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import abc
import logging
from typing import Dict, List, Callable, Tuple

from llama_index.core.schema import BaseComponent, BaseNode
from llama_index.core.bridge.pydantic import Field

DEFAULT_SCOPE = '__ALL__'

logger = logging.getLogger(__name__)


def default_scope_fn(node):
    """
    Determines and returns the default scope for a given node.

    This function takes a node as input and returns the default scope
    associated with it. It uses a predefined constant, DEFAULT_SCOPE,
    to provide a consistent scope value.

    :param node: The input node for which the default scope is being determined.
    :return: A constant representing the default scope of the given node.
    """
    return DEFAULT_SCOPE


class ScopedValueStore(BaseComponent):
    """
    Defines the ScopedValueStore as an abstract base class.

    This class provides method signatures for managing values categorized by
    specific labels and scopes. It serves as a blueprint for subclasses that
    will handle operations like fetching and saving scoped values.

    :ivar scoped_values: A dictionary or similar data structure to store values
        categorized by labels and scopes. It acts as a central repository for
        managing scoped values in derived implementations.
    :type scoped_values: Dict[str, Dict[str, List[str]]]
    :ivar label_validator: A callable or utility function for validating labels
        before performing any operation, ensuring data consistency and integrity.
    :type label_validator: Callable[[str], bool]
    :ivar scope_validator: A callable or utility function for validating scopes
        before performing any operation, ensuring the scope conforms to expected
        formats or rules.
    :type scope_validator: Callable[[str], bool]
    """

    @abc.abstractmethod
    def get_scoped_values(self, label: str, scope: str) -> List[str]:
        """
        Retrieves a list of values associated with a given label, filtered by a specific scope.

        This abstract method is intended to be implemented by subclasses to provide
        custom logic for retrieving scoped values. The returned values are expected to
        represent data relevant to the specified label and scope.

        :param label: The identifier for the group of values to retrieve.
        :type label: str
        :param scope: The scope to filter the values by, such as a specific context
            or condition.
        :type scope: str
        :return: A list of values that correspond to the given label and scope.
        :rtype: List[str]
        """
        pass

    @abc.abstractmethod
    def save_scoped_values(self, label: str, scope: str, values: List[str]) -> None:
        """
        Save a set of scoped values associated with a specific label and scope.
        This abstract method should be implemented to handle the saving
        of specific values within a defined scope, referenced by a unique label.

        :param label: A string representing the unique label to identify the scoped values.
        :type label: str
        :param scope: A string indicating the scope under which the values are categorized.
        :type scope: str
        :param values: A list of strings representing the specific values
            to be saved within the given scope and identified by the label.
        :type values: List[str]
        :return: This method does not return any value.
        :rtype: None
        """
        pass


class FixedScopedValueStore(ScopedValueStore):
    """
    A class for managing scoped values with fixed storage mechanisms.

    This class provides methods for storing and retrieving values associated
    with specific scopes and labels. The values are organized in a way that
    allows context-based categorization. It is designed to handle structured
    data efficiently, enabling easy access and updates within a predefined
    data structure.

    :ivar scoped_values: A dictionary mapping scope keys to lists of string
        values. This is the main internal storage structure used for
        organizing and retrieving data.
    :type scoped_values: Dict[str, List[str]]
    """

    scoped_values: Dict[str, List[str]] = Field(default={})

    def get_scoped_values(self, label: str, scope: str) -> List[str]:
        """
        Retrieve scoped values based on the given label and scope.

        This method fetches a list of values associated with a specified
        scope from the internal `scoped_values` mapping. The `label` parameter
        does not influence the retrieval process but might provide contextual
        information for the lookup.

        :param label: A descriptive label, potentially used to provide
            additional context for the scoped value retrieval. It
            does not directly affect the returned results.
        :param scope: Specifies the scope for which the values are retrieved.
            This is used as a key to query the `scoped_values` mapping.
        :return: A list of values associated with the given scope. If the scope
            key is not found, an empty list is returned.
        :rtype: List[str]
        """
        return self.scoped_values.get(scope, [])

    def save_scoped_values(self, label: str, scope: str, values: List[str]) -> None:
        """
        Saves the provided values associated with a specific label and scope.

        The function is intended to store a collection of `values` related
        to a particular `label` within a specific `scope`. The implementation
        details for the storage mechanism are omitted here.

        :param label: A string that serves as the identifier for categorizing
            the provided `values`.
        :param scope: A string indicating the domain or context within which
            the `label` is applicable.
        :param values: A list of strings representing the values to be saved
            under the given `label` and `scope`.

        :return: None. This function does not return any value.
        """
        pass


class ScopedValueProvider(BaseComponent):
    """
    Represents a scoped value provider for managing scoped values linked to nodes,
    facilitating the retrieval and updating of these values based on defined
    scoping logic.

    This class is designed to manage scoped value mappings by leveraging a provided
    scope function and scoped value store. It supports an initializer for initial
    value setup, value retrieval for specific nodes, and updates to scoped values
    based on scope and changes.

    :ivar label: The identifier for the scoped value provider instance.
    :type label: str
    :ivar scope_func: The function determining scope from an input node.
    :type scope_func: Callable[[BaseNode], str]
    :ivar scoped_value_store: The store managing scoped values.
    :type scoped_value_store: ScopedValueStore
    """

    label: str = Field(description='Scoped value label')

    scope_func: Callable[[BaseNode], str] = Field(
        description='Function for determining scope given an input node'
    )

    scoped_value_store: ScopedValueStore = Field(description='Scoped value store')

    @classmethod
    def class_name(cls) -> str:
        """
        Provides the name of the class as a string.

        This class method is used to return the name of the class in which the
        method is defined. It provides a scoped and consistent way to retrieve
        the class name dynamically at runtime.

        :return: The name of the class as a string.
        :rtype: str
        """
        return 'ScopedValueProvider'

    def __init__(
        self,
        label: str,
        scoped_value_store: ScopedValueStore,
        scope_func: Callable[[BaseNode], str] = None,
        initial_scoped_values: Dict[str, List[str]] = {},
    ):
        """
        Initializes an instance of a custom class with specific scoped value storage and
        initial scoped values.

        This constructor sets up scoped values into a scoped value store associated with
        the given label and initializes the parent class with the provided label, scoped
        value store, and scope function. If a scope function is not provided, a default
        scope function is used.

        :param label: A unique identifier for the instance.
        :param scoped_value_store: An instance of ScopedValueStore that handles the
            storage of scoped attribute values.
        :param scope_func: A callable function used to determine the scope for a
            given BaseNode. If not provided, a default scope function is applied.
        :param initial_scoped_values: A dictionary mapping scope keys to lists of
            string values. These values are added to the scoped value store during
            initialization.
        """
        for k, v in initial_scoped_values.items():
            scoped_value_store.save_scoped_values(label, k, v)

        super().__init__(
            label=label,
            scope_func=scope_func or default_scope_fn,
            scoped_value_store=scoped_value_store,
        )

    def get_current_values(self, node: BaseNode) -> Tuple[str, List[str]]:
        """
        Retrieves the current scoped values for a given node.

        This method determines the appropriate scope for the given node, retrieves
        the list of values associated with the determined scope and label, and
        returns both the scope and the associated values.

        :param node: The node object for which the scoped values are being retrieved.
        :type node: BaseNode

        :return: A tuple containing the scope string and the list of current scoped
                 values associated with the given node and label.
        :rtype: Tuple[str, List[str]]
        """
        scope = self.scope_func(node)
        current_values = self.scoped_value_store.get_scoped_values(self.label, scope)
        return (scope, current_values)

    def update_values(self, scope: str, old_values: List[str], new_values: List[str]):
        """
        Updates the scoped values by comparing the old values with the new values and
        adding only the ones that are different. If there are new values to add, these
        are persisted into the scoped value store.

        :param scope: A string representing the scope under which the values
            are categorized.
        :param old_values: A list of strings representing previously existing
            scoped values.
        :param new_values: A list of strings representing the updated set of
            scoped values.
        :return: None
        """
        values = list(set(new_values).difference(set(old_values)))
        if values:
            logger.debug(
                f'Adding scoped values: [label: {self.label}, scope: {scope}, values: {values}]'
            )
            self.scoped_value_store.save_scoped_values(self.label, scope, values)


class FixedScopedValueProvider(ScopedValueProvider):
    """
    Provides a fixed set of scoped values tied to a specific label. The class
    allows initialization with a pre-defined dictionary of scoped values,
    where each scope is mapped to a list of values. It inherits functionality
    from ScopedValueProvider and is suited for use cases requiring static or
    predefined scoped data.

    :ivar label: A fixed label used to identify this provider.
    :type label: str
    :ivar scoped_value_store: Stores and manages the predefined set of scoped
        values.
    :type scoped_value_store: FixedScopedValueStore
    """

    def __init__(self, scoped_values: Dict[str, List[str]] = {}):
        """
        Initializes the object with a fixed scoped value store and assigns a default label.

        The constructor accepts a dictionary of scoped values and uses it to initialize a
        FixedScopedValueStore. The created scoped value store is then utilized in the parent
        class initialization. Also, a fixed label '__FIXED__' is assigned to the instance.

        :param scoped_values: A dictionary of scoped values where the keys are strings
            representing scope names and the values are lists of strings representing
            the corresponding scoped values.
        :type scoped_values: Dict[str, List[str]]
        """
        super().__init__(
            label='__FIXED__',
            scoped_value_store=FixedScopedValueStore(scoped_values=scoped_values),
        )
