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
    Determines the default scope for the given node.

    This function evaluates the provided node and determines the default scope
    associated with it. The returned scope is typically defined as a constant
    value within the system.

    Args:
        node: The node for which the default scope is being determined. The exact
            format and type of this node depend on the implementation details of
            the system utilizing this function.

    Returns:
        str: A constant string representing the default scope associated with
        the provided node.
    """
    return DEFAULT_SCOPE

class ScopedValueStore(BaseComponent):
    """Manages the scoped storage and retrieval of labeled values.

    This class provides an abstract interface for handling scoped values, identified
    by a label and scope, across different implementations. It facilitates storing
    and retrieving lists of string values within a specific labeled and scoped
    context, suitable for use in modular or scalable systems requiring such
    functionality.

    Attributes:
        None
    """
    @abc.abstractmethod
    def get_scoped_values(self, label:str, scope:str) -> List[str]:
        """
        Defines an abstract method that must be implemented by subclasses to fetch scoped values
        based on a specified label and scope. This method enables retrieving values
        restricted or categorized by a particular context or scope.

        Args:
            label: A string representing the identifier or key for the desired values.
            scope: A string indicating the context or boundary within which
                the values should be retrieved.

        Returns:
            List[str]: A list of strings containing the values matching the provided label
                and restricted by the specified scope.

        Raises:
            NotImplementedError: This method must be implemented by any subclass of the
                abstract class; invoking this method directly without implementation
                will raise this error.
        """
        pass

    @abc.abstractmethod
    def save_scoped_values(self, label:str, scope:str, values:List[str]) -> None:
        """
        Saves values under a given label and scope.

        This method is an abstract method, meant to be implemented by subclasses. It
        allows values to be saved in association with a specific label and scope. The
        implementation should define the behavior for persisting these scoped values.

        Args:
            label (str): The label associated with the values to be saved.
            scope (str): The scope under which the values are categorized.
            values (List[str]): A list of string values to be saved under the given
                label and scope.

        """
        pass

class FixedScopedValueStore(ScopedValueStore):
    """
    A fixed scoped value store for managing and retrieving scoped values.

    This class extends the ScopedValueStore and provides functionality for
    storing and retrieving values associated with specific labels and scopes.
    It stores the values in a fixed format based on scopes and supports operations
    to retrieve or save values within given scopes.

    Attributes:
        scoped_values (Dict[str, List[str]]): A dictionary where the keys are scopes
            (as strings) and the values are lists of strings associated with the
            corresponding scope.
    """
    scoped_values:Dict[str,List[str]] = Field(default={})

    def get_scoped_values(self, label:str, scope:str) -> List[str]:
        """
        Retrieves a list of values corresponding to a specific scope and label.

        This method fetches the stored values associated with the provided scope from the
        internal data structure. If the scope does not exist in the storage, it returns
        an empty list. The label is used to indicate the context or category for the
        retrieved values.

        Args:
            label (str): A descriptive identifier providing context for the retrieval
                operation.
            scope (str): The key within the internal data structure to look up for
                associated values.

        Returns:
            List[str]: A list of strings containing the values associated with the
                specified scope, or an empty list if no match is found.
        """
        return self.scoped_values.get(scope, [])

    def save_scoped_values(self, label:str, scope:str, values:List[str]) -> None:
        """
        Saves a set of values under a specific label and scope for future retrieval.

        This method allows storage of a list of values associated with a specific label
        and scope. It is useful for organizing related data under logical groupings.

        Args:
            label (str): The label under which the values will be stored, representing
                the category or identifier of the data.
            scope (str): The scope representing the context or domain in which the
                values belong, which is utilized for structured organization.
            values (List[str]): A list of string values to be stored, representing
                the actual data entries associated with the label and scope.
        """
        pass

class ScopedValueProvider(BaseComponent):
    """
    Provides a scoped value management system for handling and storing values
    specific to a particular scope, identified using a scope function.

    This class facilitates the management of values that are tied to a specific
    context or scope, determined dynamically through a user-defined function.
    It uses a scoped value store for persisting and retrieving values within the
    defined scope. Additionally, it supports initialization with predefined scoped
    values and provides functionalities for retrieving and updating scoped values.

    Attributes:
        label (str): Label to identify the scoped value provider.
        scope_func (Callable[[BaseNode], str]): Function used to determine the scope
            given an input node.
        scoped_value_store (ScopedValueStore): Storage system for managing scoped
            values.
    """
    label:str = Field(
        description='Scoped value label'
    )

    scope_func:Callable[[BaseNode], str] = Field(
        description='Function for determining scope given an input node'
    )

    scoped_value_store:ScopedValueStore = Field(
        description='Scoped value store'
    )

    @classmethod
    def class_name(cls) -> str:
        """
        Provides a class method to return the name of the class.

        This method is used to retrieve the properly formatted name of the class as
        a string. It is intended to be used in contexts where the class's name is
        required programmatically.

        Returns:
            str: The name of the class, 'ScopedValueProvider'.
        """
        return 'ScopedValueProvider'

    def __init__(self, 
                 label:str,
                 scoped_value_store: ScopedValueStore,
                 scope_func:Callable[[BaseNode], str]=None, 
                 initial_scoped_values: Dict[str, List[str]]={}):
        """
        Represents the initialization of a class instance managing scoped values with an optional scope function and
        predefined initial scoped values.

        The initializer ensures scoped values are stored at the point of creation based on the provided initial scoped
        values. It also delegates initialization of parameters like label, scope function, and the scoped value store
        to its superclass, enabling powerful and customizable scope-based value management.

        Args:
            label: A string used to identify the instance.
            scoped_value_store: An instance of ScopedValueStore, facilitating saving and retrieval of scoped values.
            scope_func: A callable which, given a BaseNode, returns a scope string. Defaults to None, in which case the
                default_scope_fn is used.
            initial_scoped_values: A dictionary that maps scope names to lists of values. These values are saved upon
                initialization.
        """
        for k,v in initial_scoped_values.items():
            scoped_value_store.save_scoped_values(label, k, v)
        
        super().__init__(
            label=label,
            scope_func=scope_func or default_scope_fn,
            scoped_value_store=scoped_value_store
        )

    def get_current_values(self, node:BaseNode) -> Tuple[str, List[str]]:
        """
        Retrieves the current scope and associated values for a specific node.

        This method fetches the scope derived from the given node and retrieves the
        corresponding values stored in the scoped value store. It combines this
        scope and the associated values into a tuple.

        Args:
            node (BaseNode): A node used to determine the scope for retrieving
                associated values.

        Returns:
            Tuple[str, List[str]]: A tuple containing the scope as a string and
                a list of strings representing the current scoped values.
        """
        scope = self.scope_func(node)
        current_values = self.scoped_value_store.get_scoped_values(self.label, scope)
        return (scope, current_values)
    
    def update_values(self, scope:str, old_values:List[str], new_values:List[str]):
        """
        Updates the scoped values in the store by calculating the difference between
        old and new values. If there are any new values to add, they are saved to the
        scoped value store and logged.

        Args:
            scope (str): The scope under which the values will be updated.
            old_values (List[str]): The list of existing values to compare against.
            new_values (List[str]): The list of new values to be added.

        """
        values = list(set(new_values).difference(set(old_values)))
        if values:
            logger.debug(f'Adding scoped values: [label: {self.label}, scope: {scope}, values: {values}]')
            self.scoped_value_store.save_scoped_values(self.label, scope, values)


class FixedScopedValueProvider(ScopedValueProvider):
    """
    Provides a fixed set of scoped values for context-specific operations.

    This class is a specialized implementation of `ScopedValueProvider` that
    supplies pre-defined, fixed scoped values for use in contexts where needed.
    Its purpose is to provide a deterministic and consistent set of scoped values
    that do not change dynamically, ensuring stability and predictability in its
    behavior. Typically used when static scoped values must be provided within
    a defined scope.

    Attributes:
        label (str): A fixed label identifying this provider as '__FIXED__'.
        scoped_value_store (FixedScopedValueStore): The store containing the
            fixed set of scoped values provided.
    """
    def __init__(self, scoped_values: Dict[str, List[str]]={}):
        """
        Initializes an object with a pre-defined label and a fixed scoped value store.

        Args:
            scoped_values (Dict[str, List[str]]): A dictionary where the key is a string representing
                the scope, and the value is a list of strings representing scoped values. This
                parameter is optional and defaults to an empty dictionary.
        """
        super().__init__(
            label='__FIXED__',
            scoped_value_store=FixedScopedValueStore(scoped_values=scoped_values)
        )


