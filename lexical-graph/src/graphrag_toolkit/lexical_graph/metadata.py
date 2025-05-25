# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import abc
from typing import Callable, Any, Dict, List, Optional, Union
from dateutil.parser import parse
from datetime import datetime, date

from graphrag_toolkit.lexical_graph import GraphRAGConfig

from llama_index.core.vector_stores.types import (
    FilterCondition,
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
)
from llama_index.core.bridge.pydantic import BaseModel

logger = logging.getLogger(__name__)

MetadataFiltersType = Union[MetadataFilters, MetadataFilter, List[MetadataFilter]]


def is_datetime_key(key):
    """
    Determines if the provided key ends with any of the suffixes specified
    in the metadata datetime suffixes. This function is typically used
    to validate if a key corresponds to a datetime field based on the
    configuration.

    :param key: A string representing the key to check.
    :type key: str
    :return: A boolean indicating whether the key matches any configured
        metadata datetime suffix.
    :rtype: bool
    """
    return key.endswith(tuple(GraphRAGConfig.metadata_datetime_suffixes))


def format_datetime(s: Any):
    """
    Formats a given input into an ISO 8601 formatted datetime string. The function
    accepts input of type `datetime`, `date`, or a parsable string that represents
    a date and time. If the input is `datetime` or `date`, the ISO 8601 format is
    produced directly from the object. If the input is a string, it parses the string
    to create a corresponding datetime object, which is then formatted to ISO 8601.

    The function is designed to provide a reliable and standard way to convert
    various date/time representations into a uniform string format.

    :param s: The input to be formatted, which can be of type `datetime`, `date`,
        or a string representing a date and time.
    :type s: Any
    :return: A string representing the provided input in ISO 8601 format.
    :rtype: str
    """
    if isinstance(s, datetime) or isinstance(s, date):
        return s.isoformat()
    else:
        return parse(s, fuzzy=False).isoformat()


def type_name_for_key_value(key: str, value: Any) -> str:
    """
    Determines the type name for a given key-value pair based on the value's type and
    other contextual information.

    This function analyzes the type of the provided value and returns a string
    representing the type. If the value is a list, dictionary, or set, an exception
    is raised indicating the unsupported type. For other value types, it maps
    supported types to corresponding string identifiers. Additionally, it applies
    specific logic to determine if a value is a timestamp based on the key and value
    combination.

    :param key: The key of the key-value pair used for determining additional context,
        such as whether the value might correspond to a timestamp.
    :type key: str
    :param value: The value of the key-value pair whose type is to be determined.
    :type value: Any
    :return: A string representing the value type, which can be one of the following:
        - 'int'
        - 'float'
        - 'timestamp'
        - 'text'
    :rtype: str
    :raises ValueError: If the value is of type list, dict, or set, which are unsupported.
    """
    if isinstance(value, list) or isinstance(value, dict) or isinstance(value, set):
        raise ValueError(f'Unsupported value type: {type(value)}')

    if isinstance(value, int):
        return 'int'
    elif isinstance(value, float):
        return 'float'
    else:
        if isinstance(value, datetime) or isinstance(value, date):
            return 'timestamp'
        elif is_datetime_key(key):
            try:
                parse(value, fuzzy=False)
                return 'timestamp'
            except ValueError as e:
                return 'text'
        else:
            return 'text'


def formatter_for_type(type_name: str) -> Callable[[Any], str]:
    """
    Returns a formatter function for the specified type name. The formatter function
    is a callable that processes the input value according to the rules of the
    specified type.

    :param type_name: The name of the type for which a formatter function is
        required. Supported types are 'text', 'timestamp', 'int', and 'float'.
    :raises ValueError: If the given type name is unsupported.
    :return: A callable that formats the input data according to the specified
        type.
    """
    if type_name == 'text':
        return lambda x: x
    elif type_name == 'timestamp':
        return lambda x: format_datetime(x)
    elif type_name == 'int':
        return lambda x: int(x)
    elif type_name == 'float':
        return lambda x: float(x)
    else:
        raise ValueError(f'Unsupported type name: {type_name}')


class SourceMetadataFormatter(BaseModel):
    """
    Provides an abstract base class for formatting source metadata.

    This class is designed to define a common interface for formatting
    metadata related to various sources. Subclasses must implement the
    abstract method `format` to provide custom formatting logic. The
    purpose of this class is to ensure a consistent structure and
    behavior across implementations that handle the transformation
    of source metadata.

    :ivar __abstractmethods__: Set of abstract methods that must
        be implemented by subclasses.
    :type __abstractmethods__: frozenset
    :ivar __validators__: Validators applied to the model's fields.
    :type __validators__: dict
    :ivar __fields_set__: Fields explicitly set during initialization.
    :type __fields_set__: set
    :ivar __config__: Configuration class for model behavior.
    :type __config__: Type[BaseConfig]
    """

    @abc.abstractmethod
    def format(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError()


class DefaultSourceMetadataFormatter(SourceMetadataFormatter):
    """
    DefaultSourceMetadataFormatter provides a default implementation for
    formatting metadata by processing each key-value pair using a formatter
    determined by the type of the values.

    This class inherits from SourceMetadataFormatter and overrides its
    functionality to customize metadata formatting.

    :ivar SomeAttribute: Description of SomeAttribute.
    :type SomeAttribute: type_of_attribute
    """

    def format(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Formats the given metadata dictionary by applying appropriate formatting
        based on the type of each value. The function determines the type name for
        each key-value pair in the metadata, retrieves the relevant formatter for
        each type, and applies the formatter to the value. If formatting fails for
        any reason, the original value is retained in the resulting dictionary.

        :param metadata: Dictionary containing key-value pairs to be formatted.
            Each value is subjected to type-based formatting.
        :type metadata: Dict[str, Any]
        :return: A new dictionary containing formatted values for the given
            metadata, where formatting is applied based on each value's type.
        :rtype: Dict[str, Any]
        """
        formatted_metadata = {}
        for k, v in metadata.items():
            try:
                type_name = type_name_for_key_value(k, v)
                formatter = formatter_for_type(type_name)
                value = formatter(v)
                formatted_metadata[k] = value
            except ValueError as e:
                formatted_metadata[k] = v
        return formatted_metadata


class FilterConfig(BaseModel):
    """
    Represents a configuration for filtering metadata dictionaries. This class is built
    to handle filtering logic determined by source filters and a custom dictionary filter
    function. It is designed to initialize with or without user-provided metadata filters
    and applies those specified filters during operations.

    This class ensures appropriate handling and validation of filter inputs during
    initialization. It applies user-defined filtering logic to metadata dictionaries
    to decide whether they meet the desired criteria.

    :ivar source_filters: Metadata filters applied to configure filtering behavior, which
        defines what criteria the metadata should meet. Can be None if no filters are provided.
    :type source_filters: Optional[MetadataFilters]
    :ivar source_metadata_dictionary_filter_fn: A callable filter function that determines
        whether a metadata dictionary passes the filtering criteria. Used internally to apply
        the filter logic at runtime.
    :type source_metadata_dictionary_filter_fn: Callable[[Dict[str, Any]], bool]
    """

    source_filters: Optional[MetadataFilters]
    source_metadata_dictionary_filter_fn: Callable[[Dict[str, Any]], bool]

    def __init__(self, source_filters: Optional[MetadataFiltersType] = None):
        """
        Initializes the object with optional source filters, which can configure how
        data or metadata is filtered. The inputs are validated and transformed into
        the appropriate format based on the provided type. Raises a ValueError if the
        type of source_filters does not match any recognized types.

        :param source_filters: Optional metadata filters to configure the object's
            filtering behavior. Acceptable types include:

              - MetadataFilters: A collection of metadata filters.
              - MetadataFilter: A single metadata filter.
              - list: A list of `MetadataFilter` objects.

            If no filters are provided, it defaults to None.
        :type source_filters: Optional[MetadataFiltersType]
        """
        if not source_filters:
            source_filters = None
        elif isinstance(source_filters, MetadataFilters):
            source_filters = source_filters
        elif isinstance(source_filters, MetadataFilter):
            source_filters = MetadataFilters(filters=[source_filters])
        elif isinstance(source_filters, list):
            source_filters = MetadataFilters(filters=source_filters)
        else:
            raise ValueError(f'Invalid source filters type: {type(source_filters)}')

        super().__init__(
            source_filters=source_filters,
            source_metadata_dictionary_filter_fn=(
                DictionaryFilter(source_filters) if source_filters else lambda x: True
            ),
        )

    def filter_source_metadata_dictionary(self, d: Dict[str, Any]) -> bool:
        """
        Filters the provided metadata dictionary using a user-defined filter function
        and logs the result. The filtering function is expected to be a callable stored
        in the attribute `source_metadata_dictionary_filter_fn`. The filtering decision
        (True/False) is based on the output of this function for the input metadata.

        :param d: The input metadata dictionary that needs to be filtered.
        :type d: Dict[str, Any]
        :return: A boolean indicating whether the dictionary passes the filter
                 (True if it passes, False otherwise).
        :rtype: bool
        """
        result = self.source_metadata_dictionary_filter_fn(d)
        logger.debug(f'filter result: [{str(d)}: {result}]')
        return result


class DictionaryFilter(BaseModel):
    """
    Filters a dictionary of metadata using specified filter conditions.

    This class is designed to validate and evaluate metadata provided as a dictionary
    against a set of defined rules. The filtering can be based on logical conditions such
    as AND, OR, and NOT. It also supports filter operators like equality, containment,
    text matching, and more, allowing for flexible and recursive filtering mechanisms.

    :ivar metadata_filters: MetadataFilters object defining rules and conditions to filter
        the provided metadata.
    :type metadata_filters: MetadataFilters
    """

    metadata_filters: MetadataFilters

    def __init__(self, metadata_filters: MetadataFilters):
        """
        Initializes an instance of the class with the given metadata filters.

        This constructor initializes the class by setting up the metadata filters
        that determine how the underlying resources or actions are filtered or
        processed. It is crucial for implementing context-specific logic or rules
        applicable to metadata.

        :param metadata_filters: Filters used to process or evaluate metadata.
        :type metadata_filters: MetadataFilters
        """
        super().__init__(metadata_filters=metadata_filters)

    def _apply_filter_operator(
        self, operator: FilterOperator, metadata_value: Any, value: Any
    ) -> bool:
        """
        Evaluates whether a given filter operator, when applied to metadata and a user-provided value,
        results in a match (True) or not (False). This internal function is utilized for filtering
        based on specific criteria dictated by the provided operator and values.

        :param operator: The operator to apply for comparison.
        :type operator: FilterOperator
        :param metadata_value: The value from the metadata to compare.
        :type metadata_value: Any
        :param value: The value to compare against the `metadata_value` using the specified operator.
        :type value: Any
        :return: A boolean indicating whether the applied operator results in a match or not.
        :rtype: bool
        :raises ValueError: If the provided `operator` is unsupported or invalid.
        """
        if metadata_value is None:
            return False
        if operator == FilterOperator.EQ:
            return metadata_value == value
        if operator == FilterOperator.NE:
            return metadata_value != value
        if operator == FilterOperator.GT:
            return metadata_value > value
        if operator == FilterOperator.GTE:
            return metadata_value >= value
        if operator == FilterOperator.LT:
            return metadata_value < value
        if operator == FilterOperator.LTE:
            return metadata_value <= value
        if operator == FilterOperator.IN:
            return metadata_value in value
        if operator == FilterOperator.NIN:
            return metadata_value not in value
        if operator == FilterOperator.CONTAINS:
            return value in metadata_value
        if operator == FilterOperator.TEXT_MATCH:
            return value.lower() in metadata_value.lower()
        if operator == FilterOperator.ALL:
            return all(val in metadata_value for val in value)
        if operator == FilterOperator.ANY:
            return any(val in metadata_value for val in value)
        raise ValueError(f'Unsupported filter operator: {operator}')

    def _apply_metadata_filters_recursive(
        self, metadata_filters: MetadataFilters, metadata: Dict[str, Any]
    ) -> bool:
        """
        Determines the applicability of metadata filters recursively. This method processes
        a collection of metadata filters against a provided metadata dictionary and evaluates
        their conditions based on the logical operations associated with the filters.

        :param metadata_filters: The collection of metadata filter conditions to
            evaluate. This can include nested filters or a combination of filters.
        :type metadata_filters: MetadataFilters
        :param metadata: The dictionary-like metadata structure containing key-value
            pairs to be validated against the filters.
        :type metadata: Dict[str, Any]
        :return: A boolean indicating whether the metadata satisfies the applied filters
            and their conditions.
        :rtype: bool
        """
        results: List[bool] = []

        def get_filter_result(f: MetadataFilter, metadata: Dict[str, Any]):
            """
            Represents a filtering mechanism that applies metadata filters to a dictionary
            of metadata recursively.

            This class provides functionality to evaluate metadata filters against a set
            of metadata values using recursive operations. The filters are applied based
            on specific operators, such as checking if a metadata field is empty or
            evaluating other conditions depending on the operator, metadata type, and
            value.

            :param metadata_filters: A collection of metadata filter rules to be applied
                                     recursively on the metadata.
            :type metadata_filters: MetadataFilters
            :param metadata: A dictionary containing metadata key-value pairs to be
                             evaluated against the provided filters.
            :type metadata: Dict[str, Any]
            :return: True if all specified filters are satisfied by the metadata,
                     False otherwise.
            :rtype: bool
            """
            metadata_value = metadata.get(f.key, None)
            if f.operator == FilterOperator.IS_EMPTY:
                return (
                    metadata_value is None
                    or metadata_value == ''
                    or metadata_value == []
                )
            else:
                type_name = type_name_for_key_value(f.key, f.value)
                formatter = formatter_for_type(type_name)
                value = formatter(f.value)
                metadata_value = formatter(metadata_value)
                return self._apply_filter_operator(
                    operator=f.operator, metadata_value=metadata_value, value=value
                )

        for metadata_filter in metadata_filters.filters:
            if isinstance(metadata_filter, MetadataFilter):
                if metadata_filters.condition == FilterCondition.NOT:
                    raise ValueError(
                        f'Expected MetadataFilters for FilterCondition.NOT, but found MetadataFilter'
                    )
                results.append(get_filter_result(metadata_filter, metadata))
            elif isinstance(metadata_filter, MetadataFilters):
                results.append(
                    self._apply_metadata_filters_recursive(metadata_filter, metadata)
                )
            else:
                raise ValueError(
                    f'Invalid metadata filter type: {type(metadata_filter)}'
                )

        if metadata_filters.condition == FilterCondition.NOT:
            return not all(results)
        elif metadata_filters.condition == FilterCondition.AND:
            return all(results)
        elif metadata_filters.condition == FilterCondition.OR:
            return any(results)
        else:
            raise ValueError(
                f'Unsupported filters condition: {metadata_filters.condition}'
            )

    def __call__(self, metadata: Dict[str, Any]) -> bool:
        """
        Executes the metadata filters recursively and determines if the given metadata satisfies
        the specified filter conditions.

        :param metadata: A dictionary containing metadata key-value pairs to be tested
            against the defined metadata filters.
        :type metadata: Dict[str, Any]

        :return: A boolean indicating whether the metadata passes the filter conditions.
        :rtype: bool
        """
        return self._apply_metadata_filters_recursive(self.metadata_filters, metadata)
