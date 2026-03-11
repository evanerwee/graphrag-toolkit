# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Callable, Dict, Any, Optional
from dateutil.parser import parse

logger = logging.getLogger(__name__)

DEFAULT_BUILD_FILTER = lambda s: False


def default_source_metadata_formatter(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Formats source metadata by converting date and datetime fields to ISO 8601 format.

    This function processes the input dictionary `metadata` and checks for keys that end
    with '_date' or '_datetime'. If such keys are found, it attempts to parse their
    corresponding values into an ISO 8601 formatted string. If parsing fails, the original
    value is preserved. Other key-value pairs are added to the formatted metadata as is.

    Args:
        metadata (Dict[str, Any]): A dictionary containing metadata, where certain values
        may represent date or datetime information.

    Returns:
        Dict[str, Any]: A dictionary with formatted metadata, where date and datetime
        fields are converted to ISO 8601 format if possible.
    """
    formatted_metadata = {}
    for k, v in metadata.items():
        if k.endswith('_date') or k.endswith('_datetime'):
            try:
                dt = parse(v, fuzzy=False).isoformat()
                formatted_metadata[k] = dt
            except ValueError as e:
                formatted_metadata[k] = v
        else:
            formatted_metadata[k] = v
    return formatted_metadata


class BuildFilter():
    """
    Provides a filtering mechanism for processing topics, statements, and source metadata.

    This class is designed to allow customizable filtering of topics and statements,
    as well as formatting of source metadata. It enables users to specify custom
    functions for filtering and formatting, promoting flexibility and adaptability
    to different use cases.

    Attributes:
        topic_filter_fn (Callable[[str], bool]): Function to filter topics. Default
        filter is `DEFAULT_BUILD_FILTER`.
        statement_filter_fn (Callable[[str], bool]): Function to filter statements.
        Default filter is `DEFAULT_BUILD_FILTER`.
        source_metadata_formatter_fn (Optional[Callable[[Dict[str, Any]], Dict[str, Any]]]):
        Function to format source metadata. Default formatter is
        `default_source_metadata_formatter`.
    """
    def __init__(self,
                 topic_filter_fn: Callable[[str], bool] = None,
                 statement_filter_fn: Callable[[str], bool] = None,
                 source_metadata_formatter_fn: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
                 ):
        """
        Initializes the instance with optional filter functions and a metadata formatter.

        The constructor allows the user to provide custom functions for filtering topics,
        statements, and formatting metadata. Defaults are applied if no custom functions
        are provided.

        Args:
            topic_filter_fn (Callable[[str], bool], optional): A function to filter topics.
            Defaults to a predefined filtering function.
            statement_filter_fn (Callable[[str], bool], optional): A function to filter
            statements. Defaults to a predefined filtering function.
            source_metadata_formatter_fn (Optional[Callable[[Dict[str, Any]], Dict[str, Any]]],
            optional): A function to format source metadata. Defaults to a predefined
            metadata formatting function.
        """
        self.topic_filter_fn = topic_filter_fn or DEFAULT_BUILD_FILTER
        self.statement_filter_fn = statement_filter_fn or DEFAULT_BUILD_FILTER
        self.source_metadata_formatter_fn = source_metadata_formatter_fn or default_source_metadata_formatter

    def ignore_topic(self, topic: str) -> bool:
        """
        Evaluates whether a given topic should be ignored based on a filtering
        function and logs the decision if the topic is ignored.

        Args:
            topic: Topic string to evaluate against the filtering function.

        Returns:
            bool: True if the topic should be ignored, False otherwise.
        """
        result = self.topic_filter_fn(topic)
        if result:
            logger.debug(f'Ignore topic: {topic}')
        return result

    def ignore_statement(self, statement: str) -> bool:
        """
        Determines if a statement should be ignored based on the statement filter function
        provided. This method evaluates the given statement through the filter function
        and logs debug information if the statement is ignored.

        Args:
            statement (str): The statement that needs to be evaluated for ignoring.

        Returns:
            bool: True if the statement should be ignored; otherwise, False.
        """
        result = self.statement_filter_fn(statement)
        if result:
            logger.debug(f'Ignore statement: {statement}')
        return result

    def format_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Formats the provided metadata dictionary using a source metadata formatter function.

        This function applies a formatting operation to a given metadata dictionary, enabling
        modification or transformation of the information contained within it according to the
        defined source metadata formatting logic.

        Args:
            metadata: Dictionary containing metadata to be formatted. The keys and values
                are determined by the specific use case.

        Returns:
            A dictionary with formatted metadata resulting from the application of the source
            metadata formatting function.
        """
        return self.source_metadata_formatter_fn(metadata=metadata)
