# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import logging.config
import warnings
from typing import List, Dict, Optional, TypeAlias, Union, cast

LoggingLevel: TypeAlias = int


class CompactFormatter(logging.Formatter):
    """
    A custom logging formatter that temporarily modifies log record names to produce
    shortened, customized log messages. This formatter is useful when a concise representation
    of the logger name in log messages is desired, particularly when dealing with deeply nested
    or fully qualified logger names.

    :ivar _style: The logging style used by this formatter.
    :type _style: logging.PercentStyle | logging.StrFormatStyle | logging.StringTemplateStyle
    :ivar _fmt: The formatted log string template used by the formatter.
    :type _fmt: str
    """

    def format(self, record: logging.LogRecord) -> str:
        """
        Formats a log record by temporarily shortening its name, formatting it,
        and then restoring the original name. This function overrides the parent
        formatter's `format` method to achieve custom name handling.

        :param record: The log record to be formatted.
        :type record: logging.LogRecord
        :return: The formatted log message as a string.
        :rtype: str
        """
        original_record_name = record.name
        record.name = self._shorten_record_name(record.name)
        result = super().format(record)
        record.name = original_record_name
        return result

    @staticmethod
    def _shorten_record_name(name: str) -> str:
        """
        Shortens a given fully qualified record name by reducing each segment (except the last)
        to its first letter while keeping the last segment intact.

        This method is helpful in scenarios where a concise representation
        of a long record name is needed by maintaining some level of readability
        and differentiability.

        :param name: The fully qualified record name to shorten. Must be in dotted notation.
        :type name: str
        :return: The shortened record name where all parts except the last are reduced to their
                 first letters.
        :rtype: str
        """
        if '.' not in name:
            return name

        parts = name.split('.')
        return f"{'.'.join(p[0] for p in parts[0:-1])}.{parts[-1]}"


class ModuleFilter(logging.Filter):
    """
    Implements a logging filter to include or exclude log records based on specified
    modules and messages. This class provides fine-grained control over log record
    filtering, allowing for inclusion and exclusion rules to be defined per logging
    level.

    The ModuleFilter class enables tailored logging configurations by allowing
    users to specify which log messages or originating modules should be included
    or excluded at various logging levels. This is particularly useful in scenarios
    where logs need to be filtered dynamically based on severity, source, or content.

    :ivar _included_modules: Mapping of logging levels to the list of modules' names to be included.
        Modules names can be provided as strings or lists of strings. '*' can be used as a wildcard.
    :type _included_modules: dict[LoggingLevel, list[str]]
    :ivar _excluded_modules: Mapping of logging levels to the list of modules' names to be excluded.
        Modules names can be provided as strings or lists of strings. '*' can be used as a wildcard.
    :type _excluded_modules: dict[LoggingLevel, list[str]]
    :ivar _included_messages: Mapping of logging levels to the list of messages to be included.
        Messages can be provided as strings or lists of strings. '*' can be used as a wildcard.
    :type _included_messages: dict[LoggingLevel, list[str]]
    :ivar _excluded_messages: Mapping of logging levels to the list of messages to be excluded.
        Messages can be provided as strings or lists of strings. '*' can be used as a wildcard.
    :type _excluded_messages: dict[LoggingLevel, list[str]]
    """

    def __init__(
        self,
        included_modules: Optional[Dict[LoggingLevel, Union[str, List[str]]]] = None,
        excluded_modules: Optional[Dict[LoggingLevel, Union[str, List[str]]]] = None,
        included_messages: Optional[Dict[LoggingLevel, Union[str, List[str]]]] = None,
        excluded_messages: Optional[Dict[LoggingLevel, Union[str, List[str]]]] = None,
    ) -> None:
        """
        Initializes the logging setting filters by configuring which modules and messages
        to include or exclude for specific logging levels. The class supports specifying
        allowed or restricted modules and messages either as single strings or as lists
        associated with different logging levels.

        The parameters `included_modules`, `excluded_modules`,
        `included_messages`, and `excluded_messages` are optional and allow fine-grained
        control over logging behavior. These will be converted internally into uniform
        dictionary structures associating logging levels to lists of strings for processing.

        :param included_modules: Dictionary mapping logging levels to modules to be
            included. Values can either be a single string representing a module or
            a list of module names.
        :type included_modules: Optional[Dict[LoggingLevel, Union[str, List[str]]]]

        :param excluded_modules: Dictionary mapping logging levels to modules to
            be excluded. Values can either be a single string representing a module
            or a list of module names.
        :type excluded_modules: Optional[Dict[LoggingLevel, Union[str, List[str]]]]

        :param included_messages: Dictionary mapping logging levels to messages to be
            included. Values can be a single message string or a list of message strings.
        :type included_messages: Optional[Dict[LoggingLevel, Union[str, List[str]]]]

        :param excluded_messages: Dictionary mapping logging levels to messages to be
            excluded. Values can be a single message string or a list of message strings.
        :type excluded_messages: Optional[Dict[LoggingLevel, Union[str, List[str]]]]
        """
        super().__init__()
        self._included_modules: dict[LoggingLevel, list[str]] = {
            l: v if isinstance(v, list) else [v]
            for l, v in (included_modules or {}).items()
        }
        self._excluded_modules: dict[LoggingLevel, list[str]] = {
            l: v if isinstance(v, list) else [v]
            for l, v in (excluded_modules or {}).items()
        }
        self._included_messages: dict[LoggingLevel, list[str]] = {
            l: v if isinstance(v, list) else [v]
            for l, v in (included_messages or {}).items()
        }
        self._excluded_messages: dict[LoggingLevel, list[str]] = {
            l: v if isinstance(v, list) else [v]
            for l, v in (excluded_messages or {}).items()
        }

    def filter(self, record: logging.LogRecord) -> bool:
        """
        Filters log records based on their message content and module.

        This method evaluates a log record and determines whether it should be
        allowed or filtered out. It checks the record's message and its originating
        module against predefined inclusion and exclusion criteria that are dependent
        on the log level of the record. The decision-making process includes:

        1. Exclusion of messages starting with specified substrings.
        2. Inclusion of messages starting with specified substrings or matching
           a wildcard ('*').
        3. Exclusion of modules starting with specified substrings or matching
           a wildcard ('*').
        4. Inclusion of modules starting with specified substrings or matching
           a wildcard ('*').

        The inclusion and exclusion criteria are stored in dictionaries categorized
        by log level.

        :param record: The log record to be evaluated.
        :type record: logging.LogRecord
        :return: True if the record passes the filtering criteria,
                 False otherwise.
        :rtype: bool
        """
        record_message = record.getMessage()

        excluded_messages = self._excluded_messages.get(record.levelno, [])
        if any(record_message.startswith(x) for x in excluded_messages):
            return False

        included_messages = self._included_messages.get(record.levelno, [])
        if (
            any(record_message.startswith(x) for x in included_messages)
            or '*' in included_messages
        ):
            return True

        record_module = record.name

        excluded_modules = self._excluded_modules.get(record.levelno, [])
        if (
            any(record_module.startswith(x) for x in excluded_modules)
            or '*' in excluded_modules
        ):
            return False

        included_modules = self._included_modules.get(record.levelno, [])
        if (
            any(record_module.startswith(x) for x in included_modules)
            or '*' in included_modules
        ):
            return True

        return False


BASE_LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'filters': {
        'moduleFilter': {
            '()': ModuleFilter,
            'included_modules': {
                logging.INFO: '*',
                logging.DEBUG: ['graphrag_toolkit'],
                logging.WARNING: '*',
                logging.ERROR: '*',
            },
            'excluded_modules': {
                logging.INFO: ['opensearch', 'boto', 'urllib'],
                logging.DEBUG: ['opensearch', 'boto', 'urllib'],
                logging.WARNING: ['urllib'],
            },
            'excluded_messages': {
                logging.WARNING: ['Removing unpickleable private attribute'],
            },
            'included_messages': {},
        }
    },
    'formatters': {
        'default': {
            '()': CompactFormatter,
            'fmt': '%(asctime)s:%(levelname)s:%(name)-15s:%(message)s',
            'datefmt': '%Y-%m-%d %H:%M:%S',
        }
    },
    'handlers': {
        'stdout': {
            'class': 'logging.StreamHandler',
            'stream': 'ext://sys.stdout',
            'filters': ['moduleFilter'],
            'formatter': 'default',
        },
        'file_handler': {
            'formatter': 'default',
            'class': 'logging.FileHandler',
            'filename': 'output.log',
            'filters': ['moduleFilter'],
            'mode': 'a',
        },
    },
    'loggers': {'': {'handlers': ['stdout'], 'level': logging.INFO}},
}


def set_logging_config(
    logging_level: Union[str, LoggingLevel],
    debug_include_modules: Optional[Union[str, List[str]]] = None,
    debug_exclude_modules: Optional[Union[str, List[str]]] = None,
    filename: Optional[str] = None,
) -> None:
    """
    Configures the logging settings for the application, allowing fine-grained control
    of logging levels and module-specific debug inclusion or exclusion based on the
    provided parameters. This function simplifies advanced logging setup by wrapping
    the internal implementation of `set_advanced_logging_config`.

    :param logging_level: Defines the overall logging level for the application.
        Acceptable values can include standard levels such as `DEBUG`, `INFO`,
        `WARNING`, or custom `LoggingLevel` values.
    :type logging_level: Union[str, LoggingLevel]
    :param debug_include_modules: Specifies the list of modules or a single module
        whose debug-level logs should be included. If the value is None, this
        parameter is ignored.
    :type debug_include_modules: Optional[Union[str, List[str]]]
    :param debug_exclude_modules: Specifies the list of modules or a single module
        whose debug-level logs should be excluded. If the value is None, this
        parameter is ignored.
    :type debug_exclude_modules: Optional[Union[str, List[str]]]
    :param filename: Defines the name of the file where logs will be stored. If None,
        logs are not written to a file.
    :type filename: Optional[str]
    :return: This function does not return any value.
    :rtype: None
    """
    set_advanced_logging_config(
        logging_level,
        included_modules=(
            {logging.DEBUG: debug_include_modules}
            if debug_include_modules is not None
            else None
        ),
        excluded_modules=(
            {logging.DEBUG: debug_exclude_modules}
            if debug_exclude_modules is not None
            else None
        ),
        filename=filename,
    )


def set_advanced_logging_config(
    logging_level: Union[str, LoggingLevel],
    included_modules: Optional[Dict[LoggingLevel, Union[str, List[str]]]] = None,
    excluded_modules: Optional[Dict[LoggingLevel, Union[str, List[str]]]] = None,
    included_messages: Optional[Dict[LoggingLevel, Union[str, List[str]]]] = None,
    excluded_messages: Optional[Dict[LoggingLevel, Union[str, List[str]]]] = None,
    filename: Optional[str] = None,
) -> None:
    """
    Configures advanced logging settings for the application. This function allows
    customization of logging levels, included and excluded modules based on the
    specified logging levels, inclusion and exclusion of specific log messages, and
    output to a file if a filename is provided. It overrides the existing logging
    configuration with the provided parameters.

    :param logging_level: The logging level to be set for the root logger. It supports
        both string values (e.g., "DEBUG", "INFO") and integer levels.
    :param included_modules: A dictionary mapping logging levels to modules to be
        included for logging. Allows specifying single module names or a list of
        module names.
    :param excluded_modules: A dictionary mapping logging levels to modules to be
        excluded from logging. Like included_modules, supports both single and
        multiple module names.
    :param included_messages: A dictionary mapping logging levels to specific log
        messages or patterns that should be included in logging. Can specify single
        messages or a list of messages.
    :param excluded_messages: A dictionary mapping logging levels to specific log
        messages or patterns that should be excluded from logging. Accepts both
        single and multiple messages.
    :param filename: An optional string specifying the file where logs should be
        written. If provided, logs will also be output to the specified file.
    :return: None
    """
    if not _is_valid_logging_level(logging_level):
        warnings.warn(f'Unknown logging level {logging_level!r} provided.', UserWarning)
    if isinstance(logging_level, int):
        logging_level = logging.getLevelName(logging_level)

    config = BASE_LOGGING_CONFIG.copy()
    config['loggers']['']['level'] = logging_level.upper()
    config['filters']['moduleFilter']['included_modules'].update(
        included_modules or dict()
    )
    config['filters']['moduleFilter']['excluded_modules'].update(
        excluded_modules or dict()
    )
    config['filters']['moduleFilter']['included_messages'].update(
        included_messages or dict()
    )
    config['filters']['moduleFilter']['excluded_messages'].update(
        excluded_messages or dict()
    )

    if filename:
        config['handlers']['file_handler']['filename'] = filename
        config['loggers']['']['handlers'].append('file_handler')

    logging.config.dictConfig(config)


def _is_valid_logging_level(level: Union[str, LoggingLevel]) -> bool:
    """
    Determines whether the provided logging level is valid. The level can be given
    as a string (e.g., "DEBUG", "INFO") or as a numeric value corresponding to a
    logging level. It checks the validity of numeric levels by ensuring they exist
    in the internal mapping of logging levels to names and validates string levels
    by comparing their uppercase form with the internal mapping of logging names
    to levels.

    :param level: The logging level to validate. Can be of type str for logging
        level names (e.g., "DEBUG", "INFO") or of type LoggingLevel for numeric
        logging levels.
    :type level: Union[str, LoggingLevel]
    :return: True if the given logging level is valid; otherwise, False.
    :rtype: bool
    """
    if isinstance(level, int):
        return level in cast(dict[LoggingLevel, str], logging._levelToName)  # type: ignore
    elif isinstance(level, str):
        return level.upper() in cast(dict[str, LoggingLevel], logging._nameToLevel)  # type: ignore
    return False
