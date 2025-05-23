# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import abc
import uuid
from dataclasses import dataclass
from tenacity import Retrying, stop_after_attempt, wait_random
from tenacity import RetryCallState
from typing import Callable, List, Dict, Any, Optional

from graphrag_toolkit.lexical_graph import TenantId

from llama_index.core.bridge.pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

REDACTED = '**REDACTED**'
NUM_CHARS_IN_DEBUG_RESULTS = 256


def get_log_formatting(args):
    """
    Retrieves and validates the log formatting configuration from the provided arguments.

    This function extracts the `log_formatting` key from the `args` dictionary, if available,
    or defaults to a `RedactedGraphQueryLogFormatting` instance. It ensures that the
    retrieved or default log formatting is of type `GraphQueryLogFormatting`. If the type
    does not comply, an exception is raised.

    :param args: A dictionary of arguments which may contain an override for
        `log_formatting`. This parameter is expected to include the necessary
        configuration for logging purposes.
    :return: The validated or defaulted log formatting value. It is guaranteed to
        be an instance of `GraphQueryLogFormatting`.
    :rtype: GraphQueryLogFormatting
    :raises ValueError: If `log_formatting` is present but not of type
        `GraphQueryLogFormatting`.
    """
    log_formatting = args.pop('log_formatting', RedactedGraphQueryLogFormatting())
    if not isinstance(log_formatting, GraphQueryLogFormatting):
        raise ValueError('log_formatting must be of type GraphQueryLogFormatting')
    return log_formatting


@dataclass
class NodeId:
    """
    Represents an identifier for a node with associated key and value.

    This class is used to define the identity of a node by specifying a key
    and value, along with an optional flag indicating whether it is based
    on a property. It provides a string representation of the node's value.

    :ivar key: The key associated with the node identifier.
    :type key: str
    :ivar value: The value associated with the node identifier.
    :type value: str
    :ivar is_property_based: Indicates whether the identifier is property-based.
                           Defaults to True.
    :type is_property_based: bool
    """

    key: str
    value: str
    is_property_based: bool = True

    def __str__(self):
        return self.value


def format_id(id_name: str):
    """
    Formats an identifier string into a NodeId object.

    The function parses an identifier string, typically containing sections
    separated by a period ('.'). If the identifier consists of only one
    section, it uses the same value for both the identifier and name
    fields of the NodeId object. If the identifier contains multiple
    sections separated by a period, it uses the second section as the
    identifier and the entire input string as the name.

    :param id_name: The input identifier string to format.
    :type id_name: str

    :return: A NodeId object with the parsed identifier and name.
    :rtype: NodeId
    """
    parts = id_name.split('.')
    if len(parts) == 1:
        return NodeId(parts[0], parts[0])
    else:
        return NodeId(parts[1], id_name)


@dataclass
class GraphQueryLogEntryParameters:
    """
    Represents the parameters and related functionality for a graph query log entry.

    This class is a data structure that encapsulates the details of a graph query
    log entry, including the query reference, the query itself, associated
    parameters, and optional results. It also provides a utility method to format
    queries with a query reference prefix.

    :ivar query_ref: Reference identifier for the query.
    :type query_ref: str
    :ivar query: The SQL query string.
    :type query: str
    :ivar parameters: Parameters associated with the query.
    :type parameters: str
    :ivar results: The optional results of the query execution.
    :type results: Optional[str]
    """

    query_ref: str
    query: str
    parameters: str
    results: Optional[str] = None

    def format_query_with_query_ref(self, q):
        """
        Formats a query string by appending a query reference as a comment at the
        beginning of the string.

        :param q: The query string to format.
        :type q: str
        :return: The formatted query string with a query reference prepended.
        :rtype: str
        """
        return f'//query_ref: {self.query_ref}\n{q}'


class GraphQueryLogFormatting(BaseModel):
    """
    This class defines the structure for formatting log entries of graph query executions.

    It serves as an abstract base class, ensuring that subclasses provide a specific
    implementation for formatting query metadata such as query references, parameters,
    and result sets into a structured log entry. The class enables consistent and
    customizable logging of graph database query operations.

    :ivar model_config: Configuration options for the BaseModel class.
    :type model_config: Config
    """

    @abc.abstractmethod
    def format_log_entry(
        self,
        query_ref: str,
        query: str,
        parameters: Dict[str, Any] = {},
        results: Optional[List[Any]] = None,
    ) -> GraphQueryLogEntryParameters:
        """
        Formats a log entry for a graph query operation. This method is intended to
        standardize the format of query logs for consistent logging and debugging
        purposes. It is an abstract method and must be implemented in any subclass,
        providing the specific logic for formatting.

        :param query_ref: Unique reference identifier for the query being logged.
        :Array query_ref:
        :param query: The graph query that was executed.
        :Array query:
        :param parameters: A dictionary of parameters utilized in executing the query,
            defaulting to an empty dictionary.
        :Array parameters:
        :param results: A list containing the query operation's results, optional and
            defaults to None when no results are present.
        :Array Resultado_result.describeSerializableparameters])/May) [:
        :return: An instance of ``GraphQueryLogEntryParameters`` containing the
            formatted log entry data.
        :rtype: GraphQueryLogEntryParameters
        """
        raise NotImplementedError


class RedactedGraphQueryLogFormatting(GraphQueryLogFormatting):
    """
    Handles the formatting of logs related to graph queries by redacting sensitive
    information in the query, parameters, and results.

    This class extends `GraphQueryLogFormatting` and is responsible for creating
    log entries containing redacted details for secure logging purposes. It is
    designed to ensure that the logs remain informative while protecting sensitive
    information.

    :ivar REDACTED: A placeholder indicating that the corresponding information
        has been redacted.
    :type REDACTED: str
    """
    def format_log_entry(
        self,
        query_ref: str,
        query: str,
        parameters: Dict[str, Any] = {},
        results: Optional[List[Any]] = None,
    ) -> GraphQueryLogEntryParameters:
        """
        Formats a log entry containing information about a graph query. The method
        receives data concerning the query reference, the query text itself, parameters
        associated with the query, and query results, then processes them to create a
        log entry. Sensitive information (such as the query text, parameters and
        results) is redacted for security purposes before inclusion in the log entry.

        :param query_ref: Reference ID of the query being logged.
        :type query_ref: str
        :param query: The original query text to be logged.
        :type query: str
        :param parameters: Dictionary containing any parameters used in the query.
                           Defaults to an empty dictionary if no parameters provided.
        :type parameters: Dict[str, Any], optional
        :param results: List of results generated by the query execution.
                        Defaults to None if not provided.
        :type results: Optional[List[Any]]
        :return: An instance of `GraphQueryLogEntryParameters` containing the redacted
                 query, parameters, and results for logging purposes.
        :rtype: GraphQueryLogEntryParameters
        """
        return GraphQueryLogEntryParameters(
            query_ref=query_ref, query=REDACTED, parameters=REDACTED, results=REDACTED
        )


class NonRedactedGraphQueryLogFormatting(GraphQueryLogFormatting):
    """
    Represents a log formatting implementation that formats log entries
    for graph query executions without redacting any information.

    This class extends the functionality of `GraphQueryLogFormatting`
    to provide detailed log entries containing query references,
    query strings, parameters, and results of graph queries. Truncation
    is applied to the results if they exceed a predefined character limit.

    :ivar NUM_CHARS_IN_DEBUG_RESULTS: Defines the maximum number of characters
        allowed for displaying results in a log entry before truncation.
    :type NUM_CHARS_IN_DEBUG_RESULTS: int
    """
    def format_log_entry(
        self,
        query_ref: str,
        query: str,
        parameters: Dict[str, Any] = {},
        results: Optional[List[Any]] = None,
    ) -> GraphQueryLogEntryParameters:
        """
        Formats a log entry for a graph query operation. Converts the query reference, query,
        parameters, and results into a structured log entry of type `GraphQueryLogEntryParameters`.
        If the string representation of the results exceeds a predefined character limit, it is truncated
        with an indication of the truncation.

        :param query_ref: The reference identifier of the query.
        :type query_ref: str
        :param query: The query text to be logged.
        :type query: str
        :param parameters: A dictionary of parameters associated with the query. Defaults to an empty dictionary.
        :type parameters: Dict[str, Any]
        :param results: The results of the query operation, optionally provided. If omitted or None, it is skipped in formatting.
        :type results: Optional[List[Any]]
        :return: A `GraphQueryLogEntryParameters` object containing the formatted log entry.
        :rtype: GraphQueryLogEntryParameters
        """
        results_str = str(results)
        if len(results_str) > NUM_CHARS_IN_DEBUG_RESULTS:
            results_str = f'{results_str[:NUM_CHARS_IN_DEBUG_RESULTS]}... <{len(results_str) - NUM_CHARS_IN_DEBUG_RESULTS} more chars>'
        return GraphQueryLogEntryParameters(
            query_ref=query_ref,
            query=query,
            parameters=str(parameters),
            results=results_str,
        )


def on_retry_query(
    logger: 'logging.Logger',
    log_level: int,
    log_entry_parameters: GraphQueryLogEntryParameters,
    exc_info: bool = False,
) -> Callable[[RetryCallState], None]:
    """
    Logs information about a retried query using the provided logger, suitable
    for integration with retry mechanisms. Outputs details about the retry
    attempt, including timing, prior outcomes, and context-specific metadata.

    :param logger: Logging instance to use for outputting retry information.
    :param log_level: Logging level to use when generating log entries (e.g.,
        INFO, DEBUG).
    :param log_entry_parameters: Instance of `GraphQueryLogEntryParameters`
        storing query-related metadata, such as its reference, query string,
        and parameters.
    :param exc_info: Boolean flag indicating whether to include exception
        details in log entries when the query fails. Defaults to False.
    :return: Callable function that accepts a `RetryCallState` instance
        and logs retry-related information.
    """

    def log_it(retry_state: 'RetryCallState') -> None:
        """
        Logs information about a query retry process, including details about the retry
        reason, the time to the next retry, and relevant query parameters. This is
        intended to be used as a callback in retry logic to provide structured logging
        for debugging or monitoring purposes.

        The function sets up a logging callback that generates a log entry for every retry
        attempt made by the retry logic. The log entry contains details about the query,
        reason for retry, time until the next retry attempt, and other configurable
        parameters.

        :param logger: A logger instance used to log the retry information.
        :type logger: logging.Logger

        :param log_level: The log level set for the retry log entries. This can be any standard
            logging level such as logging.INFO, logging.WARNING, etc.
        :type log_level: int

        :param log_entry_parameters: An object encapsulating details of the query, its reference,
            and any associated parameters. This is used to format the log message.
        :type log_entry_parameters: GraphQueryLogEntryParameters

        :param exc_info: A boolean flag indicating whether exception information should
            be included in the logs if the retry is triggered by a failure.
            Default is False.
        :type exc_info: bool

        :return: A callable function designed to be used as a callback in retry logic.
            The returned callable logs detailed retry information whenever invoked.
        :rtype: Callable[[RetryCallState], None]
        """
        local_exc_info: BaseException | bool | None

        if retry_state.outcome is None:
            raise RuntimeError('log_it() called before outcome was set')

        if retry_state.next_action is None:
            raise RuntimeError('log_it() called before next_action was set')

        if retry_state.outcome.failed:
            ex = retry_state.outcome.exception()
            verb, value = 'raised', f'{ex.__class__.__name__}: {ex}'

            if exc_info:
                local_exc_info = retry_state.outcome.exception()
            else:
                local_exc_info = False
        else:
            verb, value = 'returned', retry_state.outcome.result()
            local_exc_info = False  # exc_info does not apply when no exception

        logger.log(
            log_level,
            f'[{log_entry_parameters.query_ref}] Retrying query in {retry_state.next_action.sleep} seconds because it {verb} {value} [attempt: {retry_state.attempt_number}, query: {log_entry_parameters.query}, parameters: {log_entry_parameters.parameters}]',
            exc_info=local_exc_info,
        )

    return log_it


def on_query_failed(
    logger: 'logging.Logger',
    log_level: int,
    max_attempts: int,
    log_entry_parameters: GraphQueryLogEntryParameters,
) -> Callable[['RetryCallState'], None]:
    """
    Handles logging of query retries and failures after reaching defined arguments,
    like the maximum retry attempts, providing contextual details about the query
    failure and its associated parameters.

    :param logger: The logger instance to output query failure logs.
    :type logger: logging.Logger
    :param log_level: Logging level (e.g., logging.INFO, logging.ERROR).
    :param max_attempts: Defined maximum attempts before considering the operation failed.
    :param log_entry_parameters: A `GraphQueryLogEntryParameters` object encapsulating
        query information, reference, and associated parameters.
    :type log_entry_parameters: GraphQueryLogEntryParameters
    :return: A callable function for internal logging on retry or final query attempt failure.
    :rtype: Callable[['RetryCallState'], None]
    """

    def log_it(retry_state: 'RetryCallState') -> None:
        """
        Logs query failures when the number of retry attempts reaches its maximum. The function
        produces a log entry that includes the query reference, the number of retry attempts, the
        cause of the failure, the query, and its parameters.

        :param logger: A logger instance used to log the message.
        :type logger: logging.Logger
        :param log_level: The severity level at which the log message will be logged.
        :type log_level: int
        :param max_attempts: The maximum number of retry attempts before logging failure.
        :type max_attempts: int
        :param log_entry_parameters: Object containing the query reference, actual query,
            and its parameters for logging.
        :type log_entry_parameters: GraphQueryLogEntryParameters
        :return: A callable function that processes the retry state and logs the relevant
            information when the specified conditions are met.
        :rtype: Callable[[RetryCallState], None]
        """
        if retry_state.attempt_number == max_attempts:
            ex: BaseException | bool | None
            if retry_state.outcome.failed:
                ex = retry_state.outcome.exception()
                verb, value = 'raised', f'{ex.__class__.__name__}: {ex}'
            logger.log(
                log_level,
                f'[{log_entry_parameters.query_ref}] Query failed after {retry_state.attempt_number} retries because it {verb} {value} [query: {log_entry_parameters.query}, parameters: {log_entry_parameters.parameters}]',
                exc_info=ex,
            )

    return log_it


class GraphStore(BaseModel):
    """

    """

    log_formatting: GraphQueryLogFormatting = Field(
        default_factory=lambda: RedactedGraphQueryLogFormatting()
    )
    tenant_id: TenantId = Field(default_factory=lambda: TenantId())

    def execute_query_with_retry(
        self,
        query: str,
        parameters: Dict[str, Any],
        max_attempts=3,
        max_wait=5,
        **kwargs,
    ):
        """
        Executes a SQL query with retry logic based on the provided maximum number of attempts and
        maximum wait time between retries. This method ensures that transient failures while executing
        a SQL query are handled by retrying the operation according to the configured strategy.

        :param query: The SQL query string to be executed.
        :type query: str
        :param parameters: The parameters to be bound to the SQL query execution.
        :type parameters: Dict[str, Any]
        :param max_attempts: The maximum number of attempts to retry the SQL query execution
            if it fails. Default is 3.
        :type max_attempts: int, optional
        :param max_wait: The maximum wait time, in seconds, between retry attempts. Default is 5.
        :type max_wait: int, optional
        :param kwargs: Additional keyword arguments passed to the query execution.
        :type kwargs: dict, optional
        :return: None
        """
        correlation_id = uuid.uuid4().hex[:5]
        if 'correlation_id' in kwargs:
            correlation_id = f'{kwargs["correlation_id"]}/{correlation_id}'
        kwargs['correlation_id'] = correlation_id

        log_entry_parameters = self.log_formatting.format_log_entry(
            f'{correlation_id}/*', query, parameters
        )

        attempt_number = 0
        for attempt in Retrying(
            stop=stop_after_attempt(max_attempts),
            wait=wait_random(min=0, max=max_wait),
            before_sleep=on_retry_query(logger, logging.WARNING, log_entry_parameters),
            after=on_query_failed(
                logger, logging.WARNING, max_attempts, log_entry_parameters
            ),
            reraise=True,
        ):
            with attempt:
                attempt_number += 1
                attempt.retry_state.attempt_number
                self.execute_query(query, parameters, **kwargs)

    def _logging_prefix(self, query_id: str, correlation_id: Optional[str] = None):
        """
        Generates a logging prefix string based on the provided query ID and optional correlation ID.

        This function is used to construct a standardized prefix that can be
        included in log messages, helping to correlate log data with specific
        queries or operations. The prefix includes the correlation ID if provided,
        followed by the query ID, separated by a forward slash. If no correlation
        ID is provided, only the query ID is included in the prefix.

        :param query_id: Identifier for the query.
        :type query_id: str
        :param correlation_id: Optional identifier used for correlating specific queries or operations.
        :type correlation_id: Optional[str]
        :return: A logging prefix string in the format `correlation_id/query_id` or `query_id`
        :rtype: str
        """
        return f'{correlation_id}/{query_id}' if correlation_id else f'{query_id}'

    def node_id(self, id_name: str) -> NodeId:
        """
        Generates a NodeId object based on the given identifier name string.

        This function takes a string id_name and formats it into a NodeId
        object. It ensures the input is processed appropriately to return an
        instance of NodeId, which represents a structured or encoded identifier.

        :param id_name: The identifier name string to be formatted into a
            NodeId object.
        :type id_name: str
        :return: A NodeId object representing the formatted identifier.
        :rtype: NodeId
        """
        return format_id(id_name)

    def property_assigment_fn(self, key: str, value: Any) -> Callable[[str], str]:
        """
        Assigns a value to a key through a lambda function, returning a callable.

        The callable returned by this function takes a string as input and returns
        the same string. This provides a way to structure value assignments in a
        decorative or functional manner while maintaining flexibility in code execution.

        :param key: The key to which the value should be assigned.
        :type key: str
        :param value: The value to associate with the given key.
        :type value: Any
        :return: A callable function that takes a string as an argument and returns it unchanged.
        :rtype: Callable[[str], str]
        """
        return lambda x: x

    @abc.abstractmethod
    def execute_query(
        self, cypher, parameters={}, correlation_id=None
    ) -> Dict[str, Any]:
        """
        Executes a given Cypher query on the database with the provided parameters and
        optional correlation ID. This method is abstract and must be implemented by
        subclasses to specify the behavior for query execution.

        :param cypher: The Cypher query string to be executed.
        :type cypher: str
        :param parameters: A dictionary of parameters to pass into the Cypher query.
                           Defaults to an empty dictionary.
        :type parameters: dict
        :param correlation_id: An optional identifier to correlate logs or tracing
                               information. Defaults to None if not provided.
        :type correlation_id: str, optional
        :return: A dictionary containing the result of the query execution, which may
                 include data fields, metadata, or other relevant information.
        :rtype: Dict[str, Any]
        :raises NotImplementedError: If the method is not implemented by a subclass.
        """
        raise NotImplementedError
