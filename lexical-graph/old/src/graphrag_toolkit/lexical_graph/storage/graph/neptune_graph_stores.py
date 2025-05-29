# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import json
import logging
import time
import uuid
from botocore.config import Config
from typing import Optional, Any, Callable
from importlib.metadata import version, PackageNotFoundError
from dateutil.parser import parse

from graphrag_toolkit.lexical_graph.storage.graph import (
    GraphStoreFactoryMethod,
    GraphStore,
    NodeId,
    get_log_formatting,
)
from graphrag_toolkit.lexical_graph.metadata import format_datetime, is_datetime_key
from graphrag_toolkit.lexical_graph import GraphRAGConfig
from llama_index.core.bridge.pydantic import PrivateAttr

NEPTUNE_ANALYTICS = 'neptune-graph://'
NEPTUNE_DATABASE = 'neptune-db://'
NEPTUNE_DB_DNS = 'neptune.amazonaws.com'

logger = logging.getLogger(__name__)


def format_id_for_neptune(id_name: str):
    """
    Formats an identifier string into an instance of `NodeId`, tailoring
    the format based on the input string structure. The input can
    contain a single segment or segments separated by a period,
    and the function processes these segments to construct and
    return a corresponding `NodeId` object.

    :param id_name: The identifier string to format, which can be a
        single word or composed of segments separated by periods.
    :type id_name: str
    :return: A `NodeId` instance where the segments of the `id_name`
        are appropriately structured into the `NodeId` fields.
    :rtype: NodeId
    """
    parts = id_name.split('.')
    if len(parts) == 1:
        return NodeId(parts[0], '`~id`', False)
    else:
        return NodeId(parts[1], f'id({parts[0]})', False)


def create_config(config: Optional[str] = None):
    """
    Creates and returns a configuration object for the system.

    This function attempts to determine the version of the
    `graphrag-toolkit-lexical-graph` package. If the package is not available,
    the version falls back to 'unknown'. An optional configuration string in JSON
    format can be provided, which is parsed and merged with the default
    configuration.

    :param config: A JSON-formatted string containing configuration values
        (optional).
    :type config: Optional[str]
    :return: A populated `Config` object with merged configuration values.
    :rtype: Config
    """
    toolkit_version = 'unknown'

    try:
        toolkit_version = version('graphrag-toolkit-lexical-graph')
    except PackageNotFoundError:
        pass

    config_args = {}
    if config:
        config_args = json.loads(config)
    return Config(
        retries={'total_max_attempts': 1, 'mode': 'standard'},
        read_timeout=600,
        user_agent_appid=f'graphrag-toolkit-lexical-graph-{toolkit_version}',
        **config_args,
    )


def create_property_assigment_fn_for_neptune(
    key: str, value: Any
) -> Callable[[str], str]:
    """
    Creates a property assignment function for Neptune based on the given key and value. The function
    generated will determine how the property is formatted or transformed when assigned,
    depending on whether the key corresponds to a datetime or a generic property.

    :param key: The key name of the property.
    :type key: str
    :param value: The value associated with the property, which may be validated or transformed.
    :type value: Any
    :return: A callable function that accepts a string representing the property identifier
        and returns a formatted string for property assignment.
    :rtype: Callable[[str], str]
    """
    if is_datetime_key(key):
        try:
            format_datetime(value)
            return lambda x: f'datetime({x})'
        except ValueError as e:
            return lambda x: x
    else:
        return lambda x: x


class NeptuneAnalyticsGraphStoreFactory(GraphStoreFactoryMethod):
    """
    Factory class for creating instances of NeptuneAnalyticsClient based on specific graph
    information.

    This class provides a factory method for initializing a graph connection
    to `NeptuneAnalytics`. It evaluates the given graph information and decides whether
    to create and return a `NeptuneAnalyticsClient` instance based on predefined matching
    criteria.

    :ivar NEPTUNE_ANALYTICS: A prefix used for checking whether the provided
        graph information corresponds to a Neptune Analytics graph.
    :type NEPTUNE_ANALYTICS: str
    """

    def try_create(self, graph_info: str, **kwargs) -> GraphStore:
        """
        Attempts to create a `GraphStore` instance based on the provided graph information
        string. If the `graph_info` string starts with the predefined `NEPTUNE_ANALYTICS`
        prefix, this function initializes a `NeptuneAnalyticsClient` using the deriving
        specific details and configuration provided in `kwargs`. Otherwise, it returns None.

        :param graph_info: A string containing the information about the graph. It is
            expected to possibly contain the `NEPTUNE_ANALYTICS` prefix.
        :type graph_info: str
        :param kwargs: Additional keyword arguments that may include optional configuration
            parameters, such as 'config', which defines specific setup details for the
            client.
        :type kwargs: dict
        :return: A `NeptuneAnalyticsClient` instance if the graph information matches
            the `NEPTUNE_ANALYTICS` criteria. Returns None otherwise.
        :rtype: GraphStore or None
        """
        if graph_info.startswith(NEPTUNE_ANALYTICS):

            graph_id = graph_info[len(NEPTUNE_ANALYTICS) :]
            config = kwargs.pop('config', {})

            logger.debug(f'Opening Neptune Analytics graph [graph_id: {graph_id}]')
            return NeptuneAnalyticsClient(
                graph_id=graph_id,
                log_formatting=get_log_formatting(kwargs),
                config=json.dumps(config),
            )
        else:
            return None


class NeptuneDatabaseGraphStoreFactory(GraphStoreFactoryMethod):
    """
    Factory class responsible for creating instances of GraphStore configured specifically
    for Neptune database connections. This class interprets provided graph database
    identifiers or URLs, optionally processes configuration arguments, and initializes
    a Neptune database client if the connection information is valid.

    Designed to handle multiple ways of identifying Neptune database instances
    and allows for flexible customization of connection parameters.

    :ivar graph_endpoint: The resolved or processed endpoint URL of the Neptune database.
    :type graph_endpoint: str or None
    """

    def try_create(self, graph_info: str, **kwargs) -> GraphStore:
        """
        Attempts to create and return a NeptuneDatabaseClient instance based on the given
        graph_info string. Determines the appropriate endpoint and configuration details
        for the Neptune database, and initializes the client if a valid endpoint can
        be identified. If the input graph_info does not match expected patterns or no
        valid endpoint is found, returns None without any instantiation.

        :param graph_info: The string containing information about the graph database,
            including the possible endpoint or DNS details.
        :param kwargs: Keyword arguments that can include additional configurations,
            such as 'endpoint_url', 'port', and 'config'.
        :return: An instance of NeptuneDatabaseClient if a valid database endpoint
            is determined, otherwise None.
        :rtype: GraphStore or None
        """
        graph_endpoint = None

        if graph_info.startswith(NEPTUNE_DATABASE):
            graph_endpoint = graph_info[len(NEPTUNE_DATABASE) :]
        elif graph_info.endswith(NEPTUNE_DB_DNS):
            graph_endpoint = graph_info
        elif NEPTUNE_DB_DNS in graph_info:
            graph_endpoint = graph_info.replace('https://', '')

        if graph_endpoint:
            logger.debug(f'Opening Neptune database [endpoint: {graph_endpoint}]')
            endpoint_url = kwargs.pop('endpoint_url', None)
            port = kwargs.pop('port', 8182)
            if not endpoint_url:
                endpoint_url = (
                    f'https://{graph_endpoint}'
                    if ':' in graph_endpoint
                    else f'https://{graph_endpoint}:{port}'
                )
            config = kwargs.pop('config', {})
            return NeptuneDatabaseClient(
                endpoint_url=endpoint_url,
                log_formatting=get_log_formatting(kwargs),
                config=json.dumps(config),
            )
        else:
            return None


class NeptuneAnalyticsClient(GraphStore):
    """
    Represents a client for interacting with a Neptune Graph database system.

    This class provides an interface for managing and interacting with Neptune-powered
    graph databases. It supports features such as client initialization, node ID
    formatting, property assignment functions, and Cypher query executions, while
    abstracting away implementation-specific details for easier use.

    :ivar graph_id: The unique identifier of the Neptune graph instance.
    :type graph_id: str
    :ivar config: Optional configuration settings for the Neptune graph client.
    :type config: Optional[str]
    :ivar _client: Internal storage for the initialized Neptune Graph API client.
    :type _client: Optional[Any]
    """

    graph_id: str
    config: Optional[str] = None
    _client: Optional[Any] = PrivateAttr(default=None)

    def __getstate__(self):
        self._client = None
        return super().__getstate__()

    @property
    def client(self):
        """
        Retrieves the Neptune graph client. If the client is not previously initialized,
        it creates a new client instance using the defined session and configuration.

        :raises AttributeError: If the session or configuration is improperly defined.

        :return: The initialized or existing client instance.
        :rtype: typing.Any
        """
        if self._client is None:
            session = GraphRAGConfig.session
            self._client = session.client(
                'neptune-graph', config=create_config(self.config)
            )
        return self._client

    def node_id(self, id_name: str) -> NodeId:
        """
        Formats the given node identifier for compatibility with Neptune's expected format.

        :param id_name: The identifier of the node that needs to be formatted.
        :type id_name: str
        :return: A `NodeId` object representing the formatted node identifier.
        :rtype: NodeId
        """
        return format_id_for_neptune(id_name)

    def property_assigment_fn(self, key: str, value: Any) -> Callable[[str], str]:
        """
        Creates a function that assigns a specific property to a Neptune object.

        This function generates a callable that associates a given key-value pair with
        a Neptune property. It facilitates creating dynamic property assignment
        functions for Neptune objects.

        :param key: The key of the property to be assigned.
        :type key: str
        :param value: The value of the property to be assigned.
        :type value: Any
        :return: A callable function that accepts a string input and applies the key-value
                 property assignment for the Neptune object.
        :rtype: Callable[[str], str]
        """
        return create_property_assigment_fn_for_neptune(key, value)

    def execute_query(self, cypher, parameters={}, correlation_id=None):
        """
        Executes a Cypher query against a database and returns the results. The function logs the query
        and execution details, including the time taken to process the query. If debug logging is enabled,
        it logs more detailed information about the response.

        :param cypher: The Cypher query to be executed, as a string.
        :type cypher: str
        :param parameters: A dictionary of parameters to bind to the Cypher query. Defaults to an empty
            dictionary.
        :type parameters: dict, optional
        :param correlation_id: Optional identifier used to correlate the request in logs. If not provided,
            this is set to None.
        :type correlation_id: str, optional
        :return: The results of the query execution, parsed as a JSON object.
        :rtype: dict
        """
        query_id = uuid.uuid4().hex[:5]

        request_log_entry_parameters = self.log_formatting.format_log_entry(
            self._logging_prefix(query_id, correlation_id), cypher, parameters
        )

        logger.debug(
            f'[{request_log_entry_parameters.query_ref}] Query: [query: {request_log_entry_parameters.query}, parameters: {request_log_entry_parameters.parameters}]'
        )

        start = time.time()

        response = self.client.execute_query(
            graphIdentifier=self.graph_id,
            queryString=request_log_entry_parameters.format_query_with_query_ref(
                cypher
            ),
            parameters=parameters,
            language='OPEN_CYPHER',
            planCache='DISABLED',
        )

        end = time.time()

        results = json.loads(response['payload'].read())['results']

        if logger.isEnabledFor(logging.DEBUG):
            response_log_entry_parameters = self.log_formatting.format_log_entry(
                self._logging_prefix(query_id, correlation_id),
                cypher,
                parameters,
                results,
            )
            logger.debug(
                f'[{response_log_entry_parameters.query_ref}] {int((end-start) * 1000)}ms Results: [{response_log_entry_parameters.results}]'
            )

        return results


class NeptuneDatabaseClient(GraphStore):
    """
    A client for interacting with the Neptune Data service using openCypher queries.

    This class provides methods for formatting identifiers, creating property assignment
    functions, and executing openCypher queries. It manages client initialization lazily,
    ensuring efficient access to the Neptune Data service. The class is designed to handle
    dynamic endpoint URLs and configurations, allowing for flexible integration.

    :ivar endpoint_url: The endpoint URL for the `neptunedata` service.
    :type endpoint_url: str
    :ivar config: Optional configuration for the client setup. Defaults to None.
    :type config: Optional[str]
    :ivar _client: The internal client instance for interacting with the `neptunedata` service.
        Initialized lazily when accessed.
    :type _client: Optional[Any]
    """

    endpoint_url: str
    config: Optional[str] = None
    _client: Optional[Any] = PrivateAttr(default=None)

    def __getstate__(self):
        self._client = None
        return super().__getstate__()

    @property
    def client(self):
        """
        Provides the AWS Neptune data client associated with the current object. This property
        initializes the client if it does not already exist, leveraging configuration details
        such as the endpoint URL and session settings.

        :return: The initialized or previously existing AWS Neptune data client
        :rtype: boto3.client
        """
        if self._client is None:
            session = GraphRAGConfig.session
            self._client = session.client(
                'neptunedata',
                endpoint_url=self.endpoint_url,
                config=create_config(self.config),
            )
        return self._client

    def node_id(self, id_name: str) -> NodeId:
        """
        Formats the provided identifier to match the required format for Neptune.

        :param id_name: The identifier that needs to be formatted for Neptune.
        :type id_name: str
        :return: A properly formatted NodeId object, adhering to the format expected
            by Neptune.
        :rtype: NodeId
        """
        return format_id_for_neptune(id_name)

    def property_assigment_fn(self, key: str, value: Any) -> Callable[[str], str]:
        """
        Creates a property assignment function for Neptune API usage. The generated
        function can be used to set properties associated with a Neptune object.

        :param key: A string representing the key of the property to assign.
        :param value: The value of any type to be associated with the provided key.
        :return: A callable function that takes a single string parameter and returns
            a string representing the outcome of the property assignment.
        """
        return create_property_assigment_fn_for_neptune(key, value)

    def execute_query(self, cypher, parameters={}, correlation_id=None):
        """
        Executes a cypher query with the provided parameters and logs the request and response
        details for debugging purposes. It uses a unique query identifier for each execution
        and measures the execution time. The results of the query are returned after execution.

        :param cypher: The cypher query to execute.
        :type cypher: str
        :param parameters: A dictionary of parameters to be used within the cypher query.
            Defaults to an empty dictionary.
        :type parameters: dict, optional
        :param correlation_id: An optional identifier used to correlate log entries
            with specific requests. Defaults to None.
        :type correlation_id: str, optional
        :return: The results of the executed cypher query.
        :rtype: dict
        """
        query_id = uuid.uuid4().hex[:5]

        params = json.dumps(parameters)

        request_log_entry_parameters = self.log_formatting.format_log_entry(
            self._logging_prefix(query_id, correlation_id), cypher, params
        )

        logger.debug(
            f'[{request_log_entry_parameters.query_ref}] Query: [query: {request_log_entry_parameters.query}, parameters: {request_log_entry_parameters.parameters}]'
        )

        start = time.time()

        response = self.client.execute_open_cypher_query(
            openCypherQuery=request_log_entry_parameters.format_query_with_query_ref(
                cypher
            ),
            parameters=params,
        )

        end = time.time()

        results = response['results']

        if logger.isEnabledFor(logging.DEBUG):
            response_log_entry_parameters = self.log_formatting.format_log_entry(
                self._logging_prefix(query_id, correlation_id),
                cypher,
                parameters,
                results,
            )
            logger.debug(
                f'[{response_log_entry_parameters.query_ref}] {int((end-start) * 1000)}ms Results: [{response_log_entry_parameters.results}]'
            )

        return results
