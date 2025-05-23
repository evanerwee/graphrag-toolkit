# Copyright FalkorDB.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import time
import uuid
from typing import Optional, Any, List, Union

from llama_index.core.bridge.pydantic import PrivateAttr

from graphrag_toolkit.lexical_graph.storage.graph import GraphStore, NodeId, format_id

logger = logging.getLogger(__name__)

try:
    import falkordb
    from falkordb.node import Node
    from falkordb.edge import Edge
    from falkordb.path import Path
    from falkordb.graph import Graph
    from redis.exceptions import ResponseError, AuthenticationError
except ImportError as e:
    raise ImportError(
        "FalkorDB and/or redis packages not found, install with 'pip install FalkorDB redis'"
    ) from e


DEFAULT_DATABASE_NAME = 'graphrag'
QUERY_RESULT_TYPE = Union[List[List[Node]], List[List[List[Path]]], List[List[Edge]]]

class FalkorDBDatabaseClient(GraphStore):
    """
    Client for interacting with a FalkorDB database.

    Provides methods to connect to a FalkorDB instance, execute queries, and handle authentication.

    :ivar endpoint_url: The URL of the database endpoint.
    :type endpoint_url: str
    :ivar database: The name of the database to connect to.
    :type database: str
    :ivar username: The username for database authentication. Defaults to None.
    :type username: Optional[str]
    :ivar password: The password corresponding to the username. Defaults to None.
    :type password: Optional[str]
    :ivar ssl: Whether SSL should be used for the connection. Defaults to False.
    :type ssl: Optional[bool]
    :ivar _client: Internal attribute to cache the FalkorDB client instance. Defaults to None.
    :type _client: Optional[Any]
    """
    endpoint_url:str
    database:str
    username:Optional[str] = None
    password:Optional[str] = None
    ssl:Optional[bool] = False
        
    _client: Optional[Any] = PrivateAttr(default=None)

    """
    Client for interacting with a FalkorDB database.

    Provides methods to connect to a FalkorDB instance, execute queries, and handle authentication.
    """
    def __init__(self,
                 endpoint_url: str = None,
                 database: str = DEFAULT_DATABASE_NAME,
                 username: str = None,
                 password: str = None,
                 ssl: bool = False,
                 **kwargs
                 ) -> None:
        """
        Initializes the configuration with parameters for connecting to a database or an
        external service. The constructor validates the provided inputs to ensure proper
        values are passed for mandatory fields and other optional settings.

        :param endpoint_url: The URL endpoint to connect to. Default is None.
        :param database: The database name to connect. Must be alphanumeric and not empty.
                         Default is defined by DEFAULT_DATABASE_NAME.
        :param username: The username for authentication. If provided, `password`
                         is required. Default is None.
        :param password: The password for authentication. Required if `username` is provided.
                         Default is None.
        :param ssl: A boolean indicating if SSL should be used for the connection.
                    Default is False.
        :param kwargs: Additional keyword arguments to be passed for extended configurations.

        :raises ValueError: If `username` is provided without a `password`.
        :raises ValueError: If `endpoint_url` is present but is not a string.
        :raises ValueError: If `database` is not alphanumeric or is empty.
        """
        if username and not password:
            raise ValueError("Password is required when username is provided")
        
        if endpoint_url and not isinstance(endpoint_url, str):
            raise ValueError("Endpoint URL must be a string")

        if not database or not database.isalnum():
            raise ValueError("Database name must be alphanumeric and non-empty")

        super().__init__(
            endpoint_url=endpoint_url,
            database=database,
            username=username,
            password=password,
            ssl=ssl,
            **kwargs
        )

    def __getstate__(self):
        """
        Handles the serialization of the instance state by omitting non-serializable
        attributes. This ensures that the instance can be safely pickled and unpickled
        without issues caused by attributes that cannot be serialized, such as
        socket or client connections.

        :return: A dictionary containing the serializable state of the instance.
        :rtype: dict
        """
        self._client = None
        return super().__getstate__()
    
    @property
    def client(self) -> Graph:
        """
        This property provides access to a Graph database client. If the `endpoint_url` is provided, it attempts to
        parse the URL to extract the host and port information. If parsing fails or the format is invalid, it raises
        an error. If no `endpoint_url` is defined, it defaults to using "localhost" and a default port of 6379. It
        establishes a connection to the database using the FalkorDB library, provided the attributes `username`,
        `password`, `ssl`, and `database` are properly configured. If connecting to the database fails due to
        connection, authentication, or unexpected errors, they are logged and re-raised with additional context.

        :raises ValueError: Raised when the `endpoint_url` cannot be parsed or is in an invalid format.
        :raises ConnectionError: Raised for issues during the database connection process including
            connection, authentication, or unexpected errors.
        :return: Returns a FalkorDB `Graph` object representing the selected database connection.
        :rtype: Graph
        """
        if self.endpoint_url:
            try:
                parts = self.endpoint_url.split(':')
                if len(parts) != 2:
                    raise ValueError("Invalid endpoint URL format. Expected format: "
                                     "'falkordb://host:port' or for local use 'falkordb://' ")
                host = parts[0]
                port = int(parts[1])
            except Exception as e:
                raise ValueError(f"Error parsing endpoint url: {e}") from e
        else:
            host = "localhost"
            port = 6379

        if self._client is None:
            try:
                self._client = falkordb.FalkorDB(
                        host=host,
                        port=port,
                        username=self.username,
                        password=self.password,
                        ssl=self.ssl,
                    ).select_graph(self.database)
                
            except ConnectionError as e:
                logger.error(f"Failed to connect to FalkorDB: {e}")
                raise ConnectionError(f"Could not establish connection to FalkorDB: {e}") from e
            except AuthenticationError as e:
                logger.error(f"Authentication failed: {e}")
                raise ConnectionError(f"Authentication failed: {e}") from e
            except Exception as e:
                logger.error(f"Unexpected error while connecting to FalkorDB: {e}")
                raise ConnectionError(f"Unexpected error while connecting to FalkorDB: {e}") from e
        return self._client
        
    
    def node_id(self, id_name: str) -> NodeId:
        """
        Formats the given identifier name into a NodeId.

        The function takes an identifier name as input and formats it into
        a standardized NodeId object to be used within other processes.

        :param id_name: The identifier name to be formatted.
        :type id_name: str

        :return: The formatted NodeId object.
        :rtype: NodeId
        """
        return format_id(id_name)

    def execute_query(self, 
                      cypher: str, 
                      parameters: Optional[dict] = None, 
                      correlation_id: Any = None) -> QUERY_RESULT_TYPE:
        """
        Executes a database query using the provided Cypher query string and parameters. The query is executed via the
        defined client, and logs are generated for the query request and response. If the execution fails due to a
        response error or unexpected error, appropriate exceptions are raised.

        The method also measures and logs the execution time of the query in milliseconds. Query results are returned
        as a list of dictionaries, where each dictionary represents a single result row.

        :param cypher: A Cypher query string that specifies the database operations to execute.
        :param parameters: (Optional) A dictionary containing query parameters to be substituted into the Cypher query.
                           Defaults to None, meaning no parameters are provided.
        :param correlation_id: A unique identifier to correlate this query with other operations, or None if such
                               correlation is not required.
        :return: A list of dictionaries, where each dictionary represents a row of the result set. Each row maps
                 the column names to their corresponding values.
        """
        if parameters is None:
            parameters = {}

        query_id = uuid.uuid4().hex[:5]

        request_log_entry_parameters = self.log_formatting.format_log_entry(
            self._logging_prefix(query_id, correlation_id), 
            cypher, 
            json.dumps(parameters),
        )

        logger.debug(f'[{request_log_entry_parameters.query_ref}] Query: [query: {request_log_entry_parameters.query}, parameters: {request_log_entry_parameters.parameters}]')

        start = time.time()

        try:
            response = self.client.query(
                q=request_log_entry_parameters.format_query_with_query_ref(cypher),
                params=parameters
            )
        except ResponseError as e:
            logger.error(f"Query execution failed: {e}. Query: {cypher}, Parameters: {parameters}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during query execution: {e}. Query: {cypher}, Parameters: {parameters}")
            raise ResponseError(f"Unexpected error during query execution: {e}") from e

        end = time.time()

        results = [{h[1]: d[i] for i, h in enumerate(response.header)} for d in response.result_set]

        if logger.isEnabledFor(logging.DEBUG):
            response_log_entry_parameters = self.log_formatting.format_log_entry(
                self._logging_prefix(query_id, correlation_id), 
                cypher, 
                parameters, 
                results
            )
            logger.debug(f'[{response_log_entry_parameters.query_ref}] {int((end-start) * 1000)}ms Results: [{response_log_entry_parameters.results}]')
        
        return results
