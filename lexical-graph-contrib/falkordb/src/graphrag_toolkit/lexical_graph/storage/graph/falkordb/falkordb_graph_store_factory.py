import logging
from typing import List, Union
from falkordb.node import Node
from falkordb.edge import Edge
from falkordb.path import Path

from graphrag_toolkit.lexical_graph.storage.graph import GraphStoreFactoryMethod, GraphStore, get_log_formatting

logger = logging.getLogger(__name__)

FALKORDB = 'falkordb://'
FALKORDB_DNS = 'falkordb.com'
DEFAULT_DATABASE_NAME = 'graphrag'
QUERY_RESULT_TYPE = Union[List[List[Node]], List[List[List[Path]]], List[List[Edge]]]

class FalkorDBGraphStoreFactory(GraphStoreFactoryMethod):
    """
    Factory class for creating instances of FalkorDB-based graph storage.

    Provides factory methods to generate FalkorDB graph storage instances, leveraging
    certain identifiers in the `graph_info` to construct valid storage clients. This
    class ensures compatibility with the FalkorDB library and handles cases when the
    required FalkorDB library is unavailable, raising an appropriate ImportError in
    such cases.

    :ivar FALKORDB: Identifier string indicating the start of a valid FalkorDB endpoint.
    :type FALKORDB: str
    :ivar FALKORDB_DNS: Identifier string indicating the end of a FalkorDB DNS-based
        endpoint.
    :type FALKORDB_DNS: str
    """
    def try_create(self, graph_info:str, **kwargs) -> GraphStore:
        """
        Attempts to initialize and return a FalkorDB database client using the provided
        graph information and optional arguments. The method determines the endpoint
        URL based on the provided graph_info and initializes the client accordingly.

        :param graph_info: Graph database connection information. It must either start
                           with the `FALKORDB` prefix or end with `FALKORDB_DNS`.
        :type graph_info: str
        :param kwargs: Optional additional arguments passed to the database client.
        :type kwargs: dict
        :return: A FalkorDB database client instance if the connection information is
                 valid and the required library is available, otherwise None.
        :rtype: GraphStore or None
        :raises ImportError: If the FalkorDB library cannot be imported.

        """
        endpoint_url = None
        if graph_info.startswith(FALKORDB):
            endpoint_url = graph_info[len(FALKORDB):]
        elif graph_info.endswith(FALKORDB_DNS):
            endpoint_url = graph_info
        if endpoint_url:
            try:
                from graphrag_toolkit.lexical_graph.storage.graph.falkordb import FalkorDBDatabaseClient
                logger.debug(f'Opening FalkorDB database [endpoint: {endpoint_url}]')
                return FalkorDBDatabaseClient(
                    endpoint_url=endpoint_url,
                    log_formatting=get_log_formatting(kwargs), 
                    **kwargs
                )
            except ImportError as e:
                raise e
            
        else:
            return None