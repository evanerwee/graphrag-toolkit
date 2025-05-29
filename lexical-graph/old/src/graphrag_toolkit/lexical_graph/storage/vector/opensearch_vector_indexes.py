# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import json
import logging
from typing import List, Any, Optional
from dataclasses import dataclass

from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.schema import BaseNode, NodeWithScore, QueryBundle
from llama_index.core.vector_stores.types import (
    VectorStoreQueryResult,
    VectorStoreQueryMode,
    MetadataFilters,
    MetadataFilter,
)
from llama_index.core.indices.utils import embed_nodes
from llama_index.core.vector_stores.types import MetadataFilters

from graphrag_toolkit.lexical_graph.metadata import (
    FilterConfig,
    is_datetime_key,
    format_datetime,
)
from graphrag_toolkit.lexical_graph.config import GraphRAGConfig, EmbeddingType
from graphrag_toolkit.lexical_graph.storage.vector import VectorIndex, to_embedded_query
from graphrag_toolkit.lexical_graph.storage.constants import INDEX_KEY

logger = logging.getLogger(__name__)

try:
    from llama_index.vector_stores.opensearch import OpensearchVectorClient
    from opensearchpy.exceptions import NotFoundError, RequestError
    from opensearchpy import AWSV4SignerAsyncAuth, AsyncHttpConnection
    from opensearchpy import Urllib3AWSV4SignerAuth, Urllib3HttpConnection
    from opensearchpy import OpenSearch, AsyncOpenSearch
except ImportError as e:
    raise ImportError(
        "opensearch-py and/or llama-index-vector-stores-opensearch packages not found, install with 'pip install opensearch-py llama-index-vector-stores-opensearch'"
    ) from e


def _get_opensearch_version(self) -> str:
    """
    Retrieves the OpenSearch version in use.

    The method fetches the OpenSearch version from a hypothetical asynchronous
    client and returns it. The current implementation assumes the version is
    statically defined as '2.0.9'. This method is designed to return a string
    representation of the OpenSearch version.

    :return: The OpenSearch version string.
    :rtype: str
    """
    # info = asyncio_run(self._os_async_client.info())
    return '2.0.9'


import llama_index.vector_stores.opensearch

llama_index.vector_stores.opensearch.OpensearchVectorClient._get_opensearch_version = (
    _get_opensearch_version
)


@dataclass
class DummyAuth:
    """
    Represents a dummy authentication class.

    This class serves as a basic example or placeholder for authentication logic
    and functionality. It is not intended for production use and may require
    modification based on specific authentication needs.

    :ivar service: The name of the service requiring authentication.
    :type service: str
    """

    service: str


def create_os_client(endpoint, **kwargs):
    """
    Creates and returns an OpenSearch client. The client is configured to use AWS
    Signature Version 4 for authentication, and it includes specific settings
    for SSL/TLS, connection retries, and timeouts. Configuration is based on
    the session and region provided by the `GraphRAGConfig`.

    :param endpoint: The Amazon OpenSearch Service endpoint to connect to.
    :type endpoint: str
    :param kwargs: Additional keyword arguments that will be passed directly
        to the OpenSearch client.
    :return: An OpenSearch client configured with AWS SigV4 authentication
        and additional connection parameters.
    :rtype: OpenSearch
    """
    session = GraphRAGConfig.session
    region = GraphRAGConfig.aws_region
    credentials = session.get_credentials()
    service = 'aoss'

    auth = Urllib3AWSV4SignerAuth(credentials, region, service)

    return OpenSearch(
        hosts=[endpoint],
        http_auth=auth,
        use_ssl=True,
        verify_certs=True,
        connection_class=Urllib3HttpConnection,
        timeout=300,
        max_retries=10,
        retry_on_timeout=True,
        **kwargs,
    )


def create_os_async_client(endpoint, **kwargs):
    """
    Creates an asynchronous OpenSearch client configured with the provided
    endpoint, AWS credentials, and additional optional parameters. The client
    utilizes AWS Signature Version 4 for authentication and is pre-configured
    to handle connections securely with SSL and certificate verification.

    :param endpoint: The endpoint URL for the OpenSearch service.
    :type endpoint: str
    :param kwargs: Additional optional parameters to customize the OpenSearch
        client configuration, such as custom timeouts, connection settings, etc.
    :return: An AsyncOpenSearch object initialized with the specified
        configurations.
    :rtype: AsyncOpenSearch
    """
    session = GraphRAGConfig.session
    region = GraphRAGConfig.aws_region
    credentials = session.get_credentials()
    service = 'aoss'

    auth = AWSV4SignerAsyncAuth(credentials, region, service)

    return AsyncOpenSearch(
        hosts=[endpoint],
        http_auth=auth,
        use_ssl=True,
        verify_certs=True,
        connection_class=AsyncHttpConnection,
        timeout=300,
        max_retries=10,
        retry_on_timeout=True,
        **kwargs,
    )


def index_exists(endpoint, index_name, dimensions, writeable) -> bool:
    """
    Checks if the specified OpenSearch index exists. If it does not exist and the `writeable`
    parameter is True, it creates the index with the given configuration.

    The function uses the OpenSearch client to check for the existence of the index. If the
    index does not exist and is required to be created, it configures the index with specific
    settings optimized for k-nearest neighbor (k-NN) search, including defining the embedding
    field and the required settings for the search method. The function ensures proper cleanup
    of the OpenSearch client regardless of errors or success.

    :param endpoint: The OpenSearch cluster endpoint.
    :type endpoint: str
    :param index_name: The name of the index to check or create.
    :type index_name: str
    :param dimensions: The dimensionality of the embedding vector for the index configuration.
    :type dimensions: int
    :param writeable: Whether to create the index if it does not exist.
    :type writeable: bool
    :return: True if the index exists or was created successfully, otherwise False.
    :rtype: bool
    """
    client = create_os_client(endpoint, pool_maxsize=1)

    embedding_field = 'embedding'
    method = {
        "name": "hnsw",
        "space_type": "l2",
        "engine": "nmslib",
        "parameters": {"ef_construction": 256, "m": 48},
    }

    idx_conf = {
        "settings": {"index": {"knn": True, "knn.algo_param.ef_search": 100}},
        "mappings": {
            "properties": {
                embedding_field: {
                    "type": "knn_vector",
                    "dimension": dimensions,
                    "method": method,
                },
            }
        },
    }

    index_exists = False

    try:
        index_exists = client.indices.exists(index_name)
        if not index_exists and writeable:
            logger.debug(
                f'Creating OpenSearch index [index_name: {index_name}, endpoint: {endpoint}]'
            )
            client.indices.create(index=index_name, body=idx_conf)
            index_exists = True
    except RequestError as e:
        if e.error == 'resource_already_exists_exception':
            pass
        else:
            logger.exception('Error creating an OpenSearch index')
    finally:
        client.close()

    return index_exists


def create_opensearch_vector_client(endpoint, index_name, dimensions, embed_model):
    """
    Creates and initializes an OpenSearch vector client with specified configurations.

    The function attempts to create an instance of the OpensearchVectorClient. If a
    connection attempt fails due to an error, the function retries up to three times before
    raising the exception. It configures the client with specific text and embedding fields,
    a synchronous HTTP client, and an asynchronous HTTP client for communication with the
    OpenSearch service.

    :param endpoint: The endpoint URL of the OpenSearch service.
    :type endpoint: str
    :param index_name: The name of the index to use in OpenSearch.
    :type index_name: str
    :param dimensions: The number of dimensions for vector embeddings.
    :type dimensions: int
    :param embed_model: The name or type of the embedding model to use.
    :type embed_model: str
    :return: An instance of OpensearchVectorClient, configured and connected.
    :rtype: OpensearchVectorClient
    :raises NotFoundError: If the OpenSearch vector client could not be created within the
        retry limit due to errors such as missing configurations or invalid endpoint.
    """
    text_field = 'value'
    embedding_field = 'embedding'

    logger.debug(
        f'Creating OpenSearch vector client [endpoint: {endpoint}, index_name={index_name}, embed_model={embed_model}, dimensions={dimensions}]'
    )

    client = None
    retry_count = 0
    while not client:
        try:
            client = OpensearchVectorClient(
                endpoint,
                index_name,
                dimensions,
                embedding_field=embedding_field,
                text_field=text_field,
                os_client=create_os_client(endpoint),
                os_async_client=create_os_async_client(endpoint),
                http_auth=DummyAuth(service='aoss'),
            )
        except NotFoundError as err:
            retry_count += 1
            logger.warning(
                f'Error while creating OpenSearch vector client [retry_count: {retry_count}, error: {err}]'
            )
            if retry_count > 3:
                raise err

    logger.debug(
        f'Created OpenSearch vector client [client: {client}, retry_count: {retry_count}]'
    )

    return client


class DummyOpensearchVectorClient:
    """Represents a dummy client for handling vector-based search operations
    integrated with an OpenSearch backend.

    This class provides a simulated implementation for indexing and querying
    vector-based data within a datastore. It is designed for testing or
    placeholder purposes and does not connect to an actual OpenSearch instance.
    The primary functionalities include indexing data and executing queries
    against a vector-based data store, returning simulated results. The class
    can be extended or integrated with actual OpenSearch APIs for full
    functionality.

    :ivar _os_async_client: An internal attribute representing the backend
        asynchronous OpenSearch client. For the dummy implementation, this is
        initialized to ``None``.
    :type _os_async_client: Any
    """

    def __init__(self):
        self._os_async_client = None

    def index_results(self, nodes: List[BaseNode], **kwargs: Any) -> List[str]:
        """
        Indexes the given nodes and returns a list of identifiers for the indexed nodes.

        This method processes a list of nodes and, optionally, additional keyword arguments to
        index the nodes. The function will return a list of identifiers corresponding to the
        successfully indexed nodes. Each node is expected to follow the structure defined by
        the `BaseNode` type.

        :param nodes: A list of nodes of type `BaseNode` that are to be indexed.
        :type nodes: List[BaseNode]
        :param kwargs: Additional keyword arguments that may be used for indexing.
        :type kwargs: Any
        :return: A list of strings representing identifiers for the indexed nodes.
        :rtype: List[str]
        """
        return []

    def query(
        self,
        query_mode: VectorStoreQueryMode,
        query_str: Optional[str],
        query_embedding: List[float],
        k: int,
        filters: Optional[MetadataFilters] = None,
    ) -> VectorStoreQueryResult:
        """
        Execute a query in the vector store using a specified query mode. The query can be conducted
        either with a query string or a query embedding. Optionally, additional filters can be applied
        to refine the query results. The method retrieves a specified number of results, sorted
        based on similarity.

        :param query_mode: The mode in which the query will be processed. This determines how
            the query is interpreted and executed within the vector store.
        :param query_str: The textual query string used for searching, if applicable.
        :param query_embedding: A list of floats representing the query embedding.
        :param k: The number of top results to retrieve based on similarity.
        :param filters: Additional metadata filters applied to refine the search query.
        :return: A `VectorStoreQueryResult` instance containing the nodes, their corresponding
            similarities, and IDs returned from the query.
        """
        return VectorStoreQueryResult(nodes=[], similarities=[], ids=[])


class OpenSearchIndex(VectorIndex):
    """
    Represents an OpenSearch index used for vector-based search and data storage.

    This class provides methods and properties to interact with an OpenSearch index
    designed for handling vector embeddings. It includes functionalities such as
    embedding addition, result conversion, and metadata filter updates. The
    OpenSearchIndex class enables seamless management of vector data, allowing for
    efficient search and retrieval operations. It also incorporates support for
    custom embedding models and dimensional configurations.

    :ivar endpoint: The URL endpoint for the OpenSearch instance.
    :type endpoint: str
    :ivar index_name: The name of the OpenSearch index.
    :type index_name: str
    :ivar dimensions: The number of dimensions for vector embeddings in the index.
    :type dimensions: int
    :ivar embed_model: The embedding model used for generating embeddings.
    :type embed_model: EmbeddingType
    :ivar _client: An instance of the OpenSearch vector client, either initialized
        dynamically or set to None by default. This is considered a private
        attribute and is excluded from serialization.
    :type _client: OpensearchVectorClient
    """

    @staticmethod
    def for_index(index_name, endpoint, embed_model=None, dimensions=None):
        """
        Creates and returns an instance of the OpenSearchIndex class configured for a
        specific index. This method derives default values for certain parameters
        from the GraphRAGConfig if they are not explicitly provided.

        :param index_name: The name of the OpenSearch index.
        :type index_name: str
        :param endpoint: The endpoint URL for the OpenSearch service.
        :type endpoint: str
        :param embed_model: Optional embedding model to use for the index configuration.
        :type embed_model: Optional[str]
        :param dimensions: Optional dimensionality for the embedded vectors.
        :type dimensions: Optional[int]
        :return: An instance of OpenSearchIndex that is configured with the specified
            or derived settings.
        :rtype: OpenSearchIndex
        """
        embed_model = embed_model or GraphRAGConfig.embed_model
        dimensions = dimensions or GraphRAGConfig.embed_dimensions

        return OpenSearchIndex(
            index_name=index_name,
            endpoint=endpoint,
            dimensions=dimensions,
            embed_model=embed_model,
        )

    class Config:
        """
        Config class for specifying configuration options.

        This class provides settings that can be used to configure how objects
        behave within a certain context. The `arbitrary_types_allowed` attribute
        determines whether arbitrary types can be allowed in specific settings.

        :ivar arbitrary_types_allowed: Indicates if arbitrary types are permitted.
        :type arbitrary_types_allowed: bool
        """

        arbitrary_types_allowed = True

    endpoint: str
    index_name: str
    dimensions: int
    embed_model: EmbeddingType

    _client: OpensearchVectorClient = PrivateAttr(default=None)

    def __getstate__(self):
        """
        Returns the internal state of the current object instance for the purpose
        of serialization. This method modifies the state by setting the `_client`
        attribute to `None`, ensuring that the `client` reference is not serialized.

        :return: The picklable state of the object.
        :rtype: dict
        """
        self._client = None
        return super().__getstate__()

    @property
    def client(self) -> OpensearchVectorClient:
        """
        Retrieves or initializes the OpenSearch client for vector operations. If the client
        is not already initialized, this property dynamically checks whether the required
        index exists on the specified endpoint with the given parameters. If the index exists,
        it creates a new client for OpenSearch vector operations. If the index does not exist,
        a dummy client is initialized for compatibility.

        :raises None: The method does not explicitly raise exceptions.
        :return: Returns an instance of `OpensearchVectorClient` or `DummyOpensearchVectorClient`
                 depending on the outcome of the index check.
        :rtype: OpensearchVectorClient
        """
        if not self._client:
            if index_exists(
                self.endpoint,
                self.underlying_index_name(),
                self.dimensions,
                self.writeable,
            ):
                self._client = create_opensearch_vector_client(
                    self.endpoint,
                    self.underlying_index_name(),
                    self.dimensions,
                    self.embed_model,
                )
            else:
                self._client = DummyOpensearchVectorClient()
        return self._client

    def _clean_id(self, s):
        """
        Generates a cleaned version of a given string by removing all non-alphanumeric
        characters. The function iterates through the input string and retains only
        alphanumeric characters.

        :param s: The input string to be cleaned
        :type s: str
        :return: A string that contains only alphanumeric characters from the input
        :rtype: str
        """
        return ''.join(c for c in s if c.isalnum())

    def _to_top_k_result(self, r):
        """
        Converts the given result object into a top-k result dictionary format. The
        transformation involves extracting pertinent data from the metadata of the
        result object and structuring it into a dictionary. If specific keys such as
        INDEX_KEY are present in the metadata, their associated values are added
        to the resulting dictionary. Otherwise, all metadata key-value pairs are
        included in the result.

        :param r: Result object containing a score and metadata.
        :type r: Result
        :return: A dictionary representing the transformed top-k result, including
                 score and relevant metadata.
        :rtype: dict
        """
        result = {'score': r.score}

        if INDEX_KEY in r.metadata:
            index_name = r.metadata[INDEX_KEY]['index']
            result[index_name] = r.metadata[index_name]
            if 'source' in r.metadata:
                result['source'] = r.metadata['source']
        else:
            for k, v in r.metadata.items():
                result[k] = v

        return result

    def _to_get_embedding_result(self, hit):
        """
        Processes a given hit data object to extract and format embedding result information.

        The function extracts specific fields from the `_source` attribute of the input
        hit object, which represents a database record. It parses the 'metadata' field as
        JSON and formats the result with key fields such as 'id', 'value', 'embedding',
        and additional metadata. Fields with the key `INDEX_KEY` in metadata are skipped.

        :param hit: A dictionary representing a database record that contains `_source`
            with fields like 'metadata', 'id', 'value', and 'embedding'.
        :type hit: dict

        :return: A dictionary containing the formatted result data for the embedding,
            including 'id', 'value', 'embedding', and additional metadata fields from
            the original record, excluding fields with the key `INDEX_KEY`.
        :rtype: dict
        """
        source = hit['_source']
        data = json.loads(source['metadata']['_node_content'])

        result = {
            'id': source['id'],
            'value': source['value'],
            'embedding': source['embedding'],
        }

        for k, v in data['metadata'].items():
            if k != INDEX_KEY:
                result[k] = v

        return result

    def add_embeddings(self, nodes):
        """
        Adds embeddings to the provided nodes by assigning an embedding for each node
        using the specified embedding model and indexing the updated nodes in the data store.

        :param nodes: List of nodes to which embeddings are added. Each node should have a
                      unique `node_id` and support embedding updates.
        :type nodes: list[BaseNode]

        :return: The list of nodes with updated embeddings.
        :rtype: list[BaseNode]

        :raises IndexError: If the current index is marked as read-only.
        """
        if not self.writeable:
            raise IndexError(f'Index {self.index_name()} is read-only')

        id_to_embed_map = embed_nodes(nodes, self.embed_model)

        docs = []

        for node in nodes:

            doc: BaseNode = node.copy()
            doc.embedding = id_to_embed_map[node.node_id]

            docs.append(doc)

        if docs:
            self.client.index_results(docs)

        return nodes

    def _update_filters_recursive(self, filters: MetadataFilters):
        """
        Recursively updates the filters' keys and values to proper format. Specifically,
        it prepends 'source.metadata.' to the key for each MetadataFilter and formats
        the value if the key represents a datetime field. It also processes nested
        MetadataFilters.

        :param filters: An instance of MetadataFilters containing nested filters or
            MetadataFilter objects to be updated.
        :type filters: MetadataFilters
        :return: A modified instance of MetadataFilters with updated keys and, if
            applicable, formatted datetime values.
        :rtype: MetadataFilters
        :raises ValueError: If an unexpected filter type is encountered during the
            update.
        """
        for f in filters.filters:
            if isinstance(f, MetadataFilter):
                f.key = f'source.metadata.{f.key}'
                if is_datetime_key(f.key):
                    f.value = format_datetime(f.value)
            elif isinstance(f, MetadataFilters):
                f = self._update_filters_recursive(f)
            else:
                raise ValueError(f'Unexpected filter type: {type(f)}')
        return filters

    def _get_metadata_filters(self, filter_config: FilterConfig):
        """
        Retrieve and process metadata filters based on the given filter configuration.

        This method takes a filter configuration containing source filters, processes
        them recursively, and returns a deep copy of the updated filters. If no
        valid configuration or source filters are provided, the method returns None.

        :param filter_config: Configuration object containing the source filters.
                              Expected to be of type FilterConfig.
        :return: A processed and deep-copied version of the source filters if present,
                 updated recursively, or None if no valid input is provided.
        """
        if not filter_config or not filter_config.source_filters:
            return None

        filters_copy = filter_config.source_filters.model_copy(deep=True)
        filters_copy = self._update_filters_recursive(filters_copy)

        logger.debug(f'filters: {filters_copy.model_dump_json()}')

        return filters_copy

    def top_k(
        self,
        query_bundle: QueryBundle,
        top_k: int = 5,
        filter_config: Optional[FilterConfig] = None,
    ):
        """
        Retrieve the top k results from the vector store based on the provided query
        bundle. The query is converted to an embedded query, executed against the
        vector store, and the resulting nodes are ranked with a similarity score.
        If the filter configuration is provided, it applies the metadata filters
        to refine the query results.

        :param query_bundle: Structured query information containing the query string
            and embeddings for the vector store search.
        :type query_bundle: QueryBundle
        :param top_k: Number of top results to retrieve. Defaults to 5.
        :type top_k: int, optional
        :param filter_config: Optional filter configuration for metadata-based filtering
            in the vector store.
        :type filter_config: Optional[FilterConfig]
        :return: A list of top k scored nodes transformed into the desired result format.
        :rtype: List[Any]
        """
        query_bundle = to_embedded_query(query_bundle, self.embed_model)

        scored_nodes = []

        try:

            results: VectorStoreQueryResult = self.client.query(
                VectorStoreQueryMode.DEFAULT,
                query_str=query_bundle.query_str,
                query_embedding=query_bundle.embedding,
                k=top_k,
                filters=self._get_metadata_filters(filter_config),
            )

            scored_nodes.extend(
                [
                    NodeWithScore(node=node, score=score)
                    for node, score in zip(results.nodes, results.similarities)
                ]
            )

        except NotFoundError as e:
            if self.tenant_id.is_default_tenant():
                raise e
            else:
                logger.warning(
                    f'Multi-tenant index {self.underlying_index_name()} does not exist'
                )

        return [self._to_top_k_result(node) for node in scored_nodes]

    # opensearch has a limit of 10,000 results per search, so we use this to paginate the search
    def paginated_search(self, query, page_size=10000, max_pages=None):
        """
        Executes a paginated search query on the underlying index and yields the results
        page by page. The function supports setting a limit on the number of pages to fetch
        and controlling the maximum number of results returned per page.

        The method utilizes the Elasticsearch `search_after` mechanism for better performance
        when paginating over a large result set.

        :param query: The search query to execute.
        :type query: dict
        :param page_size: Optional. Number of results to return per page. Defaults to 10000.
        :type page_size: int
        :param max_pages: Optional. Maximum number of pages to retrieve. If None, all results
            are fetched until exhaustion.
        :type max_pages: int, optional
        :return: Yields a list of search hits for each page.
        :rtype: generator
        """
        client = self.client._os_client

        if not client:
            pass

        search_after = None
        page = 0

        while True:
            body = {"size": page_size, "query": query, "sort": [{"_id": "asc"}]}

            if search_after:
                body["search_after"] = search_after

            response = client.search(index=self.underlying_index_name(), body=body)

            hits = response['hits']['hits']
            if not hits:
                break

            yield hits

            search_after = hits[-1]['sort']
            page += 1

            if max_pages and page >= max_pages:
                break

    def get_all_embeddings(self, query: str, max_results=None):
        """
        Fetches all embeddings for a given query using paginated search and processes
        them into a structured format. Pagination is handled internally and the results
        are limited by the specified maximum number of results, if provided. Each
        embedding is retrieved and transformed using the `_to_get_embedding_result` method.

        :param query: The search query string to retrieve embeddings for.
        :type query: str
        :param max_results: Optional limit on the maximum number of results to retrieve.
            If set to None, all results will be retrieved without a limit.
        :type max_results: int, optional
        :return: A list of transformed embeddings based on the search query, limited by
            the specified maximum results if given.
        :rtype: list
        """
        all_results = []

        for page in self.paginated_search(query, page_size=10000):
            all_results.extend(self._to_get_embedding_result(hit) for hit in page)
            if max_results and len(all_results) >= max_results:
                all_results = all_results[:max_results]
                break

        return all_results

    def get_embeddings(self, ids: List[str] = []):
        """
        Fetch embeddings corresponding to the given list of IDs. The function
        builds a query to search for embeddings matching the provided IDs and
        retrieves all associated embeddings. The IDs are cleaned to align with
        system requirements before being used in the query.

        :param ids: A list of unique identifiers for which the embeddings are
            to be fetched.
        :type ids: List[str]

        :return: A collection of embeddings retrieved based on the input IDs.
        :rtype: Any
        """
        query = {
            "terms": {
                f'metadata.{INDEX_KEY}.key': [self._clean_id(i) for i in set(ids)]
            }
        }

        results = self.get_all_embeddings(query, max_results=len(ids) * 2)

        return results
