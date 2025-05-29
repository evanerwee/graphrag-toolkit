# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import json
import logging
from typing import List, Any, Optional
from dataclasses import dataclass

from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.schema import BaseNode, NodeWithScore, QueryBundle
from llama_index.core.vector_stores.types import  VectorStoreQueryResult, VectorStoreQueryMode, MetadataFilters, MetadataFilter
from llama_index.core.indices.utils import embed_nodes
from llama_index.core.vector_stores.types import MetadataFilters

from graphrag_toolkit.lexical_graph.metadata import FilterConfig, is_datetime_key, format_datetime
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
    #info = asyncio_run(self._os_async_client.info())
    return '2.0.9'

import llama_index.vector_stores.opensearch 
llama_index.vector_stores.opensearch.OpensearchVectorClient._get_opensearch_version = _get_opensearch_version

@dataclass
class DummyAuth:
    """
    Represents an authentication provider for a specific service.

    This class is used to manage and represent the authentication details specific
    to a given service. It acts as a container for storing information related to
    the service requiring authentication.

    Attributes:
        service (str): The name of the service for which authentication is being
            provided.
    """
    service:str

def create_os_client(endpoint, **kwargs):
    """
    Creates an OpenSearch client configured to use AWS Signature Version 4
    authentication.

    The function utilizes AWS credentials from a pre-configured session and region
    information. It signs requests using Urllib3-based AWS Signature V4
    authentication. The client is initialized with specific connection properties
    like SSL usage, certificate verification, timeout, retry settings, and any
    additional keyword arguments provided.

    Args:
        endpoint: str
            The OpenSearch endpoint URL to connect to.
        **kwargs: Any
            Additional keyword arguments passed to the OpenSearch client.

    Returns:
        OpenSearch
            A configured client instance for interacting with OpenSearch.

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
        **kwargs
    )

def create_os_async_client(endpoint, **kwargs):
    """
    Creates an asynchronous OpenSearch client.

    This function configures and returns an instance of the AsyncOpenSearch client,
    configured for secure communication with the specified endpoint. The client
    leverages AWS Signature Version 4 for authentication.

    Args:
        endpoint: The URL of the OpenSearch cluster endpoint.
        **kwargs: Optional parameters for customizing the AsyncOpenSearch client.

    Returns:
        AsyncOpenSearch: An instantiated asynchronous OpenSearch client.
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
        **kwargs
    )
    

def index_exists(endpoint, index_name, dimensions, writeable) -> bool:
    """
    Checks if an OpenSearch index exists, and optionally creates it if it does not exist.

    This function uses the specified endpoint to establish a connection to an
    OpenSearch client. If the index specified by `index_name` does not exist and
    the `writeable` flag is True, it will create an index using the provided
    dimensions and a predefined method for handling knn_vector elements.

    Args:
        endpoint: The OpenSearch endpoint to connect to.
        index_name: The name of the index to check for existence.
        dimensions: The number of dimensions for the knn_vector in the index.
        writeable: Flag indicating whether to create the index if it does not exist.

    Returns:
        bool: True if the index exists (or is created successfully), False otherwise.
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
        }
    }

    index_exists = False

    try:
        index_exists = client.indices.exists(index_name)
        if not index_exists and writeable:
            logger.debug(f'Creating OpenSearch index [index_name: {index_name}, endpoint: {endpoint}]')
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
    Creates an OpenSearch vector client for interacting with an OpenSearch cluster.

    This function initializes and configures the `OpensearchVectorClient`, which is
    used for storing, managing, and querying vectors in an OpenSearch index. It includes
    a retry mechanism to handle transient errors during the client initialization process.

    Args:
        endpoint: Endpoint URL of the OpenSearch cluster.
        index_name: Name of the index to be used for storing vectors.
        dimensions: Dimensions of the vector space used for embeddings.
        embed_model: Embedding model associated with the vectors.

    Returns:
        OpensearchVectorClient: A configured OpenSearch vector client instance.

    Raises:
        NotFoundError: If the client cannot be initialized after exceeding the maximum
            retry attempts.
    """
    text_field = 'value'
    embedding_field = 'embedding'

    logger.debug(f'Creating OpenSearch vector client [endpoint: {endpoint}, index_name={index_name}, embed_model={embed_model}, dimensions={dimensions}]')
 
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
                http_auth=DummyAuth(service='aoss')
            )
        except NotFoundError as err:
            retry_count += 1
            logger.warning(f'Error while creating OpenSearch vector client [retry_count: {retry_count}, error: {err}]')
            if retry_count > 3:
                raise err
                
    logger.debug(f'Created OpenSearch vector client [client: {client}, retry_count: {retry_count}]')
            
    return client
        
class DummyOpensearchVectorClient():
    """
    A client for interfacing with an OpenSearch vector store.

    The DummyOpensearchVectorClient is designed as a placeholder implementation of a
    vector client for OpenSearch. It provides methods to index content and query vectors
    against the OpenSearch store. This implementation does not perform actual indexing
    or querying but serves as a template for a functional OpenSearch vector client.

    Attributes:
        _os_async_client (Optional): Asynchronous OpenSearch client instance.
    """
    def __init__(self):
        self._os_async_client = None

    def index_results(self, nodes: List[BaseNode], **kwargs: Any) -> List[str]:
        """
        Indexes the provided nodes and returns a list of their identifiers.

        The method is responsible for processing a list of nodes, performing the
        necessary indexing, and returning a list of unique identifiers corresponding
        to the indexed nodes. Additional parameters can be passed through **kwargs for
        custom configurations or behaviors during the indexing process.

        Args:
            nodes: A list of nodes to be indexed.
            **kwargs: Arbitrary keyword arguments for additional configuration.

        Returns:
            A list of strings, where each string is the unique identifier of a
            successfully indexed node.

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
        Executes a query on a vector store and returns the results based on the specified mode, query
        parameters, and additional filters. The function supports different query mechanisms and allows users
        to search using text strings or embeddings. It retrieves matching nodes, their similarities, and unique
        identifiers.

        Args:
            query_mode: The mode in which the query should be executed. Defines how the search is processed,
                e.g., similarity-based or keyword-based.
            query_str: An optional text string to query. Only applicable for modes that support text-based
                searching.
            query_embedding: A list of floats representing the query embedding. Used in similarity-based
                searching.
            k: The number of top results to retrieve. Determines how many matching entries are returned.
            filters: Optional metadata filters to narrow down the search results based on specific criteria.

        Returns:
            VectorStoreQueryResult: An object containing nodes that match the query, their similarity scores,
            and unique identifiers.
        """
        return VectorStoreQueryResult(nodes=[], similarities=[], ids=[])
        
    
class OpenSearchIndex(VectorIndex):
    """
    Represents an OpenSearch-based vector index for storing and querying embeddings.

    This class inherits from `VectorIndex` and provides functionality specific to
    OpenSearch for creating, managing, and querying embeddings. OpenSearch enables
    vector searches with scoring and filtering mechanisms. This class supports
    integration with various embedding models and configurations. The primary
    use-case is to host embeddings in an OpenSearch-compatible format and allow
    efficient retrieval or querying.

    Attributes:
        endpoint (str): The endpoint URL for the OpenSearch service.
        index_name (str): The name of the index in OpenSearch.
        dimensions (int): The dimensionality of the embedding vectors.
        embed_model (EmbeddingType): The model used for generating embeddings.
        _client (OpensearchVectorClient): Private attribute representing a
            client instance for interacting with OpenSearch, initialized on demand.
    """
    @staticmethod
    def for_index(index_name, endpoint, embed_model=None, dimensions=None):
        """
        Creates and returns an instance of OpenSearchIndex using the provided parameters.

        This method allows the configuration of an OpenSearch index by specifying its
        name, endpoint, optional embedding model, and dimensions. It ensures fallback
        to default embedding model and dimensions defined in the GraphRAGConfig class
        when not explicitly passed as arguments.

        Args:
            index_name (str): The name of the OpenSearch index to be created or
                retrieved.
            endpoint (str): The endpoint URL for the OpenSearch instance.
            embed_model (Optional[str]): The embedding model to be used. Defaults to
                the configuration specified in GraphRAGConfig.
            dimensions (Optional[int]): The dimensions to be used for the embeddings.
                Defaults to the configuration specified in GraphRAGConfig.

        Returns:
            OpenSearchIndex: An instance of OpenSearchIndex initialized with the
                specified or default parameters.
        """
        embed_model = embed_model or GraphRAGConfig.embed_model
        dimensions = dimensions or GraphRAGConfig.embed_dimensions

        return OpenSearchIndex(index_name=index_name, endpoint=endpoint, dimensions=dimensions, embed_model=embed_model)
    
    class Config:
        """Handles configuration settings and specifies validation rules.

        The Config class defines specific configuration behaviors and settings
        used in the application. It is commonly utilized to enforce specific
        rules or facilitate custom behavior when working with certain
        types of data or attributes.

        Attributes:
            arbitrary_types_allowed (bool): Indicates whether the
                configuration will allow arbitrary types.
        """
        arbitrary_types_allowed = True

    endpoint:str
    index_name:str
    dimensions:int
    embed_model:EmbeddingType

    _client: OpensearchVectorClient = PrivateAttr(default=None)

    def __getstate__(self):
        """
        Serializes the current state of the object while excluding the client attribute.

        This method overrides the default __getstate__ method to ensure the _client
        attribute is set to None during the serialization process. It returns the
        state from the superclass's implementation of __getstate__.

        Returns:
            dict: The serialized state of the object.
        """
        self._client = None
        return super().__getstate__()
        
    @property
    def client(self) -> OpensearchVectorClient:
        """
        Retrieves or creates an OpenSearch vector client for interacting with the
        specified index. Ensures that the client is initialized appropriately,
        either as an actual OpenSearch vector client or as a dummy client,
        based on the index's existence and configuration.

        Attributes:
            client (OpensearchVectorClient): Provides access to the OpenSearch
                vector client instance.

        Returns:
            OpensearchVectorClient: The initialized or cached OpenSearch vector
                client instance.
        """
        if self._client:
            if self._client._index != self.underlying_index_name():
                self._client = None

        if not self._client:
            if index_exists(self.endpoint, self.underlying_index_name(), self.dimensions, self.writeable):
                self._client = create_opensearch_vector_client(
                    self.endpoint, 
                    self.underlying_index_name(), 
                    self.dimensions, 
                    self.embed_model
                )
            else:
                self._client = DummyOpensearchVectorClient()
        return self._client
        
    def _clean_id(self, s):
        """
        Cleans and normalizes a given string by removing all non-alphanumeric characters.

        This method processes the input string by iterating over each character and filtering
        out any character that is not alphanumeric. The resulting string, consisting only of
        alphanumeric characters, is returned as the cleaned string.

        Args:
            s: A string input that needs to be cleaned of non-alphanumeric characters.

        Returns:
            A string containing only alphanumeric characters from the input string.
        """
        return ''.join(c for c in s if c.isalnum())
    
    def _to_top_k_result(self, r):
        """
        Converts the input result object into a standardized top-k result format. The
        function processes metadata from the given result object and extracts relevant
        information to construct a dictionary representation that adheres to the top-k
        result structure.

        Args:
            r: A result object containing metadata, score, and other related information
                extracted from an operation.

        Returns:
            dict: A dictionary containing the standardized top-k result information.
            It includes the score of the result, and metadata extracted under specific
            keys such as index or source. Additional metadata is included in the resultant
            dictionary, depending on its availability in the input object.
        """
        result = {
            'score': r.score 
        }

        if INDEX_KEY in r.metadata:
            index_name = r.metadata[INDEX_KEY]['index']
            result[index_name] = r.metadata[index_name]
            if 'source' in r.metadata:
                result['source'] = r.metadata['source']
        else:
            for k,v in r.metadata.items():
                result[k] = v
            
        return result
        
    def _to_get_embedding_result(self, hit):
        """
        Converts a hit object from a search result into a structured embedding result.

        This method processes the provided `hit` object to extract and organize relevant
        fields into a structured `result` dictionary. It includes the ID, value, and embedding
        from the source, along with additional metadata extracted from the `_node_content`.

        Args:
            hit: Dict representing the search result object. It is expected to have a `_source`
                key containing the primary data, which includes an `id`, `value`, `embedding`,
                and `metadata`.

        Returns:
            Dict: A dictionary containing the `id`, `value`, `embedding`, and any additional
            metadata fields, excluding the `INDEX_KEY` field.
        """
        source = hit['_source']
        data = json.loads(source['metadata']['_node_content'])

        result = {
            'id': source['id'],
            'value': source['value'],
            'embedding': source['embedding']
        }

        for k,v in data['metadata'].items():
            if k != INDEX_KEY:
                result[k] = v
            
        return result

    def add_embeddings(self, nodes):
        """
        Adds embeddings to the given nodes and stores them in the index.

        This method takes a list of nodes, generates embeddings for them using the
        embedding model, and stores the updated nodes with embeddings into the index.
        The operation is performed only if the index is writable; otherwise, an
        `IndexError` is raised.

        Args:
            nodes: List[BaseNode]
                A list of node objects to which embeddings will be added. Each node
                must have a `node_id` that maps to their corresponding embedding.

        Returns:
            List[BaseNode]:
                The list of nodes with updated embeddings.

        Raises:
            IndexError:
                If the index is marked as read-only.
        """
        if not self.writeable:
            raise IndexError(f'Index {self.index_name()} is read-only')

        id_to_embed_map = embed_nodes(
            nodes, self.embed_model
        )

        docs = []

        for node in nodes:

            doc:BaseNode = node.copy()
            doc.embedding = id_to_embed_map[node.node_id]

            docs.append(doc)

        if docs:
            self.client.index_results(docs)
        
        return nodes
    
    def _update_filters_recursive(self, filters:MetadataFilters):
        """
        Recursively updates metadata filters to convert keys to a consistent format and updates
        values for datetime-specific keys.

        This method iterates over a given `MetadataFilters` object and modifies each filter
        contained within it. If a filter is of type `MetadataFilter`, its `key` is updated
        to a prefixed format, and if its key pertains to a datetime, its `value` is formatted
        accordingly. If a filter is itself a `MetadataFilters` instance, the method calls
        itself recursively to process nested filters. An error is raised if an unexpected
        filter type is encountered.

        Args:
            filters (MetadataFilters): The metadata filters to be recursively updated.

        Returns:
            MetadataFilters: The updated metadata filters with all keys processed
                and datetime-specific values formatted.

        Raises:
            ValueError: If a filter is found with an unexpected type.
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
                
    def _get_metadata_filters(self, filter_config:FilterConfig):
        """
        Retrieves and processes metadata filters based on the provided filter configuration.

        This function checks if the given filter configuration object and its source filters are
        valid. If valid, it creates a deep copy of the source filters from the configuration,
        applies recursive updates, and logs the resulting filters in JSON format. The updated
        filters are returned, or `None` if the filter configuration is invalid or empty.

        Args:
            filter_config (FilterConfig): The filter configuration object containing source filters
                to be processed. If this parameter is `None` or lacks source filters, the function
                will return `None`.

        Returns:
            SourceFilters | None: A deep copy of the updated source filters if valid, or `None` if
                the provided filter configuration is invalid or lacks source filters.
        """
        if not filter_config or not filter_config.source_filters:
            return None
        
        filters_copy = filter_config.source_filters.model_copy(deep=True) 
        filters_copy = self._update_filters_recursive(filters_copy)
                
        logger.debug(f'filters: {filters_copy.model_dump_json()}')

        return filters_copy
    
    def top_k(self, query_bundle:QueryBundle, top_k:int=5, filter_config:Optional[FilterConfig]=None):
        """
        Fetches the top-k most relevant nodes for a given query bundle by querying a vector store.

        This method processes a query bundle to extract query information and embeddings. It then
        interacts with the vector store client to retrieve the top-k most relevant nodes based on
        similarities to the query embedding. If no results are available or the vector store does not
        exist for a non-default tenant, it handles the scenario gracefully. The returned result is a
        list of nodes with their respective relevance scores processed into a finalized format.

        Args:
            query_bundle (QueryBundle): The query bundle containing the query string and its embedding
                for similarity computation.
            top_k (int): The maximum number of top relevant nodes to retrieve. Defaults to 5.
            filter_config (Optional[FilterConfig]): An optional configuration to apply additional
                filtering criteria while fetching nodes.

        Returns:
            List[TopKResult]: A list of processed results containing the top-k nodes ordered by
                relevance score.
        """
        query_bundle = to_embedded_query(query_bundle, self.embed_model)

        scored_nodes = []

        try:

            results:VectorStoreQueryResult = self.client.query(
                VectorStoreQueryMode.DEFAULT,
                query_str=query_bundle.query_str,
                query_embedding=query_bundle.embedding,
                k=top_k,
                filters=self._get_metadata_filters(filter_config)
            )

            scored_nodes.extend([
                NodeWithScore(node=node, score=score)
                for node, score in zip(results.nodes, results.similarities)
            ])

        except NotFoundError as e:
            if self.tenant_id.is_default_tenant():
                raise e
            else:
                logger.warning(f'Multi-tenant index {self.underlying_index_name()} does not exist')

        return [self._to_top_k_result(node) for node in scored_nodes]

    # opensearch has a limit of 10,000 results per search, so we use this to paginate the search
    def paginated_search(self, query, page_size=10000, max_pages=None):
        """
        Executes a paginated search on the specified Elasticsearch index and yields
        the results page by page. Designed to handle large datasets by leveraging
        the `search_after` parameter of Elasticsearch, allowing seamless scrolling
        through data without encountering size limitations. By setting a maximum
        number of pages, users can limit the extent of data retrieval.

        Args:
            query: The Elasticsearch query used to filter the documents to be retrieved.
            page_size: The number of documents to be retrieved in each page. Defaults to 10000.
            max_pages: Optional; The maximum number of pages to retrieve. If None, all
                pages are retrieved until the dataset is fully processed.

        Yields:
            list: A list of documents retrieved for each page.
        """
        client = self.client._os_client

        if not client:
            pass
        
        
        search_after = None
        page = 0
        
        while True:
            body = {
                "size": page_size,
                "query": query,
                "sort": [{"_id": "asc"}]
            }
            
            if search_after:
                body["search_after"] = search_after
                
            response = client.search(
                index=self.underlying_index_name(),
                body=body
            )

            hits = response['hits']['hits']
            if not hits:
                break

            yield hits

            search_after = hits[-1]['sort']
            page += 1

            if max_pages and page >= max_pages:
                break

    def get_all_embeddings(self, query:str, max_results=None):
        """
        Retrieves all embeddings for a given query, optionally limiting the maximum number of
        results returned. This method performs a paginated search using the query parameter and
        accumulates embeddings from each page. If a maximum result count is specified, the
        search halts once the threshold is reached.

        Args:
            query (str): The search query string used to retrieve embeddings.
            max_results (int, optional): The maximum number of embedding results to retrieve.
                If not specified, all matching results will be retrieved.

        Returns:
            list: A list containing all embeddings retrieved based on the search query and
            provided constraints.
        """
        all_results = []
        
        for page in self.paginated_search(query, page_size=10000):
            all_results.extend(self._to_get_embedding_result(hit) for hit in page)
            if max_results and len(all_results) >= max_results:
                all_results = all_results[:max_results]
                break
        
        return all_results
    
    def get_embeddings(self, ids:List[str]=[]):
        """
        Fetches the embeddings for the provided IDs using a specified query format to
        retrieve the relevant data from the metadata index. The method constructs a
        query with unique and cleaned IDs, performs a search, and returns the results.

        Args:
            ids (List[str], optional): A list of unique identifiers for which embeddings
                are to be retrieved. Defaults to an empty list.

        Returns:
            List: A list of embedding data corresponding to the provided IDs.
        """
        query = {
            "terms": {
                f'metadata.{INDEX_KEY}.key': [self._clean_id(i) for i in set(ids)]
            }
        }

        results = self.get_all_embeddings(query, max_results=len(ids) * 2)
        
        return results
