# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import string
import logging
from typing import Any, List, Optional, Callable

from graphrag_toolkit.lexical_graph.metadata import FilterConfig, type_name_for_key_value, format_datetime
from graphrag_toolkit.lexical_graph.config import GraphRAGConfig
from graphrag_toolkit.lexical_graph.storage import GraphStoreFactory
from graphrag_toolkit.lexical_graph.storage.graph import GraphStore, MultiTenantGraphStore
from graphrag_toolkit.lexical_graph.storage.graph.graph_utils import node_result
from graphrag_toolkit.lexical_graph.storage.graph.neptune_graph_stores import NeptuneAnalyticsClient
from graphrag_toolkit.lexical_graph.storage.vector import VectorIndex, VectorIndexFactoryMethod, to_embedded_query

from llama_index.core.indices.utils import embed_nodes
from llama_index.core.schema import QueryBundle
from llama_index.core.vector_stores.types import FilterCondition, FilterOperator, MetadataFilter, MetadataFilters

logger = logging.getLogger(__name__)

NEPTUNE_ANALYTICS = 'neptune-graph://'

def to_opencypher_operator(operator: FilterOperator) -> tuple[str, Callable[[Any], str]]:
    """
    Converts a given filter operator into its corresponding OpenCypher operator and value formatter.

    This function maps an instance of `FilterOperator` to a tuple containing the
    OpenCypher operator string and a callable for formatting values associated
    with that operator. If the given operator is not supported, the function raises
    a `ValueError`.

    Args:
        operator: The filter operator to convert. Expected to be an instance of
            `FilterOperator`.

    Returns:
        A tuple where:
        - The first element is a string representing the corresponding OpenCypher
          operator.
        - The second element is a callable, representing a function used to format
          the value for OpenCypher queries.

    Raises:
        ValueError: If the provided operator is not supported.
    """
    default_value_formatter = lambda x: x
    
    operator_map = {
        FilterOperator.EQ: ('=', default_value_formatter), 
        FilterOperator.GT: ('>', default_value_formatter), 
        FilterOperator.LT: ('<', default_value_formatter), 
        FilterOperator.NE: ('<>', default_value_formatter), 
        FilterOperator.GTE: ('>=', default_value_formatter), 
        FilterOperator.LTE: ('<=', default_value_formatter), 
        #FilterOperator.IN: ('in', default_value_formatter),  # In array (string or number)
        #FilterOperator.NIN: ('nin', default_value_formatter),  # Not in array (string or number)
        #FilterOperator.ANY: ('any', default_value_formatter),  # Contains any (array of strings)
        #FilterOperator.ALL: ('all', default_value_formatter),  # Contains all (array of strings)
        FilterOperator.TEXT_MATCH: ('CONTAINS', default_value_formatter),
        FilterOperator.TEXT_MATCH_INSENSITIVE: ('CONTAINS', lambda x: x.lower()),
        #FilterOperator.CONTAINS: ('contains', default_value_formatter),  # metadata array contains value (string or number)
        FilterOperator.IS_EMPTY: ('IS NULL', default_value_formatter),  # the field is not exist or empty (null or empty array)
    }

    if operator not in operator_map:
        raise ValueError(f'Unsupported filter operator: {operator}')
    
    return operator_map[operator]

def formatter_for_type(type_name:str) -> Callable[[Any], str]:
    """
    Determines and returns a formatting function based on the provided type name.

    This function takes a `type_name` as input and returns a corresponding
    lambda function that formats the provided value into a specific format
    associated with the type. Supported type names include 'text', 'timestamp',
    and 'number'. If the type name does not match one of the supported types,
    a `ValueError` is raised.

    Args:
        type_name: The name of the type for which the formatter is required.
            Valid values include 'text', 'timestamp', and 'number'.

    Returns:
        Callable[[Any], str]: A lambda function that formats the input value
        according to the provided `type_name`.

    Raises:
        ValueError: If an unsupported type name is provided.
    """
    if type_name == 'text':
        return lambda x: f"'{x}'"
    elif type_name == 'timestamp':
        return lambda x: f"datetime('{format_datetime(x)}')"
    elif type_name == 'number':
        return lambda x:x
    else:
        raise ValueError(f'Unsupported type name: {type_name}')

def parse_metadata_filters_recursive(metadata_filters:MetadataFilters) -> str:
    """
    Parses a `MetadataFilters` object into an OpenCypher filter string.

    This function recursively processes `MetadataFilters` and `MetadataFilter` objects
    to construct a representation suitable for OpenCypher query language based on
    filter conditions ('AND', 'OR', 'NOT') and operator transformations. The resulting
    string can be directly incorporated into OpenCypher queries for filtering nodes
    or relationships.

    Args:
        metadata_filters (MetadataFilters): The filter structure to be parsed, which
            defines the conditions (`AND`, `OR`, `NOT`) and associated filters or nested
            filters to be converted into OpenCypher query format.

    Returns:
        str: The constructed OpenCypher filter string generated from the input metadata
        filter structure.

    Raises:
        ValueError: If the metadata filter structure contains unexpected or invalid
            types, or if an unsupported filter condition is encountered.
    """
    def to_key(key: str) -> str:
        """
        Recursively parses metadata filters and converts them into a formatted string.

        This function takes a MetadataFilters object and iteratively processes it,
        transforming its content into a specific string format for further usage.

        Args:
            metadata_filters: The MetadataFilters object that contains the filtering
                criteria to be processed.

        Returns:
            A string that represents the parsed and formatted metadata filters based
            on the provided input.

        """
        return f"source.{key}"
    
    def metadata_filter_to_opencypher_filter(f: MetadataFilter) -> str:
        """
        Parses a `MetadataFilters` object recursively into a string representation usable as an OpenCypher filter expression.

        This function processes a given `MetadataFilters` object by iterating through its filters
        and recursively transforming them into OpenCypher-compatible filter expressions. The helper
        function `metadata_filter_to_opencypher_filter` ensures that each individual filter is properly
        formatted based on its key, operator, and value.

        Args:
            metadata_filters (MetadataFilters): The metadata filters to be recursively parsed into an
                OpenCypher-compatible string representation.

        Returns:
            str: The OpenCypher-compatible filter expression string after parsing the input
                `MetadataFilters` object.
        """
        key = to_key(f.key)
        (operator, operator_formatter) = to_opencypher_operator(f.operator)

        if f.operator == FilterOperator.IS_EMPTY:
            return f"({key} {operator})"
        else:
            type_name = type_name_for_key_value(f.key, f.value)
            type_formatter = formatter_for_type(type_name)
            if f.operator == FilterOperator.TEXT_MATCH_INSENSITIVE:
                return f"({key}.toLower() {operator} {type_formatter(operator_formatter(str(f.value)))})"
            else:
                return f"({key} {operator} {type_formatter(operator_formatter(str(f.value)))})"
 
    
    condition = metadata_filters.condition.value

    filter_strs = []

    for metadata_filter in metadata_filters.filters:
        if isinstance(metadata_filter, MetadataFilter):
            if metadata_filters.condition == FilterCondition.NOT:
                raise ValueError(f'Expected MetadataFilters for FilterCondition.NOT, but found MetadataFilter')
            filter_strs.append(metadata_filter_to_opencypher_filter(metadata_filter))
        elif isinstance(metadata_filter, MetadataFilters):
            filter_strs.append(parse_metadata_filters_recursive(metadata_filter))
        else:
            raise ValueError(f'Invalid metadata filter type: {type(metadata_filter)}')
        
    if metadata_filters.condition == FilterCondition.NOT:
        return f"(NOT {' '.join(filter_strs)})"
    elif metadata_filters.condition == FilterCondition.AND or metadata_filters.condition == FilterCondition.OR:
        condition = f' {metadata_filters.condition.value.upper()} '
        return f"({condition.join(filter_strs)})"
    else:
        raise ValueError(f'Unsupported filters condition: {metadata_filters.condition}')


def filter_config_to_opencypher_filters(filter_config:FilterConfig) -> str:
    """
    Converts filter configuration into a string containing filters in OpenCypher format.

    This function takes a `FilterConfig` object as input, which may include a set of
    source filters. The representation of the filters is recursively constructed
    and returned as a string formatted for OpenCypher queries. If the filter
    configuration or its source filters are not provided, an empty string is
    returned.

    Args:
        filter_config (FilterConfig): The configuration object containing source
            filters to be converted into OpenCypher format.

    Returns:
        str: The OpenCypher formatted filters as a string. Returns an empty string
            if no filters are provided or the filter configuration is invalid.
    """
    if filter_config is None or filter_config.source_filters is None:
        return ''
    return parse_metadata_filters_recursive(filter_config.source_filters)
    
class NeptuneAnalyticsVectorIndexFactory(VectorIndexFactoryMethod):
    def try_create(self, index_names:List[str], vector_index_info:str, **kwargs) -> List[VectorIndex]:
        """
        Attempts to create a list of vector indices based on the given index names and vector index
        information. If the vector index information specifies a Neptune Analytics configuration,
        the function initializes Neptune indices for each index name. Otherwise, it returns None.

        Args:
            index_names (List[str]): A list of names for the indices to attempt creation.
            vector_index_info (str): A string containing vector index information or configuration.
            **kwargs: Additional keyword arguments passed to the index creation process.

        Returns:
            List[VectorIndex] or None: A list of VectorIndex instances if the operation is
            successful and the vector index information matches the Neptune Analytics configuration.
            Returns None if the vector index information does not match the expected configuration.
        """
        graph_id = None
        if vector_index_info.startswith(NEPTUNE_ANALYTICS):
            graph_id = vector_index_info[len(NEPTUNE_ANALYTICS):]
            logger.debug(f'Opening Neptune Analytics vector indexes [index_names: {index_names}, graph_id: {graph_id}]')
            return [NeptuneIndex.for_index(index_name, vector_index_info, **kwargs) for index_name in index_names]
        else:
            return None

class NeptuneIndex(VectorIndex):
    """
    Represents a NeptuneIndex specific to a graph database analytics system.

    This class is used to manage and interact with a vector index within a Neptune
    graph store. It provides functionality for creating the index based on the given
    parameters, embedding nodes, querying for top K neighboring elements by embedding
    similarity, and retrieving embeddings for specific node IDs. The class defines
    specific traversal paths and formats data into structures compatible with the
    underlying analytics platform, utilizing a tenant-specific configuration.

    Attributes:
        neptune_client (NeptuneAnalyticsClient): The client responsible for interacting
            with the Neptune graph database.
        embed_model (Any): The model used for generating embeddings for graph nodes.
        dimensions (int): The dimensionality of embeddings.
        id_name (str): The identifier name used for indexing nodes.
        label (str): The label corresponding to the type of index in the graph database.
        path (str): The traversal path defined for the index queries.
        return_fields (str): The fields to return when executing queries related to the
            index.
    """
    @staticmethod
    def for_index(index_name, graph_id, embed_model=None, dimensions=None, **kwargs):
        """
        Creates and configures an instance of `NeptuneIndex` for a specified index type.

        The method sets up an index by defining the index name, graph database client,
        embedding model, dimensions of embeddings, and other necessary parameters. It
        also constructs specific paths and return fields for the given index type and
        validates the `index_name`.

        Args:
            index_name (str): The name of the index to be configured. Must be a valid
                index name.
            graph_id (str): The identifier for the graph database for which the index
                is created.
            embed_model (Optional[Any]): Embedding model to be used for the index. If
                not specified, a default model from `GraphRAGConfig` is used.
            dimensions (Optional[int]): The dimensionality of embeddings. Uses default
                from `GraphRAGConfig` if not provided.
            **kwargs: Arbitrary keyword arguments passed for additional graph store
                configurations.

        Returns:
            NeptuneIndex: A configured NeptuneIndex instance for the specified index
            name.

        Raises:
            ValueError: If the provided `index_name` is invalid or unrecognized.
        """
        index_name = index_name.lower()
        neptune_client:GraphStore = GraphStoreFactory.for_graph_store(graph_id, **kwargs)
        embed_model = embed_model or GraphRAGConfig.embed_model
        dimensions = dimensions or GraphRAGConfig.embed_dimensions
        id_name = f'{index_name}Id'
        label = f'__{string.capwords(index_name)}__' 
        path = f'({index_name})'
        return_fields = node_result(index_name, neptune_client.node_id(f'{index_name}.{id_name}'))

        if index_name == 'chunk':
            path = '(chunk)-[:`__EXTRACTED_FROM__`]->(source:`__Source__`)'
            return_fields = f"source:{{sourceId: {neptune_client.node_id('source.sourceId')}, {node_result('source', key_name='metadata')}}},\n{node_result('chunk', neptune_client.node_id('chunk.chunkId'), [])}"
        elif index_name == 'statement':
            path = '(statement)-[:`__MENTIONED_IN__`]->(:`__Chunk__`)-[:`__EXTRACTED_FROM__`]->(source:`__Source__`)'
        elif index_name == 'topic':
            path = '(topic)-[:`__MENTIONED_IN__`]->(:`__Chunk__`)-[:`__EXTRACTED_FROM__`]->(source:`__Source__`)'
        else:
            raise ValueError(f'Invalid index name: {index_name}')
            
        return NeptuneIndex(
            index_name=index_name,
            neptune_client=neptune_client,
            embed_model=embed_model,
            dimensions=dimensions,
            id_name=id_name,
            label=label,
            path=path,
            return_fields=return_fields
        ) 


    neptune_client: NeptuneAnalyticsClient
    embed_model: Any
    dimensions: int
    id_name: str
    label: str
    path: str
    return_fields: str

    def _neptune_client(self):
        """
        Creates and returns the appropriate Neptune client based on tenant ID.

        If the tenant ID corresponds to the default tenant, the unwrapped Neptune
        client is returned. Otherwise, it wraps the Neptune client with a
        Multi-tenant Graph Store for the corresponding tenant ID and returns it.

        Returns:
            NeptuneClient or MultiTenantGraphStore: An instance of the Neptune
            client or a wrapped Multi-tenant Graph Store object.

        """
        if self.tenant_id.is_default_tenant():
            return self.neptune_client
        else:
            return MultiTenantGraphStore.wrap(self.neptune_client, tenant_id=self.tenant_id)

    
    def add_embeddings(self, nodes):
        """
        Adds embeddings to the provided list of nodes by using the instance's embedding model
        and updates these embeddings in Neptune using the appropriate queries.

        This function takes a list of nodes, computes their embeddings using the associated
        embedding model, and updates these embeddings in the Neptune database. It also ensures
        that temporary metadata added to the nodes is removed after processing.

        Args:
            nodes (list): A list of node objects. Each node should have a `metadata` dictionary
                and a unique identifier `node_id`.

        Returns:
            list: The processed list of nodes with updated embeddings in Neptune.
        """
        text_map = { node.node_id: node.text for node in nodes }
        
        for node in nodes:
            node.metadata['index'] = self.underlying_index_name()
            node.text = f'''index: {self.underlying_index_name()}

{node.text}

index: {self.underlying_index_name()}
'''
                    
        id_to_embed_map = embed_nodes(
            nodes, self.embed_model
        )
        
        for node in nodes:
        
            statement = f"MATCH (n:`{self.label}`) WHERE {self.neptune_client.node_id('n.{self.id_name}')} = $nodeId"
            
            embedding = id_to_embed_map[node.node_id]
            
            query = '\n'.join([
                statement,
                f'WITH n CALL neptune.algo.vectors.upsert(n, {embedding}) YIELD success RETURN success'
            ])
            
            properties = {
                'nodeId': node.node_id,
                'embedding': embedding
            }

            self._neptune_client().execute_query_with_retry(query, properties)
            
            
            
        for node in nodes:
            node.metadata.pop('index', None)
            node.text = text_map[node.node_id]
        
        return nodes
    
    def top_k(self, query_bundle:QueryBundle, top_k:int=5, filter_config:Optional[FilterConfig]=None):
        """
        Fetches the top-k records ranked by similarity score based on query embeddings
        from an underlying Neptune Graph database index. The function internally performs
        embedding transformations, constructs queries, and applies filtering criteria.

        Args:
            query_bundle (QueryBundle): The query bundle containing the query string and
                potentially other context required for querying the index.
            top_k (int, optional): The number of top records to fetch based on rank. Defaults to 5.
            filter_config (Optional[FilterConfig]): Configuration for applying filters to
                the query to refine results, if provided.

        Returns:
            List[Dict]: A list of results where each result dictionary contains a 'score' field
                and the fields specified in self.return_fields.
        """
        query_str = f'''index: {self.underlying_index_name()}

{query_bundle.query_str}
'''

        query_bundle = QueryBundle(query_str=query_str) 
        query_bundle = to_embedded_query(query_bundle, self.embed_model)

        tenant_specific_label = self.tenant_id.format_label(self.label).replace('`', '')

        where_clause =  filter_config_to_opencypher_filters(filter_config)
        where_clause = f'WHERE {where_clause}' if where_clause else ''

        logger.debug(f'filter: {where_clause}')

        cypher = f'''
        CALL neptune.algo.vectors.topKByEmbedding(
            {query_bundle.embedding},
            {{   
                topK: {top_k * 5},
                concurrency: 4
            }}
        )
        YIELD node, score       
        WITH node as {self.index_name}, score WHERE '{tenant_specific_label}' in labels({self.index_name}) 
        WITH {self.index_name}, score ORDER BY score ASC LIMIT {top_k}
        MATCH {self.path}
        {where_clause}
        RETURN {{
            score: score,
            {self.return_fields}
        }} AS result ORDER BY result.score ASC LIMIT {top_k}
        '''

        results = self._neptune_client().execute_query(cypher)
        
        return [result['result'] for result in results]

    def get_embeddings(self, ids:List[str]=[]):
        """
        Fetches embeddings for the specified node IDs by executing a Cypher query on the
        database. The function retrieves node embeddings and other specified result fields
        based on tenant-specific configurations and returns the collected results.

        Args:
            ids (List[str]): A list of unique node IDs for which embeddings are to be fetched.
                The IDs should correspond to nodes in the database.

        Returns:
            List[Dict]: A list of dictionaries, where each dictionary contains the retrieved
            embedding and other specified fields for a node.
        """
        all_results = []

        tenant_specific_label = self.tenant_id.format_label(self.label).replace('`', '')
        
        for i in set(ids):

            cypher = f'''
            MATCH (n:`{self.label}`)  WHERE {self.neptune_client.node_id('n.{self.id_name}')} = $elementId
            CALL neptune.algo.vectors.get(
                n
            )
            YIELD node, embedding       
            WITH node as {self.index_name}, embedding WHERE '{tenant_specific_label}' in labels({self.index_name}) 
            MATCH {self.path}
            RETURN {{
                embedding: embedding,
                {self.return_fields}
            }} AS result
            '''
            
            params = {
                'elementId': i
            }
            
            results = self._neptune_client().execute_query(cypher, params)
            
            for result in results:
                all_results.append(result['result'])
        
        return all_results
