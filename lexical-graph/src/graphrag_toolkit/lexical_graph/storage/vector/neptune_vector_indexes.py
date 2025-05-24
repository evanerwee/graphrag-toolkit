# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import string
import logging
from typing import Any, List, Optional, Callable

from graphrag_toolkit.lexical_graph.metadata import (
    FilterConfig,
    type_name_for_key_value,
    format_datetime,
)
from graphrag_toolkit.lexical_graph.config import GraphRAGConfig
from graphrag_toolkit.lexical_graph.storage import GraphStoreFactory
from graphrag_toolkit.lexical_graph.storage.graph import (
    GraphStore,
    MultiTenantGraphStore,
)
from graphrag_toolkit.lexical_graph.storage.graph.graph_utils import node_result
from graphrag_toolkit.lexical_graph.storage.graph.neptune_graph_stores import (
    NeptuneAnalyticsClient,
)
from graphrag_toolkit.lexical_graph.storage.vector import (
    VectorIndex,
    VectorIndexFactoryMethod,
    to_embedded_query,
)

from llama_index.core.indices.utils import embed_nodes
from llama_index.core.schema import QueryBundle
from llama_index.core.vector_stores.types import (
    FilterCondition,
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
)

logger = logging.getLogger(__name__)

NEPTUNE_ANALYTICS = 'neptune-graph://'


def to_opencypher_operator(
    operator: FilterOperator,
) -> tuple[str, Callable[[Any], str]]:
    """
    Transforms a `FilterOperator` into its corresponding OpenCypher operator and a
    value formatter function. This function maps a custom filter operator to OpenCypher's
    expression syntax, ensuring compatibility between logical filtering operations
    in Python code and OpenCypher queries.

    :param operator: A `FilterOperator` instance that specifies the logical operation
        to be mapped to its OpenCypher equivalent.

    :return: A tuple containing the OpenCypher representation of the operator as a
        string, and a callable function to format values associated with the operator.

    :raises ValueError: If the provided operator is not supported or cannot be mapped
        to an OpenCypher operator.
    """
    default_value_formatter = lambda x: x

    operator_map = {
        FilterOperator.EQ: ('=', default_value_formatter),
        FilterOperator.GT: ('>', default_value_formatter),
        FilterOperator.LT: ('<', default_value_formatter),
        FilterOperator.NE: ('<>', default_value_formatter),
        FilterOperator.GTE: ('>=', default_value_formatter),
        FilterOperator.LTE: ('<=', default_value_formatter),
        # FilterOperator.IN: ('in', default_value_formatter),  # In array (string or number)
        # FilterOperator.NIN: ('nin', default_value_formatter),  # Not in array (string or number)
        # FilterOperator.ANY: ('any', default_value_formatter),  # Contains any (array of strings)
        # FilterOperator.ALL: ('all', default_value_formatter),  # Contains all (array of strings)
        FilterOperator.TEXT_MATCH: ('CONTAINS', default_value_formatter),
        FilterOperator.TEXT_MATCH_INSENSITIVE: ('CONTAINS', lambda x: x.lower()),
        # FilterOperator.CONTAINS: ('contains', default_value_formatter),  # metadata array contains value (string or number)
        FilterOperator.IS_EMPTY: (
            'IS NULL',
            default_value_formatter,
        ),  # the field is not exist or empty (null or empty array)
    }

    if operator not in operator_map:
        raise ValueError(f'Unsupported filter operator: {operator}')

    return operator_map[operator]


def formatter_for_type(type_name: str) -> Callable[[Any], str]:
    """
    Returns a formatter function for the specified type name. The formatter
    function converts input values to a specific string or format based on
    the provided type name. Supported type names include 'text', 'timestamp',
    and 'number'.

    :param type_name: Specifies the type of formatting to be applied.
        - 'text': Formats input values as quoted strings.
        - 'timestamp': Formats input values using a datetime representation.
        - 'number': Returns the input value as-is without modification.
    :return: A callable that formats input values based on the type-specific
        formatting rules.
    :raises ValueError: If an unsupported type name is provided.
    """
    if type_name == 'text':
        return lambda x: f"'{x}'"
    elif type_name == 'timestamp':
        return lambda x: f"datetime('{format_datetime(x)}')"
    elif type_name == 'number':
        return lambda x: x
    else:
        raise ValueError(f'Unsupported type name: {type_name}')


def parse_metadata_filters_recursive(metadata_filters: MetadataFilters) -> str:
    """
    Recursively parses a MetadataFilters object into an OpenCypher-compatible filter expression string.

    This function processes a MetadataFilters object consisting of multiple filtering conditions and
    evaluates them recursively to produce a valid OpenCypher filter string representation. Each filter
    is interpreted and formatted according to the specified condition, operator, and values provided
    in the MetadataFilters object.

    Key operations handled include processing the MetadataFilter and MetadataFilters instances, mapping
    filter conditions such as AND, OR, and NOT, and generating properly formatted expressions. If certain
    cases like unsupported filter conditions or invalid input types are encountered, the function raises
    appropriate exceptions.

    :param metadata_filters: A MetadataFilters object representing the filtering conditions
        to be evaluated recursively.
    :type metadata_filters: MetadataFilters
    :return: A string representation of the OpenCypher-compatible parsed filter.
    :rtype: str
    :raises ValueError: Raises an exception for invalid metadata filter types, unsupported conditions,
        or unexpected mismatches between filter conditions and MetadataFilter object types.
    """

    def to_key(key: str) -> str:
        """Recursively parses metadata filters and converts them into a
        formatted string.

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
        Recursively parses a tree of metadata filters into an OpenCypher string representation.
        Each metadata filter is converted based on its operator and associated key-value pair.

        :param metadata_filters: The root node of metadata filters to parse, containing nested
            conditions for filtering metadata.
        :type metadata_filters: MetadataFilters
        :return: An OpenCypher string representation of the metadata filters.
        :rtype: str
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
                raise ValueError(
                    f'Expected MetadataFilters for FilterCondition.NOT, but found MetadataFilter'
                )
            filter_strs.append(metadata_filter_to_opencypher_filter(metadata_filter))
        elif isinstance(metadata_filter, MetadataFilters):
            filter_strs.append(parse_metadata_filters_recursive(metadata_filter))
        else:
            raise ValueError(f'Invalid metadata filter type: {type(metadata_filter)}')

    if metadata_filters.condition == FilterCondition.NOT:
        return f"(NOT {' '.join(filter_strs)})"
    elif (
        metadata_filters.condition == FilterCondition.AND
        or metadata_filters.condition == FilterCondition.OR
    ):
        condition = f' {metadata_filters.condition.value.upper()} '
        return f"({condition.join(filter_strs)})"
    else:
        raise ValueError(f'Unsupported filters condition: {metadata_filters.condition}')


def filter_config_to_opencypher_filters(filter_config: FilterConfig) -> str:
    """
    Converts a given FilterConfig object into a string of OpenCypher filters.
    Processes the provided filter configuration and generates a recursive representation
    of the metadata filters in OpenCypher format.

    :param filter_config: Input filter configuration object containing the necessary
        source filters for translation.
    :type filter_config: FilterConfig
    :return: A string representing the OpenCypher filters, or an empty string if the
        provided filter_config is None or contains no source filters.
    :rtype: str
    """
    if filter_config is None or filter_config.source_filters is None:
        return ''
    return parse_metadata_filters_recursive(filter_config.source_filters)


class NeptuneAnalyticsVectorIndexFactory(VectorIndexFactoryMethod):
    """
    Implements a factory for creating Neptune Analytics vector indices.

    This class provides a method to attempt the creation of Neptune Analytics vector
    indices based on a given configuration and list of index names. It examines the
    provided vector index information to determine if it matches the Neptune Analytics
    configuration format and initializes the indices accordingly.

    :ivar NEPTUNE_ANALYTICS: The prefix used to identify Neptune Analytics configurations.
    :type NEPTUNE_ANALYTICS: str
    :ivar logger: Logger instance for tracking the creation process and debugging details.
    :type logger: logging.Logger
    """
    def try_create(
        self, index_names: List[str], vector_index_info: str, **kwargs
    ) -> List[VectorIndex]:
        """
        Attempts to create a list of vector indexes based on the provided index names
        and vector index information. If the vector index information starts with a
        specific prefix (NEPTUNE_ANALYTICS), a graph ID is extracted, and the relevant
        NeptuneIndexes are opened and returned. If the prefix is not found, None is
        returned.

        :param index_names: A list of index names to be created.
        :type index_names: List[str]
        :param vector_index_info: A string containing information about the vector
            index. This may include a specific prefix (NEPTUNE_ANALYTICS) that
            identifies the type of index.
        :param kwargs: Additional keyword argument parameters passed to the creation
            of the vector indexes.
        :return: A list of vector indexes created for the specified index names, or
            None if the provided vector index information does not meet the expected
            criteria.
        :rtype: List[VectorIndex] or None
        """
        graph_id = None
        if vector_index_info.startswith(NEPTUNE_ANALYTICS):
            graph_id = vector_index_info[len(NEPTUNE_ANALYTICS) :]
            logger.debug(
                f'Opening Neptune Analytics vector indexes [index_names: {index_names}, graph_id: {graph_id}]'
            )
            return [
                NeptuneIndex.for_index(index_name, vector_index_info, **kwargs)
                for index_name in index_names
            ]
        else:
            return None


class NeptuneIndex(VectorIndex):
    """
    Represents an index in a Neptune graph database that allows for efficient
    querying, embedding, and retrieval of graph node data.

    This class provides functionality to create, configure, and interact with
    indexes in a Neptune graph database. It supports embedding models for
    generating vector embeddings for nodes, fetching top-k results based on
    similarity scores, and retrieving embeddings based on node IDs. The class
    is particularly useful for tasks requiring graph-based data retrieval
    with embedding-based searches and filtering capabilities.

    :ivar neptune_client: The underlying Neptune analytics client instance used
        for executing queries and interacting with the database.
    :type neptune_client: NeptuneAnalyticsClient
    :ivar embed_model: The embedding model utilized to compute vector embeddings.
    :type embed_model: Any
    :ivar dimensions: The dimensionality of the embeddings generated by the
        embedding model.
    :type dimensions: int
    :ivar id_name: The identifier name for nodes stored in the indexed graph.
    :type id_name: str
    :ivar label: Label or tag associated with the nodes in the graph.
    :type label: str
    :ivar path: The Cypher query path used to navigate through the graph.
    :type path: str
    :ivar return_fields: The string representation of node fields to be included
        in the query responses.
    :type return_fields: str
    """

    @staticmethod
    def for_index(index_name, graph_id, embed_model=None, dimensions=None, **kwargs):
        """
        Provides a static method to create a specialized `NeptuneIndex` object tailored for
        querying specific indices in a graph database using NeptuneDB. This method allows
        configuration of the index type, associated graph, embedding model, and dimensions,
        while determining the appropriate structure and fields for the query based on the
        index type.

        The method accepts parameters to define the index name, the graph identifier,
        embedding model details, and any additional keyword arguments necessary for the
        graph store connection. Depending on the index type, it builds the query path and
        return fields dynamically, with specialized cases for certain well-known index names.
        An error is raised for unsupported or invalid index names.

        :param index_name: The name of the index to query (e.g., 'chunk', 'statement', 'topic'),
            which also determines the type of entity being queried.
        :type index_name: str
        :param graph_id: The unique identifier for the graph within the database.
        :type graph_id: str
        :param embed_model: Optional. The embedding model to use for indexing. If not provided,
            the default model from the `GraphRAGConfig` is used.
        :type embed_model: Optional[Any]
        :param dimensions: Optional. The dimensions of the embedding vector space. If not
            provided, the default dimensions from the `GraphRAGConfig` are used.
        :type dimensions: Optional[int]
        :param kwargs: Additional keyword arguments needed for constructing the graph store client.
        :type kwargs: dict
        :return: A `NeptuneIndex` instance configured for the specified index type and graph ID.
        :rtype: NeptuneIndex
        """
        index_name = index_name.lower()
        neptune_client: GraphStore = GraphStoreFactory.for_graph_store(
            graph_id, **kwargs
        )
        embed_model = embed_model or GraphRAGConfig.embed_model
        dimensions = dimensions or GraphRAGConfig.embed_dimensions
        id_name = f'{index_name}Id'
        label = f'__{string.capwords(index_name)}__'
        path = f'({index_name})'
        return_fields = node_result(
            index_name, neptune_client.node_id(f'{index_name}.{id_name}')
        )

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
            return_fields=return_fields,
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
        Returns the Neptune database client instance for the tenant. This method checks
        whether the tenant is a default tenant or a specific one and provides the
        corresponding Neptune client.

        If the tenant is the default, it directly returns the configured `neptune_client`.
        Otherwise, it wraps the `neptune_client` for multi-tenant support using
        `MultiTenantGraphStore`.

        :return: The Neptune database client, either directly or wrapped for the
            corresponding tenant.
        :rtype: NeptuneClient or MultiTenantGraphStore
        """
        if self.tenant_id.is_default_tenant():
            return self.neptune_client
        else:
            return MultiTenantGraphStore.wrap(
                self.neptune_client, tenant_id=self.tenant_id
            )

    def add_embeddings(self, nodes):
        """
        Adds embeddings to provided nodes and updates the Neptune database with the
        embeddings using a specific query. The method assumes that the class is
        working with a Neptune database setup and uses an embedding model to process
        nodes.

        The process involves:
        1. Adding metadata to the nodes.
        2. Generating embeddings using the embedding model.
        3. Executing a Cypher query for each node to update the Neptune database
           with the embedding information.
        4. Cleaning up metadata after processing is complete.

        :param nodes: List of nodes to which embeddings will be added. Each node should be
            structured and compatible with the embedding model and Neptune database setup.
        :type nodes: list
        :return: The list of nodes with their information potentially updated
            after processing.
        :rtype: list
        """
        for node in nodes:
            node.metadata['index'] = self.underlying_index_name()

        id_to_embed_map = embed_nodes(nodes, self.embed_model)

        for node in nodes:

            statement = f"MATCH (n:`{self.label}`) WHERE {self.neptune_client.node_id('n.{self.id_name}')} = $nodeId"

            embedding = id_to_embed_map[node.node_id]

            query = '\n'.join(
                [
                    statement,
                    f'WITH n CALL neptune.algo.vectors.upsert(n, {embedding}) YIELD success RETURN success',
                ]
            )

            properties = {'nodeId': node.node_id, 'embedding': embedding}

            self._neptune_client().execute_query_with_retry(query, properties)

            node.metadata.pop('index', None)

        return nodes

    def top_k(
        self,
        query_bundle: QueryBundle,
        top_k: int = 5,
        filter_config: Optional[FilterConfig] = None,
    ):
        """
        Executes a top-K query against a graph database using a given query embedding. The query retrieves
        the top-K results based on similarity scores computed from vector embeddings. The function can
        filter results using a `FilterConfig` object, modify the query context with a tenant-specific
        label, and limit the results based on the top-K value.

        :param query_bundle: The query bundle that includes query string and embeddings.
        :type query_bundle: QueryBundle
        :param top_k: The number of top results to retrieve. Defaults to 5.
        :type top_k: int
        :param filter_config: An optional filter configuration to restrict query results. Defaults to None.
        :type filter_config: Optional[FilterConfig]
        :return: A list of top-K results, with each result including a score and specified return fields.
        :rtype: List[Dict[str, Any]]
        """
        query_str = f'''index: {self.underlying_index_name()}

{query_bundle.query_str}
'''

        query_bundle = QueryBundle(query_str=query_str)
        query_bundle = to_embedded_query(query_bundle, self.embed_model)

        tenant_specific_label = self.tenant_id.format_label(self.label).replace('`', '')

        where_clause = filter_config_to_opencypher_filters(filter_config)
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

    def get_embeddings(self, ids: List[str] = []):
        """
        Fetches embeddings for the given IDs by executing cypher queries against a
        Neptune database. The function calculates embeddings for specified nodes,
        filters them based on the tenant-specific label, and matches the defined
        path to retrieve associated data.

        :param ids: List of unique node IDs for which embeddings are retrieved.
        :type ids: List[str]
        :return: List of dictionaries containing embeddings and additional fields
                 returned by the query.
        :rtype: List[dict]
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

            params = {'elementId': i}

            results = self._neptune_client().execute_query(cypher, params)

            for result in results:
                all_results.append(result['result'])

        return all_results
