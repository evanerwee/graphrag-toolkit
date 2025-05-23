# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import yaml
import logging
import time
from json2xml import json2xml
from typing import Optional, List, Type, Union

from graphrag_toolkit.lexical_graph.metadata import FilterConfig
from graphrag_toolkit.lexical_graph.tenant_id import TenantIdType, to_tenant_id
from graphrag_toolkit.lexical_graph.config import GraphRAGConfig
from graphrag_toolkit.lexical_graph.utils import LLMCache, LLMCacheType
from graphrag_toolkit.lexical_graph.retrieval.prompts import (
    ANSWER_QUESTION_SYSTEM_PROMPT,
    ANSWER_QUESTION_USER_PROMPT,
)
from graphrag_toolkit.lexical_graph.retrieval.post_processors.bedrock_context_format import (
    BedrockContextFormat,
)
from graphrag_toolkit.lexical_graph.retrieval.retrievers import (
    CompositeTraversalBasedRetriever,
    SemanticGuidedRetriever,
)
from graphrag_toolkit.lexical_graph.retrieval.retrievers import (
    StatementCosineSimilaritySearch,
    KeywordRankingSearch,
    SemanticBeamGraphSearch,
)
from graphrag_toolkit.lexical_graph.retrieval.retrievers import (
    WeightedTraversalBasedRetrieverType,
    SemanticGuidedRetrieverType,
)
from graphrag_toolkit.lexical_graph.storage import GraphStoreFactory, GraphStoreType
from graphrag_toolkit.lexical_graph.storage import VectorStoreFactory, VectorStoreType
from graphrag_toolkit.lexical_graph.storage.graph import MultiTenantGraphStore
from graphrag_toolkit.lexical_graph.storage.vector import (
    MultiTenantVectorStore,
    ReadOnlyVectorStore,
)
from graphrag_toolkit.lexical_graph.storage.vector import to_embedded_query

from llama_index.core import ChatPromptTemplate
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.schema import QueryBundle, NodeWithScore
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.base.response.schema import RESPONSE_TYPE
from llama_index.core.base.response.schema import Response
from llama_index.core.prompts.mixin import PromptDictType, PromptMixinType

logger = logging.getLogger(__name__)

RetrieverType = Union[BaseRetriever, Type[BaseRetriever]]
PostProcessorsType = Union[BaseNodePostprocessor, List[BaseNodePostprocessor]]


class LexicalGraphQueryEngine(BaseQueryEngine):
    """
    Manages query execution over a graph and vector store for multi-tenant setups,
    supporting traversal-based and semantic-guided search. This class enables
    customization with retrievers, post-processors, and filtering configurations
    to enhance search accuracy and flexibility.

    This engine facilitates two distinct search strategies:
    1. Traversal-based search: Focused on graph traversal techniques combined
       with weighted retriever logic to prioritize paths.
    2. Semantic-guided search: Designed to leverage semantic similarity,
       contextual ranking, and complex multi-layered searches.

    Attributes can be customized or extended to suit various tenant-specific
    requirements while maintaining compatibility with multi-tenant operations.

    :ivar graph_store: Multi-tenant compatible storage backend for graphs.
    :type graph_store: GraphStoreType
    :ivar vector_store: Multi-tenant compatible storage backend for vectors.
    :type vector_store: VectorStoreType
    :ivar tenant_id: Identifier for multi-tenant usage to differentiate stored data.
    :type tenant_id: Optional[TenantIdType]
    :ivar llm: Optional language model configuration for query enhancement.
    :type llm: Optional[LLMCacheType]
    :ivar system_prompt: Optional system metadata prompt used in chat templates.
    :type system_prompt: Optional[str]
    :ivar user_prompt: Optional user metadata prompt used in chat templates.
    :type user_prompt: Optional[str]
    :ivar retriever: Configured retriever instance for executing queries.
    :type retriever: Optional[RetrieverType]
    :ivar post_processors: Collection of post-processors for refining query results.
    :type post_processors: Optional[PostProcessorsType]
    :ivar callback_manager: Handles callbacks during query execution processes.
    :type callback_manager: Optional[CallbackManager]
    :ivar filter_config: Configuration for result filtering during search.
    :type filter_config: FilterConfig
    """

    @staticmethod
    def for_traversal_based_search(
        graph_store: GraphStoreType,
        vector_store: VectorStoreType,
        tenant_id: Optional[TenantIdType] = None,
        retrievers: Optional[List[WeightedTraversalBasedRetrieverType]] = None,
        post_processors: Optional[PostProcessorsType] = None,
        filter_config: FilterConfig = None,
        **kwargs,
    ):
        """
        Static method to create a LexicalGraphQueryEngine for traversal-based search.
        This method configures a multi-tenant graph and vector store, and initializes
        a composite traversal-based retriever. It encapsulates the setup process for
        performing lexical graph queries based on traversal search logic.

        :param graph_store: An instance of a graph store to be used for handling
            graph-based data queries and operations.
        :type graph_store: GraphStoreType
        :param vector_store: An instance of a vector store to manage and retrieve
            vector embeddings for the search.
        :type vector_store: VectorStoreType
        :param tenant_id: An optional identifier for the tenant, which allows
            multi-tenant support. If `None`, a default tenant configuration will
            be applied.
        :type tenant_id: Optional[TenantIdType]
        :param retrievers: A list of retrievers with weights to perform data retrieval
            based on traversal logic. If not provided, defaults to `None`.
        :type retrievers: Optional[List[WeightedTraversalBasedRetrieverType]]
        :param post_processors: An optional collection of post-processors to handle
            post-query processing or transformations. Defaults to `None` if not
            specified.
        :type post_processors: Optional[PostProcessorsType]
        :param filter_config: A configuration object for applying filters during
            the query process. Uses a default `FilterConfig` if not provided.
        :type filter_config: FilterConfig
        :param kwargs: Additional keyword arguments to customize and extend the
            initialization or processing behavior.
        :type kwargs: dict
        :return: An instance of `LexicalGraphQueryEngine` configured for traversal-based
            search with the given graph and vector store, retriever, filter settings,
            and customization options.
        :rtype: LexicalGraphQueryEngine
        """
        tenant_id = to_tenant_id(tenant_id)
        filter_config = filter_config or FilterConfig()

        graph_store = MultiTenantGraphStore.wrap(
            GraphStoreFactory.for_graph_store(graph_store), tenant_id
        )

        vector_store = ReadOnlyVectorStore.wrap(
            MultiTenantVectorStore.wrap(
                VectorStoreFactory.for_vector_store(vector_store), tenant_id
            )
        )

        retriever = CompositeTraversalBasedRetriever(
            graph_store,
            vector_store,
            retrievers=retrievers,
            filter_config=filter_config,
            **kwargs,
        )

        return LexicalGraphQueryEngine(
            graph_store,
            vector_store,
            tenant_id=tenant_id,
            retriever=retriever,
            post_processors=post_processors,
            context_format='text',
            filter_config=filter_config,
            **kwargs,
        )

    @staticmethod
    def for_semantic_guided_search(
        graph_store: GraphStoreType,
        vector_store: VectorStoreType,
        tenant_id: Optional[TenantIdType] = None,
        retrievers: Optional[List[SemanticGuidedRetrieverType]] = None,
        post_processors: Optional[PostProcessorsType] = None,
        filter_config: FilterConfig = None,
        **kwargs,
    ):
        """
        Factory method to create and configure a `LexicalGraphQueryEngine` instance for
        semantic guided search. This method sets up the graph store and vector store
        with tenant-specific configurations, initializes default retrievers for
        semantic and keyword-based search, and creates a `SemanticGuidedRetriever`
        instance to facilitate the interaction between stores and retrievers.

        :param graph_store: The graph storage system to store and query data.
        :param vector_store: The vector storage system to perform vector-based operations.
        :param tenant_id: Optional identifier to isolate data and resources for a specific tenant.
        :param retrievers: Optional list of retriever instances for semantic-guided search.
        :param post_processors: Optional post-processing stages for refining query results.
        :param filter_config: Configuration to apply filters during the search process.
        :param kwargs: Additional keyword arguments for customization.
        :return: An instance of `LexicalGraphQueryEngine` configured for semantic guided search.
        """
        tenant_id = to_tenant_id(tenant_id)
        filter_config = filter_config or FilterConfig()

        graph_store = MultiTenantGraphStore.wrap(
            GraphStoreFactory.for_graph_store(graph_store), tenant_id
        )

        vector_store = ReadOnlyVectorStore.wrap(
            MultiTenantVectorStore.wrap(
                VectorStoreFactory.for_vector_store(vector_store), tenant_id
            )
        )

        retrievers = retrievers or [
            StatementCosineSimilaritySearch(
                vector_store=vector_store,
                graph_store=graph_store,
                top_k=50,
                filter_config=filter_config,
            ),
            KeywordRankingSearch(
                vector_store=vector_store,
                graph_store=graph_store,
                max_keywords=10,
                filter_config=filter_config,
            ),
            SemanticBeamGraphSearch(
                vector_store=vector_store,
                graph_store=graph_store,
                max_depth=8,
                beam_width=100,
                filter_config=filter_config,
            ),
        ]

        retriever = SemanticGuidedRetriever(
            vector_store=vector_store,
            graph_store=graph_store,
            retrievers=retrievers,
            share_results=True,
            filter_config=filter_config,
            **kwargs,
        )

        return LexicalGraphQueryEngine(
            graph_store,
            vector_store,
            tenant_id=tenant_id,
            retriever=retriever,
            post_processors=post_processors,
            context_format='bedrock_xml',
            filter_config=filter_config,
            **kwargs,
        )

    def __init__(
        self,
        graph_store: GraphStoreType,
        vector_store: VectorStoreType,
        tenant_id: Optional[TenantIdType] = None,
        llm: LLMCacheType = None,
        system_prompt: Optional[str] = ANSWER_QUESTION_SYSTEM_PROMPT,
        user_prompt: Optional[str] = ANSWER_QUESTION_USER_PROMPT,
        retriever: Optional[RetrieverType] = None,
        post_processors: Optional[PostProcessorsType] = None,
        callback_manager: Optional[CallbackManager] = None,
        filter_config: FilterConfig = None,
        **kwargs,
    ):
        """
        This initializer sets up the necessary components for a service or system that
        involves interaction with graph databases, vector stores, and other dependencies
        to offer enriched capabilities like retrieval, processing, and customizable prompts.
        It initializes the required stores, large language model interface, prompt templates,
        retriever, and post-processing configurations. Additional optional configurations,
        such as callback management and filtering criteria, can also be provided.

        :param graph_store: An object representing the underlying graph storage solution.
        :param vector_store: A vector-based storage system for retrieval or indexing.
        :param tenant_id: An optional identifier for tenants to support multi-tenancy.
        :param llm: A large language model or its cache, used for generating responses.
        :param system_prompt: A predefined system prompt for conversational contexts.
        :param user_prompt: A predefined user prompt for conversational contexts.
        :param retriever: Optional object responsible for data or document retrieval.
        :param post_processors: Optional list or single post-processor for additional
            transformations.
        :param callback_manager: Optional callback management interface for handling asynchronous
            or event-based actions.
        :param filter_config: Configuration parameters specifying filtering options for retrieval.
        :param kwargs: Additional keyword arguments for custom configurations or extensions.
        """
        tenant_id = to_tenant_id(tenant_id)

        graph_store = MultiTenantGraphStore.wrap(
            GraphStoreFactory.for_graph_store(graph_store), tenant_id
        )
        vector_store = ReadOnlyVectorStore.wrap(
            MultiTenantVectorStore.wrap(
                VectorStoreFactory.for_vector_store(vector_store), tenant_id
            )
        )

        self.context_format = kwargs.get('context_format', 'json')

        self.llm = (
            llm
            if llm and isinstance(llm, LLMCache)
            else LLMCache(
                llm=llm or GraphRAGConfig.response_llm,
                enable_cache=GraphRAGConfig.enable_cache,
            )
        )
        self.chat_template = ChatPromptTemplate(
            message_templates=[
                ChatMessage(role=MessageRole.SYSTEM, content=system_prompt),
                ChatMessage(role=MessageRole.USER, content=user_prompt),
            ]
        )

        if retriever:
            if isinstance(retriever, BaseRetriever):
                self.retriever = retriever
            else:
                self.retriever = retriever(
                    graph_store, vector_store, filter_config=filter_config, **kwargs
                )
        else:
            self.retriever = CompositeTraversalBasedRetriever(
                graph_store, vector_store, filter_config=filter_config, **kwargs
            )

        if post_processors:
            self.post_processors = (
                post_processors
                if isinstance(post_processors, list)
                else [post_processors]
            )
        else:
            self.post_processors = []

        if self.context_format == 'bedrock_xml':
            self.post_processors.append(BedrockContextFormat())

        if callback_manager:
            for post_processor in self.post_processors:
                post_processor.callback_manager = callback_manager

        super().__init__(callback_manager)

    def _generate_response(self, query_bundle: QueryBundle, context: str) -> str:
        """
        Generates a response based on the provided query and context using the language
        model's prediction capabilities. The function uses a predefined chat template
        to construct the appropriate prompt and invokes the language model to generate
        a relevant response. In case of an error during the prediction process, it logs
        the exception details and re-raises the error for further handling.

        :param query_bundle: Contains the query string and additional query-related
                             information needed for response generation.
        :type query_bundle: QueryBundle
        :param context: Contextual information or prior search results required to
                        generate an informed response.
        :type context: str
        :return: The generated response from the language model based on the input query
                 and contextual information.
        :rtype: str
        """
        try:
            response = self.llm.predict(
                prompt=self.chat_template,
                query=query_bundle.query_str,
                search_results=context,
            )
            return response
        except Exception:
            logger.exception(
                f'Error answering query [query: {query_bundle.query_str}, context: {context}]'
            )
            raise

    def _format_as_text(self, json_results):
        """
        Formats a list of JSON objects into a summary text block. Each JSON object is
        expected to contain information about a specific topic, including related
        statements and a source. The generated text contains a heading for the topic,
        a concatenated list of its statements, and the source.

        :param json_results: List of dictionaries, where each dictionary represents
            a structured topic with keys 'topic' (str), 'statements' (list of str),
            and 'source' (str).
        :return: Formatted string that compiles the topics, their statements, and
            their sources into a single block of text.
        """
        lines = []
        for json_result in json_results:
            lines.append(f"""## {json_result['topic']}""")
            lines.append(' '.join([s for s in json_result['statements']]))
            lines.append(f"""[Source: {json_result['source']}]""")
            lines.append('\n')
        return '\n'.join(lines)

    def _format_context(
        self, search_results: List[NodeWithScore], context_format: str = 'json'
    ):
        """
        Formats the given search results into the specified context format.

        This method takes a list of search results and formats them into one of
        several supported formats such as JSON, YAML, XML, or plain text. By
        default, it formats the search results into JSON. If the specified context
        format is 'bedrock_xml', it concatenates the plain text of each result,
        separated by a newline character.

        :param search_results: A list of search results where each result is an
            instance of NodeWithScore containing details about the result.
        :type search_results: List[NodeWithScore]
        :param context_format: The format the search results should be formatted
            into. Supported formats are 'json', 'yaml', 'xml', 'text', and
            'bedrock_xml'. Default is 'json'.
        :type context_format: str
        :return: A formatted string representation of the search results in the
            specified format.
        :rtype: str
        """
        if context_format == 'bedrock_xml':
            return '\n'.join([result.text for result in search_results])

        json_results = [json.loads(result.text) for result in search_results]

        data = None

        if context_format == 'yaml':
            data = yaml.dump(json_results, sort_keys=False)
        elif context_format == 'xml':
            data = json2xml.Json2xml(json_results, attr_type=False).to_xml()
        elif context_format == 'text':
            data = self._format_as_text(json_results)
        else:
            data = json.dumps(json_results, indent=2)

        return data

    def retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """
        Retrieves a list of nodes with scores based on the given query bundle. The method
        applies a sequence of retrieval and optional post-processing steps to generate
        the final results.

        :param query_bundle: The query bundle used to retrieve information. It can be a
            string or a pre-constructed instance of the QueryBundle class.
        :type query_bundle: QueryBundle
        :return: A list of nodes paired with their respective scores, which represent
            the retrieved and processed results.
        :rtype: List[NodeWithScore]
        """
        query_bundle = (
            QueryBundle(query_bundle) if isinstance(query_bundle, str) else query_bundle
        )

        query_bundle = to_embedded_query(query_bundle, GraphRAGConfig.embed_model)

        results = self.retriever.retrieve(query_bundle)

        for post_processor in self.post_processors:
            results = post_processor.postprocess_nodes(results, query_bundle)

        return results

    def _query(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        """
        Executes a query processing workflow that involves embedding the query, retrieving relevant
        results, applying post-processing, and generating a response based on the processed context.

        :param query_bundle:
            An instance of `QueryBundle` that contains the query and any related metadata required
            for processing.
        :return:
            An instance of `RESPONSE_TYPE` containing the generated response, source nodes, and
            associated metadata.
        :raises Exception:
            Raised if an error occurs during any step of the query processing or response generation.
        """
        try:

            start = time.time()

            query_bundle = to_embedded_query(query_bundle, GraphRAGConfig.embed_model)

            results = self.retriever.retrieve(query_bundle)

            end_retrieve = time.time()

            for post_processor in self.post_processors:
                results = post_processor.postprocess_nodes(results, query_bundle)

            end_postprocessing = time.time()

            context = self._format_context(results, self.context_format)
            answer = self._generate_response(query_bundle, context)

            end = time.time()

            retrieve_ms = (end_retrieve - start) * 1000
            postprocess_ms = (end_postprocessing - end_retrieve) * 1000
            answer_ms = (end - end_retrieve) * 1000
            total_ms = (end - start) * 1000

            metadata = {
                'retrieve_ms': retrieve_ms,
                'postprocessing_ms': postprocess_ms,
                'answer_ms': answer_ms,
                'total_ms': total_ms,
                'context_format': self.context_format,
                'retriever': f'{type(self.retriever).__name__}: {self.retriever.__dict__}',
                'query': query_bundle.query_str,
                'postprocessors': [type(p).__name__ for p in self.post_processors],
                'context': context,
                'num_source_nodes': len(results),
            }

            return Response(response=answer, source_nodes=results, metadata=metadata)
        except Exception as e:
            logger.exception('Error in query processing')
            raise

    async def _aquery(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        """
        Execute an asynchronous query operation using a provided query bundle.

        :param query_bundle: The collection of data and queries encapsulated
            in a QueryBundle object.
        :type query_bundle: QueryBundle
        :return: The result of the query operation, adhering to the specified
            RESPONSE_TYPE.
        :rtype: RESPONSE_TYPE
        """
        pass

    def _get_prompts(self) -> PromptDictType:
        """
        Retrieves the dictionary containing prompts.

        This method is responsible for returning a dictionary
        structured as prompt keys mapped to their specific
        text content or configurations. The type structure of
        the returned dictionary is predefined.

        :return: Dictionary containing prompts.
        :rtype: PromptDictType
        """
        pass

    def _get_prompt_modules(self) -> PromptMixinType:
        """
        Retrieves the prompt modules associated with the current object.

        This method is intended for internal use and fetches a specified type of
        prompt modules. The functionality may be utilized as part of processing
        or generation pipelines where prompts are required.

        :return: The prompt modules associated with this object.
        :rtype: PromptMixinType
        """
        pass

    def _update_prompts(self, prompts_dict: PromptDictType) -> None:
        """
        Updates the internal state of prompts by applying the provided dictionary of prompts.

        This method modifies the existing prompts stored in the instance with the values
        from the provided dictionary. The update may override the previous entries or
        add new ones based on the keys in the provided dictionary.

        :param prompts_dict: A dictionary containing prompt keys and their corresponding values.
        :type prompts_dict: PromptDictType
        """
        pass
