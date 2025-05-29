# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import List, Optional, Union, Any
from pipe import Pipe

from graphrag_toolkit.lexical_graph.tenant_id import (
    TenantId,
    TenantIdType,
    DEFAULT_TENANT_ID,
    to_tenant_id,
)
from graphrag_toolkit.lexical_graph.metadata import (
    FilterConfig,
    SourceMetadataFormatter,
    DefaultSourceMetadataFormatter,
    MetadataFiltersType,
)
from graphrag_toolkit.lexical_graph.storage import GraphStoreFactory, GraphStoreType
from graphrag_toolkit.lexical_graph.storage import VectorStoreFactory, VectorStoreType
from graphrag_toolkit.lexical_graph.storage.graph import MultiTenantGraphStore
from graphrag_toolkit.lexical_graph.storage.graph import DummyGraphStore
from graphrag_toolkit.lexical_graph.storage.vector import MultiTenantVectorStore
from graphrag_toolkit.lexical_graph.indexing.extract import BatchConfig
from graphrag_toolkit.lexical_graph.indexing import NodeHandler
from graphrag_toolkit.lexical_graph.indexing import sink
from graphrag_toolkit.lexical_graph.indexing.constants import (
    PROPOSITIONS_KEY,
    DEFAULT_ENTITY_CLASSIFICATIONS,
)
from graphrag_toolkit.lexical_graph.indexing.extract import (
    ScopedValueProvider,
    FixedScopedValueProvider,
    DEFAULT_SCOPE,
)
from graphrag_toolkit.lexical_graph.indexing.extract import GraphScopedValueStore
from graphrag_toolkit.lexical_graph.indexing.extract import (
    LLMPropositionExtractor,
    BatchLLMPropositionExtractor,
)
from graphrag_toolkit.lexical_graph.indexing.extract import (
    TopicExtractor,
    BatchTopicExtractor,
)
from graphrag_toolkit.lexical_graph.indexing.extract import ExtractionPipeline
from graphrag_toolkit.lexical_graph.indexing.extract import (
    InferClassifications,
    InferClassificationsConfig,
)
from graphrag_toolkit.lexical_graph.indexing.build import BuildPipeline
from graphrag_toolkit.lexical_graph.indexing.build import VectorIndexing
from graphrag_toolkit.lexical_graph.indexing.build import GraphConstruction
from graphrag_toolkit.lexical_graph.indexing.build import Checkpoint
from graphrag_toolkit.lexical_graph.indexing.build import BuildFilters
from graphrag_toolkit.lexical_graph.indexing.build.null_builder import NullBuilder

from llama_index.core.node_parser import SentenceSplitter, NodeParser
from llama_index.core.schema import BaseNode, NodeRelationship

DEFAULT_EXTRACTION_DIR = 'output'

logger = logging.getLogger(__name__)


class ExtractionConfig:
    """
    Configuration for extraction processes.

    This class encapsulates configurations related to extraction processes, such
    as proposition extraction and entity classification. It provides settings
    to control various aspects of the extraction behavior, including prompts
    for proposition and topic extractions, entity classification inference, and
    filter configurations. The purpose is to allow users to customize the
    extraction process to their needs.

    :ivar enable_proposition_extraction: Determines whether proposition extraction
        is enabled in the extraction process.
    :type enable_proposition_extraction: bool
    :ivar preferred_entity_classifications: Defines a list of preferred entity
        classifications that guide the entity extraction process.
    :type preferred_entity_classifications: List[str]
    :ivar infer_entity_classifications: Specifies whether entity classifications
        should be automatically inferred. Can be configured with a specific
        configuration or as a boolean.
    :type infer_entity_classifications: Union[InferClassificationsConfig, bool]
    :ivar extract_propositions_prompt_template: A template for the prompt to be used
        when extracting propositions. This option allows custom prompts for a more
        customized extraction process.
    :type extract_propositions_prompt_template: Optional[str]
    :ivar extract_topics_prompt_template: A template for the prompt to be used when
        extracting topics, providing additional flexibility in topic extraction.
    :type extract_topics_prompt_template: Optional[str]
    :ivar extraction_filters: An optional configuration for metadata filters to
        refine the extraction process based on user-defined criteria.
    :type extraction_filters: Optional[MetadataFiltersType]
    """

    def __init__(
        self,
        enable_proposition_extraction: bool = True,
        preferred_entity_classifications: List[str] = DEFAULT_ENTITY_CLASSIFICATIONS,
        infer_entity_classifications: Union[InferClassificationsConfig, bool] = False,
        extract_propositions_prompt_template: Optional[str] = None,
        extract_topics_prompt_template: Optional[str] = None,
        extraction_filters: Optional[MetadataFiltersType] = None,
    ):
        self.enable_proposition_extraction = enable_proposition_extraction
        self.preferred_entity_classifications = preferred_entity_classifications
        self.infer_entity_classifications = infer_entity_classifications
        self.extract_propositions_prompt_template = extract_propositions_prompt_template
        self.extract_topics_prompt_template = extract_topics_prompt_template
        self.extraction_filters = FilterConfig(extraction_filters)


class BuildConfig:
    """
    Encapsulates configuration settings for a build process.

    Provides options to use build filters, include domain labels, and format
    source metadata. This class is designed to streamline and customize the
    build process with optional parameters and default values.

    :ivar build_filters: The filters applied during the build process.
        Defaults to an instance of BuildFilters.
    :type build_filters: Optional[BuildFilters]
    :ivar include_domain_labels: Indicates whether domain labels are included in
        the build output. Defaults to False.
    :type include_domain_labels: Optional[bool]
    :ivar source_metadata_formatter: Handles formatting of source metadata used
        in the build process. Defaults to an instance of
        DefaultSourceMetadataFormatter.
    :type source_metadata_formatter: Optional[SourceMetadataFormatter]
    """

    def __init__(
        self,
        build_filters: Optional[BuildFilters] = None,
        include_domain_labels: Optional[bool] = None,
        source_metadata_formatter: Optional[SourceMetadataFormatter] = None,
    ):
        """
        Initializes an instance of the class, allowing for optional customization of filter
        construction, domain labeling, and metadata formatting. These attributes enable tailored
        behavior of the object based on provided settings or defaults.

        :param build_filters: An optional instance of ``BuildFilters`` used for customizing
            filter creation. Defaults to a new instance of ``BuildFilters`` if not supplied.
        :param include_domain_labels: An optional boolean flag indicating whether domain
            labels should be included. Defaults to ``False`` if not provided.
        :param source_metadata_formatter: An optional instance of ``SourceMetadataFormatter``
            used to format metadata from the source. Defaults to an instance of
            ``DefaultSourceMetadataFormatter`` if not provided.
        """
        self.build_filters = build_filters or BuildFilters()
        self.include_domain_labels = include_domain_labels or False
        self.source_metadata_formatter = (
            source_metadata_formatter or DefaultSourceMetadataFormatter()
        )


class IndexingConfig:
    """
    Configuration class for data processing, including chunking, extraction,
    building, and batch processing.

    This class stores configurations to control how data is handled in terms
    of splitting into smaller segments, extracting relevant information,
    building processed entities, and managing batch operations during an
    indexing process.

    :ivar chunking: List of node parsers used for text chunking. If None,
        chunking is disabled. An empty list appends a default `SentenceSplitter`.
    :ivar extraction: Configuration for extracting specific information
        during processing. Defaults to a new `ExtractionConfig` instance if
        not provided.
    :ivar build: Configuration for constructing processed entities. Defaults
        to a new `BuildConfig` instance if not set.
    :ivar batch_config: Configuration for handling batch inference
        operations. If None, batch inference is not utilized.
    """

    def __init__(
        self,
        chunking: Optional[List[NodeParser]] = [],
        extraction: Optional[ExtractionConfig] = None,
        build: Optional[BuildConfig] = None,
        batch_config: Optional[BatchConfig] = None,
    ):
        """
        Initialize the class instance with configurations for various processing stages. This includes
        chunking of input data into segments, extraction of essential information, building the final
        output, and configuring batch processing. The default configurations are provided if none
        are specified during initialization.

        :param chunking: A list of `NodeParser` instances used for breaking input into chunks. If not
            provided, a default `SentenceSplitter` instance is added with predefined configurations.
        :type chunking: Optional[List[NodeParser]]

        :param extraction: Configuration for the extraction process. If not provided, a default
            `ExtractionConfig` instance is used.
        :type extraction: Optional[ExtractionConfig]

        :param build: Configuration for building final output. If not provided, a default
            `BuildConfig` instance is used.
        :type build: Optional[BuildConfig]

        :param batch_config: Configuration for batch processing. If not specified, batch inference
            is disabled.
        :type batch_config: Optional[BatchConfig]
        """
        if chunking is not None and len(chunking) == 0:
            chunking.append(SentenceSplitter(chunk_size=256, chunk_overlap=20))

        self.chunking = chunking  # None = no chunking
        self.extraction = extraction or ExtractionConfig()
        self.build = build or BuildConfig()
        self.batch_config = batch_config  # None = do not use batch inference


def get_topic_scope(node: BaseNode):
    """
    Determines and retrieves the topic scope based on a node's relationships. If the node has a
    relationship of type SOURCE, the topic scope is derived from the corresponding source node's
    identifier. Otherwise, a default topic scope is returned.

    :param node: The input object of type BaseNode whose relationships are used to
        determine the topic scope.
    :return: The topic scope corresponding to the node, either derived from a SOURCE
        relationship or the default topic scope.
    :rtype: str
    """
    source = node.relationships.get(NodeRelationship.SOURCE, None)
    if not source:
        return DEFAULT_SCOPE
    else:
        return source.node_id


IndexingConfigType = Union[
    IndexingConfig, ExtractionConfig, BuildConfig, BatchConfig, List[NodeParser]
]


def to_indexing_config(
    indexing_config: Optional[IndexingConfigType] = None,
) -> IndexingConfig:
    """
    Converts the given indexing configuration or related configuration type
    to an IndexingConfig instance. Provides a central mechanism to handle
    various supported types and ensures compatibility with the IndexingConfig
    object. If no indexing configuration is provided, a default
    IndexingConfig object is created and returned.

    :param indexing_config: The input indexing configuration or related
        configuration type to be converted. Accepts None, IndexingConfig,
        ExtractionConfig, BuildConfig, BatchConfig, or a list of NodeParser
        instances.
    :type indexing_config: Optional[IndexingConfigType]

    :return: An instance of IndexingConfig based on the input configuration
        type. If the input is invalid, a ValueError will be raised.
    :rtype: IndexingConfig

    :raises ValueError: If the provided input is of an unsupported type or if
        the list items in the input are not all instances of NodeParser.
    """
    if not indexing_config:
        return IndexingConfig()
    if isinstance(indexing_config, IndexingConfig):
        return indexing_config
    elif isinstance(indexing_config, ExtractionConfig):
        return IndexingConfig(extraction=indexing_config)
    elif isinstance(indexing_config, BuildConfig):
        return IndexingConfig(build=indexing_config)
    elif isinstance(indexing_config, BatchConfig):
        return IndexingConfig(batch_config=indexing_config)
    elif isinstance(indexing_config, list):
        for np in indexing_config:
            if not isinstance(np, NodeParser):
                raise ValueError(f'Invalid indexing config type: {type(np)}')
        return IndexingConfig(chunking=indexing_config)
    else:
        raise ValueError(f'Invalid indexing config type: {type(indexing_config)}')


class LexicalGraphIndex:
    """
    A robust class designed to manage and process semantic graphs and vector
    stores for building index pipelines. This class facilitates the extraction,
    transformation, and storage of graph and vector representations, utilizing
    configurable components and processing pipelines tailored to organizational
    or project-specific needs.

    The class supports dynamic pipeline assembly for operations such as
    proposition and topic extraction, entity classification inference,
    chunking, and other related tasks. It integrates with tenant-specific data
    configurations and guarantees flexible processing options for different
    extraction and build requirements.

    :ivar graph_store: The graph store for managing semantic graphs. It supports
        dynamic tenant wrapping.
    :type graph_store: MultiTenantGraphStore
    :ivar vector_store: The vector store for managing vectorized representations
        of nodes and graphs, extended for multi-tenant handling.
    :type vector_store: MultiTenantVectorStore
    :ivar tenant_id: The tenant identifier to segregate data and processes
        between different clients or users.
    :type tenant_id: TenantId
    :ivar extraction_dir: The directory path for storing intermediate extraction
        results or configurations during pipeline operations.
    :type extraction_dir: str
    :ivar indexing_config: The configuration object defining all indexing and
        extraction parameters, including batch settings, chunking, and inference
        options.
    :type indexing_config: IndexingConfigType
    :ivar extraction_pre_processors: A list of preprocessing steps applied before
        the main extraction pipeline, such as batch classification inference.
    :type extraction_pre_processors: list
    :ivar extraction_components: The primary components of the extraction pipeline
        including proposition and topic extractors.
    :type extraction_components: list
    :ivar allow_batch_inference: A flag indicating whether batch inference is
        enabled based on the indexing configuration.
    :type allow_batch_inference: bool
    """

    def __init__(
        self,
        graph_store: Optional[GraphStoreType] = None,
        vector_store: Optional[VectorStoreType] = None,
        tenant_id: Optional[TenantIdType] = None,
        extraction_dir: Optional[str] = None,
        indexing_config: Optional[IndexingConfigType] = None,
    ):
        """
        This constructor initializes a class instance with multiple configurable stores,
        directories, and indexing configurations. It ensures that tenant-specific
        information is correctly wrapped and managed for both graph and vector stores.
        The function also initializes extraction-related processing components based on
        the specified configuration. Additionally, it determines whether batch
        inference is allowed based on the presence of a batch configuration.

        :param graph_store: The graph store instance used for managing structured
            data; if not provided, a default instance may be supplied.
        :type graph_store: Optional[GraphStoreType]
        :param vector_store: The vector store instance utilized for storing
            vectorized information; defaults to None if not specified.
        :type vector_store: Optional[VectorStoreType]
        :param tenant_id: An optional tenant identifier that ensures tenant-specific
            scoping is applied across the graph and vector stores.
        :type tenant_id: Optional[TenantIdType]
        :param extraction_dir: Directory used for extraction-related processing; if not
            specified, defaults to a predefined location.
        :type extraction_dir: Optional[str]
        :param indexing_config: Configuration object that dictates indexing strategies,
            including batching, pre-processing, and other extraction pipeline
            configurations.
        :type indexing_config: Optional[IndexingConfigType]
        """
        tenant_id = to_tenant_id(tenant_id)

        self.graph_store = MultiTenantGraphStore.wrap(
            GraphStoreFactory.for_graph_store(graph_store), tenant_id
        )
        self.vector_store = MultiTenantVectorStore.wrap(
            VectorStoreFactory.for_vector_store(vector_store), tenant_id
        )
        self.tenant_id = tenant_id or TenantId()
        self.extraction_dir = extraction_dir or DEFAULT_EXTRACTION_DIR
        self.indexing_config = to_indexing_config(indexing_config)

        (pre_processors, components) = self._configure_extraction_pipeline(
            self.indexing_config
        )

        self.extraction_pre_processors = pre_processors
        self.extraction_components = components
        self.allow_batch_inference = self.indexing_config.batch_config is not None

    def _configure_extraction_pipeline(self, config: IndexingConfig):
        """
        Configures the extraction pipeline for processing data. This method defines
        the necessary components and pre-processors based on the provided configuration.
        It sets up chunking, proposition extraction, and classification providers, and
        incorporates topic extraction capabilities. The pipeline dynamically adapts
        depending on the specifications in the `config` parameter.

        :param config: The configuration specifying chunking, extraction, and classification
            options for building the pipeline.
        :type config: IndexingConfig
        :return: A tuple containing the list of pre-processors and the list of components
            comprising the extraction pipeline.
        :rtype: tuple[list, list]
        """
        pre_processors = []
        components = []

        if config.chunking:
            for c in config.chunking:
                components.append(c)

        if config.extraction.enable_proposition_extraction:
            if config.batch_config:
                components.append(
                    BatchLLMPropositionExtractor(
                        batch_config=config.batch_config,
                        prompt_template=config.extraction.extract_propositions_prompt_template,
                    )
                )
            else:
                components.append(
                    LLMPropositionExtractor(
                        prompt_template=config.extraction.extract_propositions_prompt_template
                    )
                )

        entity_classification_provider = None
        topic_provider = None

        classification_label = 'EntityClassification'
        classification_scope = DEFAULT_SCOPE

        if isinstance(self.graph_store, DummyGraphStore):
            entity_classification_provider = FixedScopedValueProvider(
                scoped_values={
                    DEFAULT_SCOPE: config.extraction.preferred_entity_classifications
                }
            )
            topic_provider = FixedScopedValueProvider(scoped_values={DEFAULT_SCOPE: []})
        else:
            initial_scope_values = (
                []
                if config.extraction.infer_entity_classifications
                else config.extraction.preferred_entity_classifications
            )
            entity_classification_provider = ScopedValueProvider(
                label=classification_label,
                scoped_value_store=GraphScopedValueStore(graph_store=self.graph_store),
                initial_scoped_values={classification_scope: initial_scope_values},
            )
            topic_provider = ScopedValueProvider(
                label='StatementTopic',
                scoped_value_store=GraphScopedValueStore(graph_store=self.graph_store),
                scope_func=get_topic_scope,
            )

        if config.extraction.infer_entity_classifications:
            infer_config = (
                config.extraction.infer_entity_classifications
                if isinstance(
                    config.extraction.infer_entity_classifications,
                    InferClassificationsConfig,
                )
                else InferClassificationsConfig()
            )
            pre_processors.append(
                InferClassifications(
                    classification_label=classification_label,
                    classification_scope=classification_scope,
                    classification_store=GraphScopedValueStore(
                        graph_store=self.graph_store
                    ),
                    splitter=(
                        SentenceSplitter(chunk_size=256, chunk_overlap=20)
                        if config.chunking
                        else None
                    ),
                    default_classifications=config.extraction.preferred_entity_classifications,
                    num_samples=infer_config.num_samples,
                    num_iterations=infer_config.num_iterations,
                    merge_action=infer_config.on_existing_classifications,
                    prompt_template=infer_config.prompt_template,
                )
            )

        topic_extractor = None

        if config.batch_config:
            topic_extractor = BatchTopicExtractor(
                batch_config=config.batch_config,
                source_metadata_field=(
                    PROPOSITIONS_KEY
                    if config.extraction.enable_proposition_extraction
                    else None
                ),
                entity_classification_provider=entity_classification_provider,
                topic_provider=topic_provider,
                prompt_template=config.extraction.extract_topics_prompt_template,
            )
        else:
            topic_extractor = TopicExtractor(
                source_metadata_field=(
                    PROPOSITIONS_KEY
                    if config.extraction.enable_proposition_extraction
                    else None
                ),
                entity_classification_provider=entity_classification_provider,
                topic_provider=topic_provider,
                prompt_template=config.extraction.extract_topics_prompt_template,
            )

        components.append(topic_extractor)

        return (pre_processors, components)

    def extract(
        self,
        nodes: List[BaseNode] = [],
        handler: Optional[NodeHandler] = None,
        checkpoint: Optional[Checkpoint] = None,
        show_progress: Optional[bool] = False,
        **kwargs: Any,
    ) -> None:
        """
        Initiates and executes the extraction process for the provided nodes, applying the specified
        handler, checkpoint, and other optional arguments. This extraction process involves creating
        and configuring extraction and build pipelines with defined components and settings.

        :param nodes: The list of nodes to be processed during the extraction.
        :type nodes: List[BaseNode], optional
        :param handler: An optional NodeHandler to accept the processed nodes during extraction.
        :type handler: Optional[NodeHandler], optional
        :param checkpoint: An optional checkpoint object to manage the state during the pipeline execution.
        :type checkpoint: Optional[Checkpoint], optional
        :param show_progress: Indicates whether to display progress during the extraction process.
        :type show_progress: Optional[bool], optional
        :param kwargs: Additional keyword arguments to configure the extraction and build pipelines.
        :type kwargs: Any
        :return: None
        """

        if not self.tenant_id.is_default_tenant():
            logger.warning(
                'TenantId has been set to non-default tenant id, but extraction will use default tenant id'
            )

        extraction_pipeline = ExtractionPipeline.create(
            components=self.extraction_components,
            pre_processors=self.extraction_pre_processors,
            show_progress=show_progress,
            checkpoint=checkpoint,
            num_workers=1 if self.allow_batch_inference else None,
            tenant_id=DEFAULT_TENANT_ID,
            extraction_filters=self.indexing_config.extraction.extraction_filters,
            **kwargs,
        )

        build_pipeline = BuildPipeline.create(
            components=[NullBuilder()],
            show_progress=show_progress,
            checkpoint=checkpoint,
            num_workers=1,
            tenant_id=DEFAULT_TENANT_ID,
            **kwargs,
        )

        if handler:
            nodes | extraction_pipeline | Pipe(handler.accept) | build_pipeline | sink
        else:
            nodes | extraction_pipeline | build_pipeline | sink

    def build(
        self,
        nodes: List[BaseNode] = [],
        handler: Optional[NodeHandler] = None,
        checkpoint: Optional[Checkpoint] = None,
        show_progress: Optional[bool] = False,
        **kwargs: Any,
    ) -> None:
        """
        Builds a pipeline for indexing and constructs the necessary processing flow based on the provided
        nodes, handler, checkpoint, and additional parameters.

        :param nodes: List of BaseNode instances to be passed into the build pipeline.
        :type nodes: List[BaseNode]
        :param handler: An optional NodeHandler instance to process the pipeline output.
        :type handler: Optional[NodeHandler], default is None
        :param checkpoint: An optional Checkpoint object to allow resuming or saving progress.
        :type checkpoint: Optional[Checkpoint], default is None
        :param show_progress: If True, displays progress information during the build process.
        :type show_progress: Optional[bool], default is False
        :param kwargs: Additional parameters to customize the build flow, passed into the pipeline.
        :type kwargs: Any
        :return: No return value.
        :rtype: None
        """

        build_config = self.indexing_config.build

        build_pipeline = BuildPipeline.create(
            components=[
                GraphConstruction.for_graph_store(self.graph_store),
                VectorIndexing.for_vector_store(self.vector_store),
            ],
            show_progress=show_progress,
            checkpoint=checkpoint,
            build_filters=build_config.build_filters,
            source_metadata_formatter=build_config.source_metadata_formatter,
            include_domain_labels=build_config.include_domain_labels,
            tenant_id=self.tenant_id,
            **kwargs,
        )

        sink_fn = sink if not handler else Pipe(handler.accept)
        nodes | build_pipeline | sink_fn

    def extract_and_build(
        self,
        nodes: List[BaseNode] = [],
        handler: Optional[NodeHandler] = None,
        checkpoint: Optional[Checkpoint] = None,
        show_progress: Optional[bool] = False,
        **kwargs: Any,
    ) -> None:
        """
        Executes an extraction and building pipeline to process and index nodes. This method
        utilizes two primary pipelines: the extraction pipeline for raw data extraction
        and the build pipeline to construct graphs and vector indices. It integrates
        preprocessors, handlers, progress indicators, and supports checkpointing for both
        pipelines. This function is foundational in preparing and indexing data for retrieval
        systems.

        :param nodes: List of nodes to be processed. If not provided, an empty list is used.
        :type nodes: List[BaseNode]
        :param handler: Optional handler to process output or intermediate results.
        :type handler: Optional[NodeHandler]
        :param checkpoint: Optional checkpoint object to use for incremental progress saving.
        :type checkpoint: Optional[Checkpoint]
        :param show_progress: Flag indicating whether to display progress to the user.
        :type show_progress: Optional[bool]
        :param kwargs: Additional parameters for configuring pipelines and processing options.
        :type kwargs: Any
        :return: None
        """

        if not self.tenant_id.is_default_tenant():
            logger.warning(
                'TenantId has been set to non-default tenant id, but extraction will use default tenant id'
            )

        extraction_pipeline = ExtractionPipeline.create(
            components=self.extraction_components,
            pre_processors=self.extraction_pre_processors,
            show_progress=show_progress,
            checkpoint=checkpoint,
            num_workers=1 if self.allow_batch_inference else None,
            tenant_id=DEFAULT_TENANT_ID,
            extraction_filters=self.indexing_config.extraction.extraction_filters,
            **kwargs,
        )

        build_config = self.indexing_config.build

        build_pipeline = BuildPipeline.create(
            components=[
                GraphConstruction.for_graph_store(self.graph_store),
                VectorIndexing.for_vector_store(self.vector_store),
            ],
            show_progress=show_progress,
            checkpoint=checkpoint,
            build_filters=build_config.build_filters,
            source_metadata_formatter=build_config.source_metadata_formatter,
            include_domain_labels=build_config.include_domain_labels,
            tenant_id=self.tenant_id,
            **kwargs,
        )

        sink_fn = sink if not handler else Pipe(handler.accept)
        nodes | extraction_pipeline | build_pipeline | sink_fn
