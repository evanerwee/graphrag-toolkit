# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from pipe import Pipe
from typing import List, Optional, Sequence, Dict, Iterable, Any

from graphrag_toolkit.lexical_graph import TenantId
from graphrag_toolkit.lexical_graph.config import GraphRAGConfig
from graphrag_toolkit.lexical_graph.metadata import FilterConfig
from graphrag_toolkit.lexical_graph.indexing import IdGenerator
from graphrag_toolkit.lexical_graph.indexing.utils.pipeline_utils import run_pipeline
from graphrag_toolkit.lexical_graph.indexing.model import (
    SourceType,
    SourceDocument,
    source_documents_from_source_types,
)
from graphrag_toolkit.lexical_graph.indexing.extract.pipeline_decorator import (
    PipelineDecorator,
)
from graphrag_toolkit.lexical_graph.indexing.extract.source_doc_parser import (
    SourceDocParser,
)
from graphrag_toolkit.lexical_graph.indexing.build.checkpoint import Checkpoint
from graphrag_toolkit.lexical_graph.indexing.extract.docs_to_nodes import DocsToNodes
from graphrag_toolkit.lexical_graph.indexing.extract.id_rewriter import IdRewriter

from llama_index.core.node_parser import TextSplitter
from llama_index.core.utils import iter_batch
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.extractors.interface import BaseExtractor
from llama_index.core.schema import TransformComponent
from llama_index.core.schema import BaseNode, Document
from llama_index.core.schema import NodeRelationship

logger = logging.getLogger(__name__)


class PassThroughDecorator(PipelineDecorator):
    """
    A decorator class that processes input and output documents without modification.

    The PassThroughDecorator class is a subclass of PipelineDecorator designed to
    handle input and output documents by passing them through unchanged. It allows
    integration within a pipeline where no specific transformation or processing
    logic is required, enabling smooth data flow while adhering to a consistent
    decorator interface.
    """

    def __init__(self):
        """
        Represents the initialization method of a class.

        This method is a constructor used to initialize the attributes or settings of an
        object when an instance of a class is created. It does not perform any actions
        or set any attributes in its current implementation.

        Attributes:
            No attributes are defined in this method.

        Parameters:
            No parameters are expected in this method.

        Raises:
            This method does not raise any exceptions.

        Returns:
            None
        """
        pass

    def handle_input_docs(self, nodes: Iterable[SourceDocument]):
        """
        Handles the input documents provided as an iterable of SourceDocument objects.

        Takes an iterable of documents, processes or returns them as needed.

        :param nodes: An iterable containing instances of SourceDocument, representing
             the documents to be handled.
        :type nodes: Iterable[SourceDocument]
        :return: The input iterable of SourceDocument objects, processed as per the
            requirements.
        :rtype: Iterable[SourceDocument]
        """
        return nodes

    def handle_output_doc(self, node: SourceDocument) -> SourceDocument:
        """
        Handles and returns the given source document node.

        :param node: Source document node to be processed.
        :type node: SourceDocument
        :return: The processed source document node.
        :rtype: SourceDocument
        """
        return node


class ExtractionPipeline:
    """
    ExtractionPipeline class is designed to configure and execute a document extraction pipeline.

    This pipeline integrates multiple transformation components, optional preprocessing steps, and advanced configurations
    like tenant filtering, checkpointing, and data filtering. By enabling efficient processing with optional support
    for batching and parallel execution, this class serves as a robust solution for data extraction and transformation.

    :ivar ingestion_pipeline: The constructed pipeline consisting of all configured transformation
        components responsible for document processing.
    :type ingestion_pipeline: IngestionPipeline
    :ivar pre_processors: Pre-processors to parse and prepare source documents before ingestion into the pipeline.
    :type pre_processors: List[SourceDocParser]
    :ivar extraction_decorator: The decorator used to extend or customize the extraction process during pipeline execution.
    :type extraction_decorator: PipelineDecorator
    :ivar num_workers: Number of worker threads employed for parallel processing in the pipeline.
    :type num_workers: int
    :ivar batch_size: Number of processing items managed concurrently within each batch.
    :type batch_size: int
    :ivar show_progress: A flag indicating whether progress bars are enabled during extraction pipeline operations.
    :type show_progress: bool
    :ivar id_rewriter: A utility for generating unique traceable IDs for documents processed in the pipeline.
    :type id_rewriter: IdRewriter
    :ivar extraction_filters: Holds configuration settings for applying additional extraction rules and filters.
    :type extraction_filters: FilterConfig
    :ivar pipeline_kwargs: Captures additional pipeline-specific configurations supplied via keyword arguments.
    :type pipeline_kwargs: dict
    """

    @staticmethod
    def create(
        components: List[TransformComponent],
        pre_processors: Optional[List[SourceDocParser]] = None,
        extraction_decorator: PipelineDecorator = None,
        num_workers=None,
        batch_size=None,
        show_progress=False,
        checkpoint: Optional[Checkpoint] = None,
        tenant_id: Optional[TenantId] = None,
        extraction_filters: Optional[FilterConfig] = None,
        **kwargs: Any,
    ):
        """
        Creates a pipeline for data extraction by configuring the components, pre-processors,
        decorators, filters, and execution parameters. This method combines all the necessary
        elements required to execute the pipeline with the specified configuration.

        :param components: List of transformation components to be executed in the pipeline.
            These components determine the processing logic for data transformation.
        :type components: List[TransformComponent]
        :param pre_processors: Optional list of source document parsers for preprocessing
            steps prior to extraction. Defaults to None.
        :type pre_processors: Optional[List[SourceDocParser]]
        :param extraction_decorator: Optional decorator for the pipeline that encapsulates
            additional logic or functionality around extraction processes. Defaults to None.
        :type extraction_decorator: PipelineDecorator
        :param num_workers: Number of workers to utilize in parallel processing.
            If not provided, defaults to the system's logical processors.
        :type num_workers: Optional[int]
        :param batch_size: Size of each batch for batch processing. Helps manage
            processing efficiency and memory usage. Defaults to None.
        :type batch_size: Optional[int]
        :param show_progress: Boolean indicating whether to display a progress bar
            during extraction. Defaults to False.
        :type show_progress: bool
        :param checkpoint: Optional checkpoint configuration for resuming pipeline
            processing from a saved state. Defaults to None.
        :type checkpoint: Optional[Checkpoint]
        :param tenant_id: Identifier for the tenant, used to differentiate and handle
            multi-tenant data in configurations. Defaults to None.
        :type tenant_id: Optional[TenantId]
        :param extraction_filters: Optional filter configuration for refining or limiting
            the data extraction scope based on specified criteria. Defaults to None.
        :type extraction_filters: Optional[FilterConfig]
        :param kwargs: Additional keyword arguments for further customization of
            alternative configurations or extensions.
        :type kwargs: Any
        :return: A configured pipeline instance ready to execute the extraction process
            with the specified configurations.
        :rtype: Pipe
        """
        return Pipe(
            ExtractionPipeline(
                components=components,
                pre_processors=pre_processors,
                extraction_decorator=extraction_decorator,
                num_workers=num_workers,
                batch_size=batch_size,
                show_progress=show_progress,
                checkpoint=checkpoint,
                tenant_id=tenant_id,
                extraction_filters=extraction_filters,
                **kwargs,
            ).extract
        )

    def __init__(
        self,
        components: List[TransformComponent],
        pre_processors: Optional[List[SourceDocParser]] = None,
        extraction_decorator: PipelineDecorator = None,
        num_workers=None,
        batch_size=None,
        show_progress=False,
        checkpoint: Optional[Checkpoint] = None,
        tenant_id: Optional[TenantId] = None,
        extraction_filters: Optional[FilterConfig] = None,
        **kwargs: Any,
    ):
        """
        Initializes a pipeline with configurable transformation components, optional pre-processing,
        decorators, and advanced options like concurrency, batching, filters, and checkpoints. This pipeline
        can process input data through a series of transformations, with optional customization and resource
        management.

        :param components:
            List of transformation components to sequentially process the input data.
        :param pre_processors:
            Optional list of pre-processing components for parsing and preparing data before entering the
            main transformation pipeline.
        :param extraction_decorator:
            Decorator for adding extra functionality to the extraction process. Defaults to None.
        :param num_workers:
            Desired number of worker threads for concurrent processing. If not provided, a default value is
            derived from configuration.
        :param batch_size:
            Number of data items processed per batch. Uses a configuration default value if not specified.
        :param show_progress:
            Whether to display a progress bar or indicator during processing. Defaults to False.
        :param checkpoint:
            Optional configuration to allow checkpointing, enabling processing resumption or state
            management.
        :param tenant_id:
            Identifier to associate processing components with tenant-specific contexts. Optional.
        :param extraction_filters:
            Configuration for applying filters during the data extraction process. Defaults to None.
        :param kwargs:
            Additional arguments to customize the pipeline behavior or extend its functionality properly.
        """
        components = components or []
        num_workers = num_workers or GraphRAGConfig.extraction_num_workers
        batch_size = batch_size or GraphRAGConfig.extraction_batch_size

        for c in components:
            if isinstance(c, BaseExtractor):
                c.show_progress = show_progress

        def add_id_rewriter(c):
            """
            Represents an extraction pipeline to process and transform data using a series of
            components with optional preprocessing, parallel processing, and filtering
            capabilities.

            This class orchestrates data processing by leveraging its components in a
            specific sequence. Each component can modify the data or extract certain
            information. Optional pre-processors are applied before the pipeline execution.
            The pipeline also supports parallelism, batching, progress display, and
            additional filtering functionality.

            Attributes:
                components (List[TransformComponent]): Sequence of transformation components
                  to process the data.
                pre_processors (Optional[List[SourceDocParser]]): Optional list of preprocessing objects
                  to parse or modify the source data before transformation.
                extraction_decorator (PipelineDecorator): Optional decorator to control or alter the
                  behavior of the transformation pipeline.
                num_workers (Optional[int]): Number of parallel workers for execution.
                batch_size (Optional[int]): Number of items to process in each batch.
                show_progress (bool): Flag to indicate whether to display progress of pipeline execution.
                checkpoint (Optional[Checkpoint]): Mechanism for tracking or restoring pipeline state.
                tenant_id (Optional[TenantId]): Identifier for the current tenant or context for processing.
                extraction_filters (Optional[FilterConfig]): Configuration to filter results from
                  the pipeline.
                kwargs (Any): Additional optional parameters for fine-tuned configurations.

            :param components: Sequence of components to apply during the transformation process.
            :type components: List[TransformComponent]
            :param pre_processors: Optional preprocessing objects.
            :type pre_processors: Optional[List[SourceDocParser]]
            :param extraction_decorator: Optional decorator for the pipeline execution.
            :type extraction_decorator: PipelineDecorator
            :param num_workers: Number of parallel workers.
            :type num_workers: Optional[int]
            :param batch_size: Batch size for processing data.
            :type batch_size: Optional[int]
            :param show_progress: Indicates whether to show progress during execution.
            :type show_progress: bool
            :param checkpoint: Optional mechanism for checkpointing and restoring states.
            :type checkpoint: Optional[Checkpoint]
            :param tenant_id: Tenant identifier for providing context.
            :type tenant_id: Optional[TenantId]
            :param extraction_filters: Additional filtering configuration.
            :type extraction_filters: Optional[FilterConfig]
            :param kwargs: Additional arguments for extended functionality.
            :type kwargs: Any
            """
            if isinstance(c, TextSplitter):
                logger.debug(f'Wrapping {type(c).__name__} with IdRewriter')
                return IdRewriter(
                    inner=c, id_generator=IdGenerator(tenant_id=tenant_id)
                )
            else:
                return c

        components = [add_id_rewriter(c) for c in components]

        if not any([isinstance(c, IdRewriter) for c in components]):
            logger.debug(f'Adding DocToNodes to components')
            components.insert(
                0,
                IdRewriter(
                    inner=DocsToNodes(), id_generator=IdGenerator(tenant_id=tenant_id)
                ),
            )

        if checkpoint:
            components = [checkpoint.add_filter(c) for c in components]

        logger.debug(
            f'Extract pipeline components: {[type(c).__name__ for c in components]}'
        )

        self.ingestion_pipeline = IngestionPipeline(transformations=components)
        self.pre_processors = pre_processors or []
        self.extraction_decorator = extraction_decorator or PassThroughDecorator()
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.show_progress = show_progress
        self.id_rewriter = IdRewriter(id_generator=IdGenerator(tenant_id=tenant_id))
        self.extraction_filters = extraction_filters or FilterConfig()
        self.pipeline_kwargs = kwargs

    def _source_documents_from_base_nodes(
        self, nodes: Sequence[BaseNode]
    ) -> List[SourceDocument]:
        """
        Processes a sequence of BaseNode objects and organizes them into a list of
        SourceDocument objects based on their source relationships.

        The method iterates over a sequence of BaseNode objects, determines their
        source relationships, and groups them under corresponding SourceDocuments.
        Each unique source ID corresponds to a SourceDocument instance, which contains
        a collection of nodes belonging to that particular source.

        :param nodes:
            A sequence of BaseNode objects to be processed. Each node must contain
            source information in its `relationships` attribute.

        :return:
            A list of SourceDocument objects. Each object aggregates nodes that share
            the same source ID.
        """
        results: Dict[str, SourceDocument] = {}

        for node in nodes:
            source_info = node.relationships[NodeRelationship.SOURCE]
            source_id = source_info.node_id
            if source_id not in results:
                results[source_id] = SourceDocument()
            results[source_id].nodes.append(node)

        return list(results.values())

    def extract(self, inputs: Iterable[SourceType]):
        """
        Processes the specified inputs using a series of pre-processors, filters, and an
        extraction pipeline. The method extracts data by transforming input source documents
        into nodes, filtering those nodes based on metadata, running them through an
        extraction pipeline, and post-processing the results.

        :param inputs: A collection of input source types to be processed.
        :type inputs: Iterable[SourceType]
        :return: A generator that yields processed source documents after extraction
            and post-processing.
        :rtype: Generator[SourceDocument, None, None]
        :raises ValueError: If inputs provided are not valid for processing.
        """

        def get_source_metadata(node):
            """
            A pipeline class responsible for the extraction of metadata from a collection
            of source nodes. The class facilitates the extraction process for different
            types of input sources, isolating metadata either directly from a document
            node or through its source relationships.

            Methods defined within this class include the primary extraction method
            which processes input nodes and the utility function to retrieve metadata
            dependent on the node type.

            :param inputs: An iterable collection of input nodes of SourceType from which
                metadata will be extracted.
            :type inputs: Iterable[SourceType]
            :return: None. The function operates directly on the input data.
            :rtype: None
            """
            if isinstance(node, Document):
                return node.metadata
            else:
                return node.relationships[NodeRelationship.SOURCE].metadata

        input_source_documents = source_documents_from_source_types(inputs)

        for pre_processor in self.pre_processors:
            input_source_documents = pre_processor.parse_source_docs(
                input_source_documents
            )

        for source_documents in iter_batch(input_source_documents, self.batch_size):

            source_documents = self.id_rewriter.handle_source_docs(source_documents)
            source_documents = self.extraction_decorator.handle_input_docs(
                source_documents
            )

            input_nodes = [n for sd in source_documents for n in sd.nodes]

            filtered_input_nodes = [
                node
                for node in input_nodes
                if self.extraction_filters.filter_source_metadata_dictionary(
                    get_source_metadata(node)
                )
            ]

            logger.info(
                f'Running extraction pipeline [batch_size: {self.batch_size}, num_workers: {self.num_workers}]'
            )

            node_batches = self.ingestion_pipeline._node_batcher(
                num_batches=self.num_workers, nodes=filtered_input_nodes
            )

            output_nodes = run_pipeline(
                self.ingestion_pipeline,
                node_batches,
                num_workers=self.num_workers,
                **self.pipeline_kwargs,
            )

            output_source_documents = self._source_documents_from_base_nodes(
                output_nodes
            )

            for source_document in output_source_documents:
                yield self.extraction_decorator.handle_output_doc(source_document)
