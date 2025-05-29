# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
import os
import json

from typing import Optional, List, Sequence, Dict
from datetime import datetime

from graphrag_toolkit.lexical_graph import GraphRAGConfig, BatchJobError
from graphrag_toolkit.lexical_graph.utils import LLMCache, LLMCacheType
from graphrag_toolkit.lexical_graph.indexing.utils.topic_utils import (
    parse_extracted_topics,
    format_list,
    format_text,
)
from graphrag_toolkit.lexical_graph.indexing.utils.batch_inference_utils import (
    create_inference_inputs,
    create_inference_inputs_for_messages,
    create_and_run_batch_job,
    download_output_files,
    process_batch_output,
    split_nodes,
)
from graphrag_toolkit.lexical_graph.indexing.constants import (
    TOPICS_KEY,
    DEFAULT_ENTITY_CLASSIFICATIONS,
)
from graphrag_toolkit.lexical_graph.indexing.prompts import EXTRACT_TOPICS_PROMPT
from graphrag_toolkit.lexical_graph.indexing.extract.topic_extractor import (
    TopicExtractor,
)
from graphrag_toolkit.lexical_graph.indexing.extract.batch_config import BatchConfig
from graphrag_toolkit.lexical_graph.indexing.extract.scoped_value_provider import (
    ScopedValueProvider,
    FixedScopedValueProvider,
    DEFAULT_SCOPE,
)
from graphrag_toolkit.lexical_graph.indexing.utils.batch_inference_utils import (
    BEDROCK_MIN_BATCH_SIZE,
)

from llama_index.core.extractors.interface import BaseExtractor
from llama_index.core.bridge.pydantic import Field
from llama_index.core.schema import TextNode, BaseNode
from llama_index.core.prompts import PromptTemplate

logger = logging.getLogger(__name__)


class BatchTopicExtractor(BaseExtractor):
    """
    Extracts topics in batch mode using a prompt-based large language model (LLM)
    system. This class handles the preparation, creation, and management of
    batch-level topic extraction workflows. It facilitates entities and topics
    classification, generates prompts, invokes batch LLM jobs, and integrates
    output processing seamlessly.

    The primary purpose of this class is to support scalable, efficient, and
    flexible topic extraction from input data in a batch mode. The configuration
    can be adapted for varied batch setups, metadata sources, and external systems.

    :ivar batch_config: Configuration object for batch inference, including any
        details related to the batch's operational behavior and constraints.
    :type batch_config: BatchConfig
    :ivar llm: Optional language model (LLM) cache configuration used to
        facilitate prompts and model interactions.
    :type llm: Optional[LLMCache]
    :ivar prompt_template: String template to format extraction prompts. This
        attribute defines how prompts are generated for LLM processing.
    :type prompt_template: str
    :ivar source_metadata_field: Metadata field in the input data associated with
        propositions to extract. If None, defaults to the main text content.
    :type source_metadata_field: Optional[str]
    :ivar batch_inference_dir: Path to a directory where batch inputs and outputs
        are stored. Used as a base directory for managing job files.
    :type batch_inference_dir: str
    :ivar entity_classification_provider: Scoped value provider supplying
        classification-related information for input entities.
    :type entity_classification_provider: ScopedValueProvider
    :ivar topic_provider: Scoped value provider supplying information about topics,
        scoped to specific contexts for flexibility.
    :type topic_provider: ScopedValueProvider
    """

    batch_config: BatchConfig = Field('Batch inference config')
    llm: Optional[LLMCache] = Field(description='The LLM to use for extraction')
    prompt_template: str = Field(description='Prompt template')
    source_metadata_field: Optional[str] = Field(
        description='Metadata field from which to extract propositions'
    )
    batch_inference_dir: str = Field(
        description='Directory for batch inputs and results results'
    )
    entity_classification_provider: ScopedValueProvider = Field(
        description='Entity classification provider'
    )
    topic_provider: ScopedValueProvider = Field(description='Topic provider')

    @classmethod
    def class_name(cls) -> str:
        """
        Returns the name of the class.

        This method is used to retrieve the name of the class as a string. It is
        a class-level method and does not require an instance of the class to be
        called.

        :return: The name of the class.
        :rtype: str
        """
        return 'BatchTopicExtractor'

    def __init__(
        self,
        batch_config: BatchConfig,
        llm: LLMCacheType = None,
        prompt_template: str = None,
        source_metadata_field: Optional[str] = None,
        batch_inference_dir: str = None,
        entity_classification_provider: Optional[ScopedValueProvider] = None,
        topic_provider: Optional[ScopedValueProvider] = None,
    ):
        """
        Initializes the class instance with configuration and components required for batch
        processing and topic extraction. It sets up configuration, directories, and functional
        dependencies like prompt templates and classification providers.

        :param batch_config: The configuration object defining batch processing parameters,
            detailing paths, limits, and rules that guide how the batch operations are conducted.
        :type batch_config: BatchConfig

        :param llm: An instance of the language model or a cache-enabled wrapper around it,
            used for leveraging language processing tasks within batch operations. Defaults to None.

        :param prompt_template: The textual template used to guide topic extraction or classification.
            If not provided explicitly, a default value is applied.

        :param source_metadata_field: The source metadata field from which additional
            information is extracted during processing. Defaults to None.

        :param batch_inference_dir: The directory path where batch inference results are
            stored, including any intermediate outputs. If not provided, defaults to
            'output/batch-topics'.

        :param entity_classification_provider: The provider object responsible for classifying
            entities during processing. Defaults to a fixed scoped value provider with default
            entity classifications if not explicitly specified.

        :param topic_provider: The provider object used for supplying topics during batch
            operations or extractions. A fixed scoped value provider is configured with an
            empty default scope if undefined.
        """
        super().__init__(
            batch_config=batch_config,
            llm=(
                llm
                if llm and isinstance(llm, LLMCache)
                else LLMCache(
                    llm=llm or GraphRAGConfig.extraction_llm,
                    enable_cache=GraphRAGConfig.enable_cache,
                )
            ),
            prompt_template=prompt_template or EXTRACT_TOPICS_PROMPT,
            source_metadata_field=source_metadata_field,
            batch_inference_dir=batch_inference_dir
            or os.path.join('output', 'batch-topics'),
            entity_classification_provider=entity_classification_provider
            or FixedScopedValueProvider(
                scoped_values={DEFAULT_SCOPE: DEFAULT_ENTITY_CLASSIFICATIONS}
            ),
            topic_provider=topic_provider
            or FixedScopedValueProvider(scoped_values={DEFAULT_SCOPE: []}),
        )

        logger.debug(f'Prompt template: {self.prompt_template}')

        self._prepare_directory(self.batch_inference_dir)

    def _prepare_directory(self, dir):
        """
        Prepares a directory by ensuring it exists. If the directory does not exist,
        it will be created with all intermediate directories as needed.

        :param dir: The directory path to prepare.
        :type dir: str
        :return: The prepared directory path, which is guaranteed to exist.
        :rtype: str
        """
        if not os.path.exists(dir):
            os.makedirs(dir, exist_ok=True)
        return dir

    def _get_metadata_or_default(self, metadata, key, default):
        """
        Retrieves a value from the metadata dictionary using a specified key. If the key is not present or the value associated
        with the key evaluates to a falsy value, the provided default value is returned.

        :param metadata: Dictionary containing key-value pairs used to retrieve configuration or meta information.
        :type metadata: dict
        :param key: The key to look up in the metadata dictionary.
        :type key: str
        :param default: The fallback value to return if the key is not found or its value evaluates to a falsy value.
        :type default: Any
        :return: The value associated with the key from the metadata dictionary or the provided default value.
        :rtype: Any
        """
        value = metadata.get(key, default)
        return value or default

    async def process_single_batch(
        self, batch_index: int, node_batch: List[TextNode], s3_client, bedrock_client
    ):
        """
        Processes a single batch of text nodes by creating input files, uploading
        them to S3, invoking a batch job for topic extraction, downloading the
        output files, and processing the results.

        This function handles input formatting, S3 operations, external LLM model
        invocations, and output processing using the provided clients and resources.

        :param batch_index: The index of the batch being processed.
        :type batch_index: int
        :param node_batch: A list of TextNode objects to process.
        :type node_batch: List[TextNode]
        :param s3_client: The S3 client instance used for uploading and downloading files.
        :type s3_client: Any
        :param bedrock_client: The Bedrock client instance used for batch job execution.
        :type bedrock_client: Any
        :return: A list of processed results for the given batch.
        :rtype: Any
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            input_filename = f'topic_extraction_{timestamp}_{batch_index}.jsonl'

            # 1 - Create Record Files (.jsonl)
            # prompts = []
            # for node in node_batch:
            #     (_, current_entity_classifications) = self.entity_classification_provider.get_current_values(node)
            #     (_, current_topics) = self.topic_provider.get_current_values(node)
            #     text = format_text(
            #         self._get_metadata_or_default(node.metadata, self.source_metadata_field, node.text)
            #         if self.source_metadata_field
            #         else node.text
            #     )
            #     prompt = self.prompt_template.format(
            #         text=text,
            #         preferred_entity_classifications=format_list(current_entity_classifications),
            #         preferred_topics=format_list(current_topics)
            #     )
            #     prompts.append(prompt)

            # json_inputs = create_inference_inputs(
            #     self.llm.llm,
            #     node_batch,
            #     prompts
            # )

            messages_batch = []
            for node in node_batch:
                (_, current_entity_classifications) = (
                    self.entity_classification_provider.get_current_values(node)
                )
                (_, current_topics) = self.topic_provider.get_current_values(node)
                text = format_text(
                    self._get_metadata_or_default(
                        node.metadata, self.source_metadata_field, node.text
                    )
                    if self.source_metadata_field
                    else node.text
                )
                messages = self.llm.llm._get_messages(
                    PromptTemplate(self.prompt_template),
                    text=text,
                    preferred_entity_classifications=format_list(
                        current_entity_classifications
                    ),
                    preferred_topics=format_list(current_topics),
                )
                messages_batch.append(messages)

            json_inputs = create_inference_inputs_for_messages(
                self.llm.llm, node_batch, messages_batch
            )

            input_dir = os.path.join(
                self.batch_inference_dir, timestamp, str(batch_index), 'inputs'
            )
            output_dir = os.path.join(
                self.batch_inference_dir, timestamp, str(batch_index), 'outputs'
            )
            self._prepare_directory(input_dir)
            self._prepare_directory(output_dir)

            input_filepath = os.path.join(input_dir, input_filename)
            with open(input_filepath, 'w') as file:
                for item in json_inputs:
                    json.dump(item, file)
                    file.write('\n')

            # 2 - Upload records to s3
            s3_input_key = None
            s3_output_path = None
            if self.batch_config.key_prefix:
                s3_input_key = os.path.join(
                    self.batch_config.key_prefix,
                    'batch-topics',
                    timestamp,
                    str(batch_index),
                    'inputs',
                    os.path.basename(input_filename),
                )
                s3_output_path = os.path.join(
                    self.batch_config.key_prefix,
                    'batch-topics',
                    timestamp,
                    str(batch_index),
                    'outputs/',
                )
            else:
                s3_input_key = os.path.join(
                    'batch-topics',
                    timestamp,
                    str(batch_index),
                    'inputs',
                    os.path.basename(input_filename),
                )
                s3_output_path = os.path.join(
                    'batch-topics', timestamp, str(batch_index), 'outputs/'
                )

            await asyncio.to_thread(
                s3_client.upload_file,
                input_filepath,
                self.batch_config.bucket_name,
                s3_input_key,
            )
            logger.debug(
                f'Uploaded {input_filename} to S3 [bucket: {self.batch_config.bucket_name}, key: {s3_input_key}]'
            )

            # 3 - Invoke batch job
            await asyncio.to_thread(
                create_and_run_batch_job,
                'extract-topics',
                bedrock_client,
                timestamp,
                batch_index,
                self.batch_config,
                s3_input_key,
                s3_output_path,
                self.llm.model,
            )

            await asyncio.to_thread(
                download_output_files,
                s3_client,
                self.batch_config.bucket_name,
                s3_output_path,
                input_filename,
                output_dir,
            )

            # 4 - Once complete, process batch output
            batch_results = await process_batch_output(
                output_dir, input_filename, self.llm
            )
            logger.debug(f'Completed processing of batch {batch_index}')
            return batch_results

        except Exception as e:
            raise BatchJobError(
                f'Error processing batch {batch_index}: {str(e)}'
            ) from e

    async def aextract(self, nodes: Sequence[BaseNode]) -> List[Dict]:
        """
        Asynchronous method for extracting topics from a sequence of nodes. The method is optimized
        to handle data in batches, with support for parallel processing. If the sequence contains
        fewer nodes than the minimum required batch size, it utilizes the TopicExtractor instead.
        Otherwise, it splits the data into batches for concurrent processing and retrieves results
        through Bedrock. Processes the results to extract and format topic information.

        :param nodes: Sequence of BaseNode objects representing data from which topics are to be extracted
        :type nodes: Sequence[BaseNode]
        :return: List of dictionaries containing extracted topics, formatted for further processing
        :rtype: List[Dict]
        """
        if len(nodes) < BEDROCK_MIN_BATCH_SIZE:
            logger.debug(
                f'List of nodes contains fewer records ({len(nodes)}) than the minimum required by Bedrock ({BEDROCK_MIN_BATCH_SIZE}), so running TopicExtractor instead'
            )
            extractor = TopicExtractor(
                prompt_template=self.prompt_template,
                source_metadata_field=self.source_metadata_field,
                entity_classification_provider=self.entity_classification_provider,
                topic_provider=self.topic_provider,
            )
            return await extractor.aextract(nodes)

        s3_client = GraphRAGConfig.s3
        bedrock_client = GraphRAGConfig.bedrock

        # 1 - Split nodes into batches (if needed)
        node_batches = split_nodes(nodes, self.batch_config.max_batch_size)
        logger.debug(
            f'Split nodes into {len(node_batches)} batches [sizes: {[len(b) for b in node_batches]}]'
        )

        # 2 - Process batches concurrently
        all_results = {}
        semaphore = asyncio.Semaphore(self.batch_config.max_num_concurrent_batches)

        async def process_batch_with_semaphore(batch_index, node_batch):
            """
            Represents a batch topic extractor which handles the asynchronous processing
            of nodes in batches, utilizing concurrency control to manage a limited number
            of simultaneous operations.

            The `BatchTopicExtractor` class inherits from the `BaseExtractor` and provides
            an implementation for asynchronous topic extraction. This process ensures
            efficient and thread-safe execution of batch operations using semaphores.

            :param batch_index: An integer indicating the specific index of the batch
                being processed during concurrent execution.
                Type: int

            :param node_batch: A list of `BaseNode` instances representing the batch of
                nodes for which topics are being extracted. Each node contains
                relevant information used during processing.
                Type: Sequence[BaseNode]

            :return: A list of dictionaries where each dictionary corresponds to
                the extracted topics and related information for a processed node
                within the batch.
                Type: List[Dict]
            """
            async with semaphore:
                return await self.process_single_batch(
                    batch_index, node_batch, s3_client, bedrock_client
                )

        tasks = [
            process_batch_with_semaphore(i, batch)
            for i, batch in enumerate(node_batches)
        ]
        batch_results = await asyncio.gather(*tasks)

        for result in batch_results:
            all_results.update(result)

        # 3 - Process topic nodes
        return_results = []
        for node in nodes:
            record_id = node.node_id
            if record_id in all_results:
                (topics, _) = parse_extracted_topics(all_results[record_id])
                return_results.append({TOPICS_KEY: topics.model_dump()})

        return return_results
