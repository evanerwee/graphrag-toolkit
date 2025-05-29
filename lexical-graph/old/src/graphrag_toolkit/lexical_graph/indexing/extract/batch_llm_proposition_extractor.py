# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
import os
import json

from typing import Optional, List, Sequence, Dict
from datetime import datetime

from graphrag_toolkit.lexical_graph import GraphRAGConfig, BatchJobError
from graphrag_toolkit.lexical_graph.indexing.model import Propositions
from graphrag_toolkit.lexical_graph.utils import LLMCache, LLMCacheType
from graphrag_toolkit.lexical_graph.indexing.constants import PROPOSITIONS_KEY
from graphrag_toolkit.lexical_graph.indexing.prompts import EXTRACT_PROPOSITIONS_PROMPT
from graphrag_toolkit.lexical_graph.indexing.extract.batch_config import BatchConfig
from graphrag_toolkit.lexical_graph.indexing.extract.llm_proposition_extractor import (
    LLMPropositionExtractor,
)
from graphrag_toolkit.lexical_graph.indexing.utils.batch_inference_utils import (
    create_inference_inputs,
    create_inference_inputs_for_messages,
    create_and_run_batch_job,
    download_output_files,
    process_batch_output,
    split_nodes,
)
from graphrag_toolkit.lexical_graph.indexing.utils.batch_inference_utils import (
    BEDROCK_MIN_BATCH_SIZE,
)

from llama_index.core.extractors.interface import BaseExtractor
from llama_index.core.bridge.pydantic import Field
from llama_index.core.schema import TextNode, BaseNode
from llama_index.core.prompts import PromptTemplate

logger = logging.getLogger(__name__)


class BatchLLMPropositionExtractor(BaseExtractor):
    """
    Handles batch-level proposition extraction using a Language Model (LLM),
    allowing for scalable and efficient processing of large datasets. This
    class is designed to enable concurrent batch processing with integration
    to resource-specific clients like S3 and Bedrock for data storage, inference
    job management, and result extraction.

    :ivar batch_config: Configuration for batch processing. Determines key operational
        parameters such as bucket name, region, and other infrastructural configurations.
    :type batch_config: BatchConfig
    :ivar llm: Instance of the language model or its cache. Manages and facilitates prompt-based
        proposition extraction.
    :type llm: Optional[LLMCache]
    :ivar prompt_template: String template for generating language model prompts during
        proposition extraction.
    :type prompt_template: str
    :ivar source_metadata_field: Metadata field name used to extract text propositions.
        It overrides the default extraction text if specified.
    :type source_metadata_field: Optional[str]
    :ivar batch_inference_dir: Directory path for maintaining batch inference input/output data.
    :type batch_inference_dir: str
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

    @classmethod
    def class_name(cls) -> str:
        """
        Provides the name of the class as a string.

        This method is a class-level method that is used to retrieve the name of
        the class it is invoked on. It ensures consistent retrieval of the class
        name without the need for manually specifying it.

        :returns: The name of the class as a string.
        :rtype: str
        """
        return 'BatchLLMPropositionExtractor'

    def __init__(
        self,
        batch_config: BatchConfig,
        llm: LLMCacheType = None,
        prompt_template: str = None,
        source_metadata_field: Optional[str] = None,
        batch_inference_dir: str = None,
    ):
        """
        Initializes an instance of the class, setting up the configuration for batch processing,
        including low-level language model (LLM) configurations, prompt template, metadata
        fields, and output directory for batch inference.

        :param batch_config: Configuration object for batching containing parameters
            and settings for batch processing.
        :type batch_config: BatchConfig
        :param llm: Instance of an optional low-level language model or cache type for
            LLM tasks. Defaults to `GraphRAGConfig.extraction_llm` if not provided.
        :type llm: LLMCacheType, optional
        :param prompt_template: Text template used for generating prompts. If not explicitly
            provided, defaults to the `EXTRACT_PROPOSITIONS_PROMPT` constant.
        :type prompt_template: str, optional
        :param source_metadata_field: An optional field specifying the source's metadata
            utilized in processing. If not provided, defaults to `None`.
        :type source_metadata_field: str, optional
        :param batch_inference_dir: Path to the directory for storing output results from
            batch inference. If not provided, defaults to `'output/batch-propositions'`.
        :type batch_inference_dir: str, optional

        :raises ValueError: Raises an error if the provided `llm` is not an instance of
            `LLMCache` and cannot be initialized correctly.
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
            prompt_template=prompt_template or EXTRACT_PROPOSITIONS_PROMPT,
            source_metadata_field=source_metadata_field,
            batch_inference_dir=batch_inference_dir
            or os.path.join('output', 'batch-propositions'),
        )

        logger.debug(f'Prompt template: {self.prompt_template}')

        self._prepare_directory(self.batch_inference_dir)

    def _prepare_directory(self, dir):
        """
        Prepares a directory by ensuring its existence. If the directory does not exist,
        it is created with necessary parent directories.

        :param dir: Directory path to prepare
        :type dir: str
        :return: The directory path that was prepared
        :rtype: str
        """
        if not os.path.exists(dir):
            os.makedirs(dir, exist_ok=True)
        return dir

    async def process_single_batch(
        self, batch_index: int, node_batch: List[TextNode], s3_client, bedrock_client
    ):
        """
        Process a single batch of text nodes by creating inference inputs, uploading them
        to S3, invoking a batch job, and processing the results.

        This asynchronous function handles multiple steps in a batch processing pipeline,
        including generating inference inputs from text nodes, organizing and uploading
        files to S3, invoking a batch job with the Bedrock client, and retrieving and
        parsing the outputs produced by the batch job. It ensures all necessary resources
        are properly prepared and organized, such as directories and filenames.

        All operations, including S3 interactions and batch job invocations, are handled
        asynchronously to ensure non-blocking execution.

        :param batch_index: The index of the current batch being processed.
        :type batch_index: int
        :param node_batch: A list of TextNode objects representing the batch of text
            nodes to process.
        :type node_batch: List[TextNode]
        :param s3_client: An S3 client object for interacting with an Amazon S3 bucket.
        :type s3_client: Any
        :param bedrock_client: A Bedrock client object for invoking batch jobs.
        :type bedrock_client: Any
        :return: A list of batch results derived from the processed batch output.
        :rtype: List[Any]
        """
        try:
            timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
            input_filename = (
                f'proposition_extraction_{timestamp}_batch_{batch_index}.jsonl'
            )

            # 1 - Create Record Files (.jsonl)
            # prompts = []
            # for node in node_batch:
            #     text = node.metadata.get(self.source_metadata_field, node.text) if self.source_metadata_field else node.text
            #     prompt = self.prompt_template.format(text=text)
            #     prompts.append(prompt)

            messages_batch = []
            for node in node_batch:
                text = (
                    node.metadata.get(self.source_metadata_field, node.text)
                    if self.source_metadata_field
                    else node.text
                )
                messages = self.llm.llm._get_messages(
                    PromptTemplate(self.prompt_template), text=text
                )
                messages_batch.append(messages)

            json_inputs = create_inference_inputs_for_messages(
                self.llm.llm, node_batch, messages_batch
            )

            # json_inputs = create_inference_inputs(
            #     self.llm.llm,
            #     node_batch,
            #     prompts
            # )

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
                    'batch-propositions',
                    timestamp,
                    str(batch_index),
                    'inputs',
                    os.path.basename(input_filename),
                )
                s3_output_path = os.path.join(
                    self.batch_config.key_prefix,
                    'batch-propositions',
                    timestamp,
                    str(batch_index),
                    'outputs/',
                )
            else:
                s3_input_key = os.path.join(
                    'batch-propositions',
                    timestamp,
                    str(batch_index),
                    'inputs',
                    os.path.basename(input_filename),
                )
                s3_output_path = os.path.join(
                    'batch-propositions', timestamp, str(batch_index), 'outputs/'
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
                'extract-propositions',
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
        Extracts information asynchronously from a sequence of nodes, either individually
        or in batches, depending on the minimum batch size defined for Bedrock processing.
        If the number of nodes is below the minimum threshold, a fallback extraction method
        is used. The function processes batches concurrently while managing the concurrency
        limit using semaphores, ensuring optimal performance without exceeding resource limits.

        :param nodes:
            A sequence of BaseNode instances from which information will be extracted. Each
            node represents an individual piece of data to process.
        :type nodes: Sequence[BaseNode]
        :return:
            A list of dictionaries, where each dictionary contains processed results
            including propositions associated with nodes. If a node does not contain
            results, the propositions key will have an empty list as its value.
        :rtype: List[Dict]
        """
        if len(nodes) < BEDROCK_MIN_BATCH_SIZE:
            logger.debug(
                f'List of nodes contains fewer records ({len(nodes)}) than the minimum required by Bedrock ({BEDROCK_MIN_BATCH_SIZE}), so running LLMPropositionExtractor instead'
            )
            extractor = LLMPropositionExtractor(
                prompt_template=self.prompt_template,
                source_metadata_field=self.source_metadata_field,
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
            Asynchronous batch LLM-based proposition extractor.

            This class utilizes an LLM-based approach to extract propositions asynchronously
            from a sequence of nodes in batches. It inherits from the BaseExtractor class
            and leverages semaphores to manage concurrent batch processing.

            :param nodes: Sequence of BaseNode objects to be processed for proposition
                extraction.
            :type nodes: Sequence[BaseNode]
            :return: A list of dictionaries containing extracted propositions, where each
                dictionary represents the propositions for a single node batch.
            :rtype: List[Dict]
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

        # 3 - Process proposition nodes
        return_results = []
        for node in nodes:
            if node.node_id in all_results:
                raw_response = all_results[node.node_id]
                propositions = raw_response.split('\n')
                propositions_model = Propositions(
                    propositions=[p for p in propositions if p]
                )
                return_results.append(
                    {PROPOSITIONS_KEY: propositions_model.model_dump()['propositions']}
                )
            else:
                return_results.append({PROPOSITIONS_KEY: []})

        return return_results
