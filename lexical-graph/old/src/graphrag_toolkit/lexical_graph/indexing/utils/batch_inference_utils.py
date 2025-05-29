# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import asyncio
import time
import os
import json
from typing import Any, List, Dict

from tenacity import retry, stop_after_attempt, wait_exponential
from botocore.exceptions import ClientError

from graphrag_toolkit.lexical_graph import BatchJobError
from graphrag_toolkit.lexical_graph.utils import LLMCache
from graphrag_toolkit.lexical_graph.indexing.extract.batch_config import BatchConfig

from llama_index.llms.bedrock_converse import BedrockConverse
from llama_index.llms.anthropic.utils import messages_to_anthropic_messages
from llama_index.llms.bedrock_converse.utils import messages_to_converse_messages
from llama_index.core.schema import TextNode
from llama_index.core.prompts import PromptTemplate
from llama_index.core.base.llms.types import ChatMessage


logger = logging.getLogger(__name__)

BEDROCK_MIN_BATCH_SIZE = 100
BEDROCK_MAX_BATCH_SIZE = 50000


def split_nodes(nodes: List[Any], batch_size: int) -> List[List[Any]]:
    """
    Split a list of nodes into batches based on the specified batch size.

    The function divides the given list of nodes into smaller batches, ensuring that
    the batch size adheres to pre-defined minimum and maximum constraints. If the
    batch size is smaller or larger than specified constraints, an appropriate error
    is raised. Similarly, if the input list of nodes is empty or its size falls below
    the minimum allowable batch size, an error is also raised. The last batch will
    include any remaining elements, provided that the size of this batch meets the
    minimum batch size requirement.

    :param nodes: List of nodes to be split into batches
    :type nodes: List[Any]
    :param batch_size: Size of each batch
    :type batch_size: int
    :return: A list of batches where each batch is a sublist of nodes
    :rtype: List[List[Any]]
    """
    if batch_size < BEDROCK_MIN_BATCH_SIZE:
        raise BatchJobError(
            f'Batch size ({batch_size}) is smaller than the minimum required by Bedrock ({BEDROCK_MIN_BATCH_SIZE})'
        )
    if batch_size > BEDROCK_MAX_BATCH_SIZE:
        raise BatchJobError(
            f'Batch size ({batch_size}) is larger than the maximum required by Bedrock ({BEDROCK_MAX_BATCH_SIZE})'
        )
    if not nodes:
        raise BatchJobError('Empty list of records')
    if len(nodes) < BEDROCK_MIN_BATCH_SIZE:
        raise BatchJobError(
            f'Job contains fewer records ({len(nodes)}) than the minimum required by Bedrock ({BEDROCK_MIN_BATCH_SIZE})'
        )

    i = 0
    results = []

    while i < len(nodes):
        if len(nodes) - (i + batch_size) < BEDROCK_MIN_BATCH_SIZE:
            results.append(nodes[i:])
            break
        else:
            results.append(nodes[i : i + batch_size])
        i += batch_size

    return results


def get_request_body(
    llm: BedrockConverse, messages: List[ChatMessage], inference_parameters: dict
):
    """
    Constructs the request body for the given LLM (Language Model) and input parameters.
    Depending on the model identified by `model_id`, this function adapts the request format
    and the message structure to match the expected input for that specific model. The request
    body is structured to include the required messages, and optionally, a system prompt, if applicable.

    :param llm: An instance of the `BedrockConverse` representing the LLM and its details
        such as model_id.
    :type llm: BedrockConverse
    :param messages: A list of `ChatMessage` objects representing the input dialogue context
        to the LLM.
    :type messages: List[ChatMessage]
    :param inference_parameters: A dictionary containing the inference-specific parameters
        such as 'max_tokens' and 'temperature'. These parameters control the behavior of
        the model's generation.
    :type inference_parameters: dict
    :return: A dictionary representing the request body formatted appropriately based on the
        model type.
    :rtype: dict
    """
    model_id = llm.model

    if 'amazon.nova' in model_id:
        converse_messages, system_prompt = messages_to_converse_messages(messages)
        request_body = {
            'messages': converse_messages,
            'inferenceConfig': {
                'maxTokens': inference_parameters['max_tokens'],
                'temperature': inference_parameters['temperature'],
            },
        }
        if system_prompt:
            request_body['system'] = [{'text': system_prompt}]
        return request_body
    elif 'anthropic.claude' in model_id:
        anthropic_messages, system_prompt = messages_to_anthropic_messages(messages)
        if system_prompt:
            anthropic_messages = [
                {'role': 'system"', 'content': system_prompt},
                *anthropic_messages,
            ]
        request_body = {
            'anthropic_version': inference_parameters.get(
                'anthropic_version', 'bedrock-2023-05-31'
            ),
            'messages': anthropic_messages,
            'max_tokens': inference_parameters['max_tokens'],
            'temperature': inference_parameters['temperature'],
        }
        return request_body
    elif 'meta.llama' in model_id:
        converse_messages, system_prompt = messages_to_converse_messages(messages)
        request_body = {
            'messages': converse_messages,
            'parameters': {
                'max_new_tokens': inference_parameters['max_tokens'],
                'temperature': inference_parameters['temperature'],
            },
        }
        return request_body
    else:
        raise ValueError(
            f'Unrecognized model_id: batch extraction for {model_id} is not supported'
        )


def create_inference_inputs_for_messages(
    llm: BedrockConverse,
    nodes: List[TextNode],
    messages_batch: List[List[ChatMessage]],
    **kwargs,
) -> List[Dict[str, Any]]:
    """
    Creates inference inputs for a batch of messages, associating each message batch with a corresponding node.
    Generates a list of dictionaries containing the input structure needed for inference.

    :param llm: Instance of BedrockConverse used to derive inference parameters and generate request bodies.
    :type llm: BedrockConverse
    :param nodes: List of TextNode objects, with each node representing a unique record.
    :type nodes: List[TextNode]
    :param messages_batch: Nested list of ChatMessage objects, where each inner list corresponds to
                           a batch of messages linked to a node.
    :type messages_batch: List[List[ChatMessage]]
    :param kwargs: Additional keyword arguments for customizing inference parameters.
    :type kwargs: dict
    :return: A list of dictionaries containing 'recordId' keys linked to node IDs and 'modelInput'
             keys containing the generated model inputs.
    :rtype: List[Dict[str, Any]]
    """
    inference_parameters = llm._get_all_kwargs(**kwargs)
    json_outputs = []
    for node, messages in zip(nodes, messages_batch):
        json_structure = {
            'recordId': node.node_id,
            'modelInput': get_request_body(llm, messages, inference_parameters),
        }
        json_outputs.append(json_structure)
    return json_outputs


def create_inference_inputs(
    llm: BedrockConverse, nodes: List[TextNode], prompts: List[str], **kwargs
) -> List[Dict[str, Any]]:
    """
    Generates inference input data required for processing by the LLM service.

    This function accepts a list of text nodes, corresponding prompts, and additional
    keyword arguments to generate a structured input format compatible with the LLM
    provider's API. For each node-prompt pair, the function processes the input,
    constructs the necessary payload using the LLM provider's methods, and aggregates
    the results in a list of dictionaries for further consumption.

    :param llm: The `BedrockConverse` object representing the language model
                implementation that provides methods to handle requests and
                associated configurations.
    :param nodes: A list of `TextNode` objects, where each node represents a
                  unit of textual information with a unique identifier.
    :param prompts: A list of string prompts corresponding to each text node,
                    used as input to the language model for generating inferences.
    :param kwargs: Additional keyword arguments passed to the LLM instance for
                   configuration or tuning during request generation.
    :return: A list of dictionaries, where each dictionary contains the
             `recordId` (derived from the node's unique identifier) and
             `modelInput` (structured input for the LLM provider's API).
    :rtype: List[Dict[str, Any]]
    """
    all_kwargs = llm._get_all_kwargs(**kwargs)
    json_outputs = []
    for node, prompt in zip(nodes, prompts):
        prompt = llm.completion_to_prompt(prompt)
        json_structure = {
            'recordId': node.node_id,
            'modelInput': llm._provider.get_request_body(prompt, all_kwargs),
        }
        json_outputs.append(json_structure)
    return json_outputs


def create_and_run_batch_job(
    job_name_prefix: str,
    bedrock_client: Any,
    timestamp: str,
    batch_index: int,
    batch_config: BatchConfig,
    input_key: str,
    output_path: str,
    model_id: str,
) -> None:
    """
    Creates and runs a batch job using the Bedrock client. This function constructs the
    necessary input and output data configurations and optionally includes VPC configuration
    details if specified. It also handles job creation, logging the job's ARN, and waiting
    for its completion. Any errors encountered during execution are logged and raised.

    :param job_name_prefix: Prefix string used for naming the batch job.
    :param bedrock_client: Instance of the Bedrock service client to interact with.
    :param timestamp: Unique identifier appended to the job name for timestamping purposes.
    :param batch_index: Index number used in naming the batch job to distinguish between
        multiple batches.
    :param batch_config: Configuration object containing details such as the bucket name,
        role ARN, encryption key, VPC settings, etc.
    :param input_key: Key of the input file in the S3 bucket to be processed in the batch job.
    :param output_path: Destination path in the S3 bucket where the output will be stored.
    :param model_id: Identifier of the model to be used for invocation in the batch job.

    :return: None
    """
    try:
        input_data_config = {
            's3InputDataConfig': {
                's3Uri': f's3://{batch_config.bucket_name}/{input_key}'
            }
        }
        output_data_config = {
            's3OutputDataConfig': {
                's3Uri': f's3://{batch_config.bucket_name}/{output_path}'
            }
        }

        if batch_config.s3_encryption_key_id:
            output_data_config['s3EncryptionKeyId'] = batch_config.s3_encryption_key_id

        response = None
        if batch_config.subnet_ids and batch_config.security_group_ids:
            response = bedrock_client.create_model_invocation_job(
                roleArn=batch_config.role_arn,
                modelId=model_id,
                jobName=f'{job_name_prefix}-{timestamp}-{batch_index}',
                inputDataConfig=input_data_config,
                outputDataConfig=output_data_config,
                vpcConfig={
                    'subnetIds': batch_config.subnet_ids,
                    'securityGroupIds': batch_config.security_group_ids,
                },
            )
        else:
            response = bedrock_client.create_model_invocation_job(
                roleArn=batch_config.role_arn,
                modelId=model_id,
                jobName=f'{job_name_prefix}-{timestamp}-{batch_index}',
                inputDataConfig=input_data_config,
                outputDataConfig=output_data_config,
            )

        job_arn = response.get('jobArn')
        logger.info(f'Created batch job [job_arn: {job_arn}]')

        wait_for_job_completion(bedrock_client, job_arn)
    except ClientError as e:
        logger.error(f'Error creating or running batch job: {str(e)}')
        raise BatchJobError(f'{e!s}') from e


def wait_for_job_completion(bedrock_client: Any, job_arn: str) -> None:
    """
    Waits for a batch job to complete by checking the job's status periodically.

    This function monitors the status of a batch job by repeatedly querying the
    `get_model_invocation_job` method of the given client. It logs the progress
    and raises an error if the job does not complete successfully.

    :param bedrock_client: The client instance used to interact with the service API.
        It must provide the `get_model_invocation_job` method to query the job status.
    :type bedrock_client: Any
    :param job_arn: The ARN (Amazon Resource Name) of the batch job to be monitored.
    :type job_arn: str
    :return: This function does not return a value. It will raise an exception if the
        batch job fails.
    :rtype: None
    :raises BatchJobError: If the batch job fails, an exception is raised with the
        failure status and related message from the service response.
    """
    status = 'Started'
    while status not in ['Completed', 'Failed', 'Stopped']:
        time.sleep(60)
        logger.debug(f'Waiting for batch job to complete... [job_arn: {job_arn}]')
        response = bedrock_client.get_model_invocation_job(jobIdentifier=job_arn)
        status = response['status']

    if status != 'Completed':
        logger.error(f'Batch job failed with status: {status}')
        raise BatchJobError(
            f"Batch job failed with status: {status} - {response['message']}"
        )

    logger.debug('Batch job completed successfully')


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def download_output_files(
    s3_client: Any,
    bucket_name: str,
    output_path: str,
    input_filename: str,
    local_directory: str,
) -> None:
    """
    Downloads files from an S3 bucket directory matching a specific prefix and saves them to
    a local directory. This function uses a paginator to iterate through S3 objects, identifies
    a folder containing files matching the given `input_filename`, and then downloads all files
    from that folder into the specified local directory.

    :param s3_client: An S3 client instance used to interact with Amazon S3.
    :param bucket_name: The name of the S3 bucket containing the files.
    :param output_path: The prefix path in the S3 bucket where the files are located.
    :param input_filename: The filename prefix to search for within the output path in S3.
    :param local_directory: The local directory where the downloaded files will be saved.
    :return: None
    """
    paginator = s3_client.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=bucket_name, Prefix=output_path)

    output_folder = None
    for page in pages:
        for obj in page.get('Contents', []):
            key = obj['Key']
            if os.path.basename(key).startswith(input_filename):
                output_folder = os.path.dirname(key)
                break
        if output_folder:
            break

    if not output_folder:
        raise BatchJobError(
            f"No folder containing a file matching '{input_filename}' was found in bucket {bucket_name}."
        )

    output_files = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=output_folder)
    for obj in output_files.get('Contents', []):
        key = obj['Key']
        if key.endswith('/'):
            continue

        local_file_path = os.path.join(
            local_directory, os.path.relpath(key, output_folder)
        )
        os.makedirs(os.path.dirname(local_file_path), exist_ok=True)

        s3_client.download_file(Bucket=bucket_name, Key=key, Filename=local_file_path)
        logger.debug(f'Downloaded: {key} to {local_file_path}')


def get_parse_output_text_fn(model_id: str):
    """
    Returns a function to parse the output text from JSON data based on the specified
    model ID. The returned function differs depending on the specific model identified
    by the `model_id`. This function streamlines the extraction of the text content
    from various structured outputs used by different models.

    :param model_id: Identifier for the model whose output format determines the parsing
        logic.
    :return: A function that accepts JSON data and extracts output text as per the
        specified model's format.
    :raises ValueError: If the provided `model_id` does not match any of the
        supported models.
    """
    if 'amazon.nova' in model_id:

        def get_output_text(json_data):
            contents = (
                json_data.get('modelOutput', {})
                .get('output', {})
                .get('message', {})
                .get('content', [])
            )
            return ''.join([content.get('text', '') for content in contents])

        return get_output_text
    elif 'anthropic.claude' in model_id:

        def get_output_text(json_data):
            contents = json_data.get('modelOutput', {}).get('content', [])
            return ''.join([content.get('text', '') for content in contents])

        return get_output_text
    elif 'meta.llama' in model_id:

        def get_output_text(json_data):
            return json_data['generation']

        return get_output_text
    else:
        raise ValueError(
            f'Unrecognized model_id: batch extraction for {model_id} is not supported'
        )


async def process_batch_output(
    local_output_directory: str, input_filename: str, llm: LLMCache
) -> Dict[str, str]:
    """
    Processes the output of a batch of records, attempting to parse or reprocess failed
    records using a language model.

    This function handles reading records from a specified output directory and input
    filename. Each record is processed to extract meaningful data. Records that fail
    to process are re-attempted sequentially using a provided language model via
    asynchronous calls. Final results, both from the initial parsing and any retries,
    are returned.

    :param local_output_directory: Path to the directory containing the output files
                                   to process.
    :type local_output_directory: str
    :param input_filename: Common prefix of the filenames to process in the specified
                           directory.
    :type input_filename: str
    :param llm: Language model interface used for reprocessing failed records.
    :type llm: LLMCache
    :return: A dictionary mapping record IDs to their processed outputs.
    :rtype: Dict[str, str]
    """
    results = {}
    failed_records = []

    parse_output_text = get_parse_output_text_fn(llm.llm.model)

    for filename in os.listdir(local_output_directory):
        if filename.startswith(input_filename):
            with open(
                os.path.join(local_output_directory, filename), 'r'
            ) as jsonl_file:
                for line in jsonl_file:
                    json_data = json.loads(line)
                    record_id = json_data.get('recordId')
                    error = json_data.get('error')
                    if not error:
                        model_output_text = parse_output_text(json_data)
                        results[record_id] = model_output_text
                    else:
                        failed_records.append(
                            (
                                record_id,
                                json_data.get('modelInput', {})
                                .get('messages', [{}])[0]
                                .get('content', [{}])[0]
                                .get('text', ''),
                            )
                        )

    async def process_failed_record(record):
        record_id, text = record

        def blocking_llm_call():
            return llm.predict(PromptTemplate(text))

        try:
            coro = asyncio.to_thread(blocking_llm_call)
            response = await coro
            logger.info(f'Successfully processed failed record {record_id}')
            return record_id, response
        except Exception as e:
            logger.error(f'Error processing failed record {record_id}: {str(e)}')
            return record_id, None

    failed_results = await asyncio.gather(
        *[process_failed_record(record) for record in failed_records]
    )

    for record_id, response in failed_results:
        if response:
            results[record_id] = response

    return results
