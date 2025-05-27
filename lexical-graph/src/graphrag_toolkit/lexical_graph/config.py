# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

_unspecified = object()

import os
import json
import boto3
from dataclasses import dataclass, field
from typing import Optional, Union, Dict, Set, List
from boto3 import Session as Boto3Session
from botocore.config import Config

from llama_index.llms.bedrock_converse import BedrockConverse
from llama_index.embeddings.bedrock import BedrockEmbedding
from llama_index.core.settings import Settings
from llama_index.core.llms import LLM
from llama_index.core.base.embeddings.base import BaseEmbedding

from graphrag_toolkit.lexical_graph.logging import logging

logger = logging.getLogger(__name__)

LLMType = Union[LLM, str]
EmbeddingType = Union[BaseEmbedding, str]

DEFAULT_EXTRACTION_MODEL = 'us.anthropic.claude-3-5-sonnet-20240620-v1:0'
DEFAULT_RESPONSE_MODEL = 'us.anthropic.claude-3-5-sonnet-20240620-v1:0'
DEFAULT_EMBEDDINGS_MODEL = 'cohere.embed-english-v3'
DEFAULT_RERANKING_MODEL = 'mixedbread-ai/mxbai-rerank-xsmall-v1'
DEFAULT_EMBEDDINGS_DIMENSIONS = 1024
DEFAULT_EXTRACTION_NUM_WORKERS = 2
DEFAULT_EXTRACTION_BATCH_SIZE = 4
DEFAULT_EXTRACTION_NUM_THREADS_PER_WORKER = 4
DEFAULT_BUILD_NUM_WORKERS = 2
DEFAULT_BUILD_BATCH_SIZE = 4
DEFAULT_BUILD_BATCH_WRITE_SIZE = 25
DEFAULT_BATCH_WRITES_ENABLED = True
DEFAULT_INCLUDE_DOMAIN_LABELS = False
DEFAULT_ENABLE_CACHE = False
DEFAULT_METADATA_DATETIME_SUFFIXES = ['_date', '_datetime']


def _is_json_string(s):
    """Determines if a given string is a valid JSON string by attempting to
    parse it.

    This function attempts to parse the provided string as JSON. If the parsing
    is successful, it concludes that the string is a valid JSON. Otherwise,
    it considers the string invalid and returns False.

    Args:
        s (str): The string to check for JSON validity.

    Returns:
        bool: True if the input string is a valid JSON string, False otherwise.
    """
    try:
        json.loads(s)
        return True
    except ValueError:
        return False


def string_to_bool(s, default_value: bool):
    """
    Converts a string representation of a boolean to an actual boolean value.
    The function takes a string input and checks if it can be interpreted as
    a boolean. If the string is empty or None, the provided default value will
    be returned. Otherwise, the comparison logic is case-insensitive to determine
    if the string represents a "true" boolean value.

    :param s: The string input to be evaluated. It might contain a textual
        representation of a boolean.
    :param default_value: Default boolean value to return if the string is empty
        or None.
    :return: A boolean value derived from the given string or the default value
        if the input string is empty or None.
    """
    if not s:
        return default_value
    else:
        return s.lower() in ['true']


@dataclass
class _GraphRAGConfig:
    """Configuration class for managing parameters and clients in a Graph-based
    RAG (Retrieve and Generate) system. This class encapsulates the
    configuration necessary for interacting with AWS services, LLM-based
    extractions, and embeddable models while providing utility properties and
    methods to simplify access and management.

    Attributes:
        _aws_profile (Optional[str]): The AWS profile name to be utilized for authentication.
        _aws_region (Optional[str]): The AWS region to be used for service access.
        _aws_clients (Dict): Caches AWS clients for reusability and efficiency.
        _boto3_session (Optional[boto3.Session]): The internal boto3 session, initialized on demand.
        _aws_valid_services (Optional[Set[str]]): A set of valid AWS service names for validation purposes.
        _session (Optional[boto3.Session]): Boto3 session attribute initialized or reused.
        _extraction_llm (Optional[LLM]): The LLM configured for extraction tasks.
        _response_llm (Optional[LLM]): The LLM configured for response generation tasks.
        _embed_model (Optional[BaseEmbedding]): An embedding model for vector generation.
        _embed_dimensions (Optional[int]): The dimensions of the embedding vectors.
        _reranking_model (Optional[str]): A string specifying the reranking model to use.
        _extraction_num_workers (Optional[int]): Number of parallel workers for extraction tasks.
        _extraction_num_threads_per_worker (Optional[int]): Number of threads per worker in extraction operations.
        _extraction_batch_size (Optional[int]): The batch size for processing during extraction.
        _build_num_workers (Optional[int]): The number of workers used when building structures or data.
        _build_batch_size (Optional[int]): The size of data batches processed during build operations.
        _build_batch_write_size (Optional[int]): Limit for the size of batch writes in the build process.
        _batch_writes_enabled (Optional[bool]): Flag indicating whether batch writes are enabled.
        _include_domain_labels (Optional[bool]): Whether domain-specific labels are included in processes.
        _enable_cache (Optional[bool]): Boolean flag to enable or disable caching mechanisms.
        _metadata_datetime_suffixes (Optional[List[str]]): List of datetime suffixes included in metadata handling.
    """

    _aws_profile: Optional[str] = None
    _aws_region: Optional[str] = None
    _aws_clients: Dict = field(default_factory=dict)  # Use field() for mutable default

    _boto3_session: Optional[boto3.Session] = field(
        default=None, init=False, repr=False
    )
    _aws_valid_services: Optional[Set[str]] = field(
        default=None, init=False, repr=False
    )
    _session: Optional[boto3.Session] = field(default=None, init=False, repr=False)

    _extraction_llm: Optional[LLM] = None
    _response_llm: Optional[LLM] = None
    _embed_model: Optional[BaseEmbedding] = None
    _embed_dimensions: Optional[int] = None
    _reranking_model: Optional[str] = None
    _extraction_num_workers: Optional[int] = None
    _extraction_num_threads_per_worker: Optional[int] = None
    _extraction_batch_size: Optional[int] = None
    _build_num_workers: Optional[int] = None
    _build_batch_size: Optional[int] = None
    _build_batch_write_size: Optional[int] = None
    _batch_writes_enabled: Optional[bool] = None
    _include_domain_labels: Optional[bool] = None
    _enable_cache: Optional[bool] = None
    _metadata_datetime_suffixes: Optional[List[str]] = None

    _system_prompt_arn: Optional[str] = None
    _user_prompt_arn: Optional[str] = None
    _response_prompt_arn: Optional[str] = None

    _system_prompt: Optional[str] = None
    _user_prompt: Optional[str] = None

    def _resolve_prompt_arn(self, value: str) -> str:
        """
        Resolve a prompt identifier into a full Bedrock prompt ARN.

        If the value is already a full ARN, it is returned as-is.
        Otherwise, assume it's a short name and construct the ARN using:
        - the configured AWS region (from self.aws_region)
        - the current AWS account ID (via STS)

        :param value: Full ARN or short name (e.g., "my-prompt")
        :return: Full ARN string
        """
        if value.startswith("arn:aws:bedrock:"):
            return value

        sts = self._get_or_create_client("sts")
        account_id = sts.get_caller_identity()["Account"]
        return f"arn:aws:bedrock:{self.aws_region}:{account_id}:prompt/{value}"
    def _get_or_create_client(self, service_name: str) -> boto3.client:
        """
        Retrieve or create a boto3 client for the specified AWS service. The method
        maintains a cache of previously created clients to avoid redundant
        initialization. If the requested client exists in the cache, it is returned.
        Otherwise, a new boto3 session is created using the specified AWS region and
        profile, and the client is initialized and stored in the cache.

        :param service_name: Name of the AWS service for which the boto3 client needs
            to be retrieved or created.
        :type service_name: str
        :return: An instance of boto3 client for the specified AWS service.
        :rtype: boto3.client
        :raises AttributeError: If a boto3 client cannot be created due to an error in
            session initialization or client creation.
        """
        if service_name in self._aws_clients:
            return self._aws_clients[service_name]

        region = self.aws_region

        profile = self.aws_profile

        try:
            if profile:
                session = boto3.Session(profile_name=profile, region_name=region)
            else:
                session = boto3.Session(region_name=region)

            client = session.client(service_name)
            self._aws_clients[service_name] = client
            return client

        except Exception as e:
            raise AttributeError(
                f"Failed to create boto3 client for '{service_name}'. "
                f"Profile: '{profile}', Region: '{region}'. "
                f"Original error: {str(e)}"
            ) from e

    def fetch_bedrock_prompt_text(self, prompt_identifier: str) -> str:
        """
        Retrieves the prompt text from a Bedrock Agent prompt.

        Accepts either a full ARN or a short prompt name and returns the prompt's template text.
        """
        if not prompt_identifier.startswith("arn:"):
            try:
                sts_client = self._get_or_create_client("sts")
                account_id = sts_client.get_caller_identity()["Account"]
                prompt_identifier = f"arn:aws:bedrock:{self.aws_region}:{account_id}:prompt/{prompt_identifier}"
            except Exception as e:
                raise RuntimeError(f"Failed to construct ARN from short prompt name: {e}") from e

        try:
            response = self._get_or_create_client("bedrock-agent").get_prompt(
                promptIdentifier=prompt_identifier
            )

            # ✅ Extract only the text from the default variant
            return response["variants"][0]["templateConfiguration"]["text"]["text"]

        except Exception as e:
            raise RuntimeError(
                f"Failed to retrieve Bedrock prompt for identifier '{prompt_identifier}': {e}"
            ) from e

    @property
    def system_prompt(self) -> Optional[str]:
        if self._system_prompt:
            return self._system_prompt

        self._system_prompt_arn = self._system_prompt_arn or os.environ.get("SYSTEM_PROMPT_ARN")
        if self._system_prompt_arn:
            try:
                self._system_prompt = self.fetch_bedrock_prompt_text(self._system_prompt_arn)
            except Exception as e:
                logger.warning(f"Failed to fetch system prompt from ARN: {e}")

        return self._system_prompt

    @system_prompt.setter
    def system_prompt(self, value: str):
        self._system_prompt = value
        self._system_prompt_arn = None  # clear ARN resolution

    @property
    def user_prompt(self) -> Optional[str]:
        if self._user_prompt:
            return self._user_prompt

        self._user_prompt_arn = self._user_prompt_arn or os.environ.get("USER_PROMPT_ARN")
        if self._user_prompt_arn:
            try:
                self._user_prompt = self.fetch_bedrock_prompt_text(self._user_prompt_arn)
            except Exception as e:
                logger.warning(f"Failed to fetch user prompt from ARN: {e}")

        return self._user_prompt

    @user_prompt.setter
    def user_prompt(self, value: str):
        self._user_prompt = value
        self._user_prompt_arn = None

    @property
    def response_prompt_arn(self) -> Optional[str]:
        """
        Retrieves the prompt template ARN for Bedrock-managed response prompts.
        Falls back to the environment variable RESPONSE_PROMPT_ARN if not set.
        """
        if self._response_prompt_arn is None:
            self._response_prompt_arn = os.environ.get('RESPONSE_PROMPT_ARN')
        return self._response_prompt_arn

    @response_prompt_arn.setter
    def response_prompt_arn(self, arn: str) -> None:
        """
        Sets the response prompt ARN to use for Bedrock-managed templates.
        """
        self._response_prompt_arn = arn

    @property
    def bedrock_agent(self):
        """
        Returns the boto3 client for Bedrock Agent (used to fetch prompts, agents, etc.).
        """
        return self._get_or_create_client("bedrock-agent")

    @property
    def session(self) -> Boto3Session:
        """
        Provides the `session` property which initializes and retrieves a boto3 session
        based on the AWS profile and region configuration. If the session is not already
        initialized, it attempts to create a new boto3 session using either an explicitly
        set profile or the default AWS configuration, including environment variables.

        :raises RuntimeError: If an error occurs during the initialization of the boto3
            session, providing details on the profile, region, and underlying error.

        :return: The active boto3 session instance.
        :rtype: Boto3Session
        """
        if not hasattr(self, "_boto3_session") or self._boto3_session is None:
            try:
                # Prefer explicitly set profile
                if self.aws_profile:
                    self._boto3_session = Boto3Session(
                        profile_name=self.aws_profile, region_name=self.aws_region
                    )
                else:
                    # Use environment variables or default config
                    self._boto3_session = Boto3Session(region_name=self.aws_region)

            except Exception as e:
                raise RuntimeError(
                    f"Unable to initialize boto3 session. "
                    f"Profile: {self.aws_profile}, Region: {self.aws_region}. "
                    f"Error: {e}"
                ) from e

        return self._boto3_session

    @property
    def s3(self):
        """
        Provides access to the S3 client through a property. The client is lazily created
        if it does not already exist, enabling on-demand usage without initializing it
        immediately.

        :return: An S3 client initialized and ready for interaction.
        :rtype: Any
        """
        return self._get_or_create_client("s3")

    @property
    def bedrock(self):
        """
        Provides a property to access a specific client.

        This property allows fetching or creating a specific client instance called
        'bedrock' using an internal utility method. It ensures the 'bedrock' client
        is instantiated and available for further usage when accessed.

        :return: The instance of the 'bedrock' client.
        :rtype: Any
        """
        return self._get_or_create_client("bedrock")

    @property
    def rds(self):
        """
        Retrieves or creates the Amazon RDS client instance.

        This property method checks if an Amazon RDS client already exists;
        if not, it creates one and returns it.

        :return: An Amazon RDS client instance.
        :rtype: boto3.client
        """
        return self._get_or_create_client("rds")

    @property
    def aws_profile(self) -> Optional[str]:
        """
        Retrieves the AWS profile name used for authentication. If the profile
        is not already set, it attempts to retrieve it from the environmental
        variable `AWS_PROFILE`.

        :return: The name of the AWS profile used for authentication or `None`
            if it is not set in the environment and the internal attribute is unset.
        :rtype: Optional[str]
        """
        if self._aws_profile is None:
            self._aws_profile = os.environ.get("AWS_PROFILE")
        return self._aws_profile

    @aws_profile.setter
    def aws_profile(self, profile: str) -> None:
        """
        Sets the AWS profile to use for connections. Clears the existing AWS clients
        to ensure new connections are established with the updated profile.

        :param profile: The name of the AWS profile to set.
        :type profile: str
        :return: None
        """
        self._aws_profile = profile
        self._aws_clients.clear()  # Clear old clients to force regeneration

    @property
    def aws_region(self) -> str:
        """
        A property to retrieve the AWS region used by the application. The region is resolved
        from the `AWS_REGION` environment variable or the default region configured in the
        `boto3` session. If the `_aws_region` attribute has been set already, the cached value
        will be returned. Otherwise, it initializes the attribute with the resolved AWS region.

        :return: The AWS region as a string.
        :rtype: str
        """
        if self._aws_region is None:
            self._aws_region = os.environ.get("AWS_REGION", boto3.Session().region_name)
        return self._aws_region

    @aws_region.setter
    def aws_region(self, region: str) -> None:
        """
        Sets the AWS region for the instance and clears any cached AWS clients.
        This allows the instance to work with the specified AWS region and resets
        the state of any previously initialized AWS SDK clients.

        :param region: AWS region to set for the instance
        :type region: str

        """
        self._aws_region = region
        self._aws_clients.clear()  # Optional: reset clients if a region changes

    @property
    def extraction_num_workers(self) -> int:
        """
        Computes and retrieves the number of extraction workers. If the number
        of extraction workers is not already set, it assigns a default value
        obtained from an environment variable or a predefined constant.

        :return: The current number of extraction workers.
        :rtype: int
        """
        if self._extraction_num_workers is None:
            self.extraction_num_workers = int(
                os.environ.get('EXTRACTION_NUM_WORKERS', DEFAULT_EXTRACTION_NUM_WORKERS)
            )

        return self._extraction_num_workers

    @extraction_num_workers.setter
    def extraction_num_workers(self, num_workers: int) -> None:
        """
        Sets the number of workers used for data extraction.

        This method allows configuring the number of workers that will be utilized
        to perform data extraction operations. The parameter specified will directly
        affect performance and resource allocation during these operations.

        :param num_workers: The number of workers to use for extraction.
        :type num_workers: int
        :return: None
        :rtype: None
        """
        self._extraction_num_workers = num_workers

    @property
    def extraction_num_threads_per_worker(self) -> int:
        """
        Retrieves the number of threads allocated for each worker performing the
        extraction process. If the value is not explicitly set, it attempts to
        read the configuration from the environment variable
        'EXTRACTION_NUM_THREADS_PER_WORKER'. If the environment variable is not
        set, defaults to a predefined constant value.

        :return: The number of threads set for each extraction worker.
        :rtype: int
        """
        if self._extraction_num_threads_per_worker is None:
            self.extraction_num_threads_per_worker = int(
                os.environ.get(
                    'EXTRACTION_NUM_THREADS_PER_WORKER',
                    DEFAULT_EXTRACTION_NUM_THREADS_PER_WORKER,
                )
            )

        return self._extraction_num_threads_per_worker

    @extraction_num_threads_per_worker.setter
    def extraction_num_threads_per_worker(self, num_threads: int) -> None:
        """
        Sets the number of threads used per worker during the extraction process.

        This method adjusts the number of threads for each worker responsible
        for the data extraction task.

        :param num_threads: Number of threads to be assigned per worker.
        :type num_threads: int
        :return: None
        """
        self._extraction_num_threads_per_worker = num_threads

    @property
    def extraction_batch_size(self) -> int:
        """
        A property that retrieves the extraction batch size configuration for the application.
        If the value is not already set, it attempts to retrieve it from the environment
        variable 'EXTRACTION_BATCH_SIZE'. If the environment variable is not defined, a
        default value is used.

        :raises ValueError: If the environment variable 'EXTRACTION_BATCH_SIZE' contains
                            a non-integer value or if it cannot be properly parsed.

        :rtype: int
        :return: The batch size to be used for extraction operations. Defaults to the
                 value specified in `DEFAULT_EXTRACTION_BATCH_SIZE` if not set.
        """
        if self._extraction_batch_size is None:
            self.extraction_batch_size = int(
                os.environ.get('EXTRACTION_BATCH_SIZE', DEFAULT_EXTRACTION_BATCH_SIZE)
            )

        return self._extraction_batch_size

    @extraction_batch_size.setter
    def extraction_batch_size(self, batch_size: int) -> None:
        """
        Sets the batch size for extraction operations. This setter method updates the value of
        the private variable ``_extraction_batch_size`` which is responsible for storing the
        batch size to be used during extraction processes.

        :param batch_size: The size of the batch to be used for data extraction operations.
        :type batch_size: int
        :return: None
        """
        self._extraction_batch_size = batch_size

    @property
    def build_num_workers(self) -> int:
        """
        Provides property access to the number of build workers configured in the environment.
        If not already set, this property attempts to retrieve the number of workers from the
        environment variable 'BUILD_NUM_WORKERS'. Defaults to a predefined constant
        (DEFAULT_BUILD_NUM_WORKERS) if the environment variable is not found or invalid.

        :return: The number of build workers as an integer.
        :rtype: int
        """
        if self._build_num_workers is None:
            self.build_num_workers = int(
                os.environ.get('BUILD_NUM_WORKERS', DEFAULT_BUILD_NUM_WORKERS)
            )

        return self._build_num_workers

    @build_num_workers.setter
    def build_num_workers(self, num_workers: int) -> None:
        """
        Sets the number of workers to be used for building processes.

        This function allows the adjustment of the number of workers utilized
        during a build process, enabling control over the parallelization of
        tasks. The setter updates the internal attribute `_build_num_workers`
        to reflect the supplied value.

        :param num_workers: The number of workers to set.
        :type num_workers: int
        """
        self._build_num_workers = num_workers

    @property
    def build_batch_size(self) -> int:
        """
        This property retrieves the batch size used for the build process. If the batch size
        has not been set, it initializes the value by fetching it from the environment
        variable 'BUILD_BATCH_SIZE'. If the environment variable is not set, it defaults
        to 'DEFAULT_BUILD_BATCH_SIZE'.

        :return: The batch size used for the build process.
        :rtype: int
        """
        if self._build_batch_size is None:
            self.build_batch_size = int(
                os.environ.get('BUILD_BATCH_SIZE', DEFAULT_BUILD_BATCH_SIZE)
            )

        return self._build_batch_size

    @build_batch_size.setter
    def build_batch_size(self, batch_size: int) -> None:
        """
        Sets the value of the build batch size. This defines the size of the batch
        used during the build process.

        :param batch_size: The size of the batch for building process.
        :type batch_size: int
        :return: None
        :rtype: None
        """
        self._build_batch_size = batch_size

    @property
    def build_batch_write_size(self) -> int:
        """
        Gets or sets the size of the batch write for the build process.

        This property checks if the `_build_batch_write_size` is `None` and, in such
        a case, sets it to the value fetched from the environment variable
        'BUILD_BATCH_WRITE_SIZE' or a default value (`DEFAULT_BUILD_BATCH_WRITE_SIZE`)
        if the environment variable is not set. This allows dynamic configuration
        of the batch write size during runtime.

        :return: The batch write size used during the build process.
        :rtype: int
        """
        if self._build_batch_write_size is None:
            self.build_batch_write_size = int(
                os.environ.get('BUILD_BATCH_WRITE_SIZE', DEFAULT_BUILD_BATCH_WRITE_SIZE)
            )

        return self._build_batch_write_size

    @build_batch_write_size.setter
    def build_batch_write_size(self, batch_size: int) -> None:
        """
        Sets the batch write size for building operations.

        This property allows the user to configure the size of each batch write
        operation when building data. The batch size directly affects the number
        of records processed in a single operation, enabling optimization for
        performance or resource limitations.

        :param batch_size: The size of each batch to be written during the build
            process.
        :type batch_size: int
        :return: None
        :rtype: None
        """
        self._build_batch_write_size = batch_size

    @property
    def batch_writes_enabled(self) -> bool:
        """
        Determines if batch writes are enabled.

        This property checks the ``_batch_writes_enabled`` attribute. If it is ``None``,
        it attempts to derive the value from the environment variable ``BATCH_WRITES_ENABLED``.
        If the environment variable is not set, a default value is used. The result is cached in
        ``_batch_writes_enabled``.

        :return: Whether batch writes are enabled
        :rtype: bool
        """
        if self._batch_writes_enabled is None:
            self.batch_writes_enabled = string_to_bool(
                os.environ.get('BATCH_WRITES_ENABLED'), DEFAULT_BATCH_WRITES_ENABLED
            )

        return self._batch_writes_enabled

    @batch_writes_enabled.setter
    def batch_writes_enabled(self, batch_writes_enabled: bool) -> None:
        """
        Sets whether batch writes are enabled.

        This property controls if the batch writing feature of the system is
        activated. When enabled, write operations may be grouped and performed
        more efficiently.

        :param batch_writes_enabled: Indicates whether batch writes are enabled.
        :type batch_writes_enabled: bool
        :return: None
        """
        self._batch_writes_enabled = batch_writes_enabled

    @property
    def include_domain_labels(self) -> bool:
        """
        Indicates whether domain labels should be included in processing. This
        property provides lazy initialization by checking the `INCLUDE_DOMAIN_LABELS`
        environment variable. If it is not set, a default value is used defined by
        `DEFAULT_INCLUDE_DOMAIN_LABELS`.

        :raises AttributeError: If the property fails to initialize correctly.
        :return: The current state indicating whether domain labels are included.
        :rtype: bool
        """
        if self._include_domain_labels is None:
            self.include_domain_labels = string_to_bool(
                os.environ.get('INCLUDE_DOMAIN_LABELS'), DEFAULT_INCLUDE_DOMAIN_LABELS
            )
        return self._include_domain_labels

    @include_domain_labels.setter
    def include_domain_labels(self, include_domain_labels: bool) -> None:
        """
        Sets the value of the `include_domain_labels` attribute. This attribute determines
        whether domain labels are included in the specified operation or context where
        this attribute is used.

        :param include_domain_labels: Boolean value to set the attribute. If set to ``True``,
            domain labels will be included. Otherwise, they will be excluded.

        :rtype: None
        """
        self._include_domain_labels = include_domain_labels

    @property
    def enable_cache(self) -> bool:
        """
        A property that determines whether caching is enabled based on an environment
        variable. If the caching setting has not yet been initialized, it reads the
        value from the environment and sets it to a default value if the environment
        variable is not found.

        :raises ValueError: If the environment variable is provided in an invalid format.

        :return: A boolean indicating if caching is enabled.
        :rtype: bool
        """
        if self._enable_cache is None:
            self.enable_cache = string_to_bool(
                os.environ.get('ENABLE_CACHE'), DEFAULT_ENABLE_CACHE
            )
        return self._enable_cache

    @enable_cache.setter
    def enable_cache(self, enable_cache: bool) -> None:
        """
        Sets the cache enabling state for the object.

        This method is the setter function for the ``enable_cache``
        property, allowing the user to enable or disable caching
        depending on the provided value.

        :param enable_cache: Indicates whether caching should
            be enabled or disabled.
        :type enable_cache: bool
        :return: None
        """
        self._enable_cache = enable_cache

    @property
    def metadata_datetime_suffixes(self) -> List[str]:
        """
        Provides access to a list of metadata datetime suffixes. If the suffixes have
        not been explicitly set, a default value is initialized and returned.

        :return: A list of strings containing metadata datetime suffixes.
        :rtype: List[str]
        """
        if self._metadata_datetime_suffixes is None:
            self.metadata_datetime_suffixes = DEFAULT_METADATA_DATETIME_SUFFIXES
        return self._metadata_datetime_suffixes

    @metadata_datetime_suffixes.setter
    def metadata_datetime_suffixes(self, metadata_datetime_suffixes: List[str]) -> None:
        """
        Sets the metadata datetime suffixes for the instance. This property allows
        assigning or updating the list of datetime suffixes related to metadata.
        The setter method updates the internal representation of the suffixes.

        :param metadata_datetime_suffixes: A list of strings representing the
            metadata datetime suffixes.
        :type metadata_datetime_suffixes: List[str]
        """
        self._metadata_datetime_suffixes = metadata_datetime_suffixes

    def _to_llm(self, llm: LLMType):
        """
        Converts the given LLM input to an LLM instance.

        If the input is an instance of the LLM class, it returns the input directly.
        If the input is a JSON string, it parses the string to extract configuration
        details and creates an instance of BedrockConverse with the specified parameters.
        Otherwise, it assumes the input is the model name and creates a BedrockConverse
        instance using default values.

        :param llm: The LLM input which can be an instance of LLM, a JSON string
            with configuration details, or a model name string.
        :type llm: LLMType
        :return: An instance of LLM, specifically a BedrockConverse object unless the input is
            an instance of LLM.
        :rtype: LLM

        :raises ValueError: If there is an error initializing a BedrockConverse instance.
        """
        if isinstance(llm, LLM):
            return llm

        try:
            boto3_session = self.session
            botocore_session = None
            if hasattr(boto3_session, 'get_session'):
                botocore_session = boto3_session.get_session()

            profile = self.aws_profile
            region = self.aws_region

            if _is_json_string(llm):
                config = json.loads(llm)
                return BedrockConverse(
                    model=config['model'],
                    temperature=config.get('temperature', 0.0),
                    max_tokens=config.get('max_tokens', 4096),
                    botocore_session=botocore_session,
                    region_name=config.get('region_name', region),
                    profile_name=config.get('profile_name', profile),
                    max_retries=50,
                )

            else:
                return BedrockConverse(
                    model=llm,
                    temperature=0.0,
                    max_tokens=4096,
                    botocore_session=botocore_session,
                    region_name=region,
                    profile_name=profile,
                    max_retries=50,
                )

        except Exception as e:
            raise ValueError(f'Failed to initialize BedrockConverse: {str(e)}') from e

    @property
    def extraction_llm(self) -> LLM:
        """
        Provides access to the extraction language model (LLM) instance, initializing it
        if not already set. The property retrieves the current LLM instance tied to
        extraction tasks or initializes it with a value from the environment variable
        `EXTRACTION_MODEL`, defaulting to a pre-defined model if the variable is
        unset.

        :raises ValueError: When there is an invalid assignment to the extraction_llm
            attribute during initialization.
        :rtype: LLM
        :return: The current instance of the extraction language model (LLM).
        """
        if self._extraction_llm is None:
            self.extraction_llm = os.environ.get(
                'EXTRACTION_MODEL', DEFAULT_EXTRACTION_MODEL
            )
        return self._extraction_llm

    @extraction_llm.setter
    def extraction_llm(self, llm: LLMType) -> None:
        """
        Sets the `extraction_llm` attribute for the instance. This setter method converts the provided
        parameter to the appropriate LLM format and sets it as an internal attribute. If the LLM instance
        has a `callback_manager` attribute, it is assigned the global callback manager from the settings.

        :param llm: The LLM instance to be set as the `extraction_llm`. It will be processed
            into the appropriate format using the `_to_llm` method.
        :type llm: LLMType
        """

        self._extraction_llm = self._to_llm(llm)
        if hasattr(self._extraction_llm, 'callback_manager'):
            self._extraction_llm.callback_manager = Settings.callback_manager

    @property
    def response_llm(self) -> LLM:
        """
        Returns the LLM object used for responses. If the _response_llm attribute
        is not already set, it initializes the response model using the value from
        the 'RESPONSE_MODEL' environment variable, or defaults to
        DEFAULT_RESPONSE_MODEL if the environment variable is not set.

        :return: The LLM object used for responses.
        :rtype: LLM
        """
        if self._response_llm is None:
            self.response_llm = os.environ.get('RESPONSE_MODEL', DEFAULT_RESPONSE_MODEL)
        return self._response_llm

    @response_llm.setter
    def response_llm(self, llm: LLMType) -> None:
        """
        Sets the response language model (LLM) instance for the object.

        The method assigns a suitable LLM instance to the internal attribute based on the input
        and ensures its compatibility by converting it, if necessary. Additionally, it checks
        if the response LLM contains a callback manager and aligns it with the global settings
        callback manager.

        :param llm: LLM instance or a compatible input that can be converted to an LLM.
        :type llm: LLMType
        :return: None
        :rtype: None
        """

        self._response_llm = self._to_llm(llm)
        if hasattr(self._response_llm, 'callback_manager'):
            self._response_llm.callback_manager = Settings.callback_manager

    @property
    def embed_model(self) -> BaseEmbedding:
        """
        Retrieves the embedding model. If the embedding model has not been set yet, it initializes it by
        trying to retrieve the model name from the 'EMBEDDINGS_MODEL' environment variable. If no environment
        variable is found, it defaults to the pre-defined constant.

        :return: The embedding model instance.
        :rtype: BaseEmbedding
        """
        if self._embed_model is None:
            self.embed_model = os.environ.get(
                'EMBEDDINGS_MODEL', DEFAULT_EMBEDDINGS_MODEL
            )

        return self._embed_model

    @embed_model.setter
    def embed_model(self, embed_model: EmbeddingType) -> None:
        """
        Sets the embedding model for the instance. Handles the initialization of
        the `BedrockEmbedding` model if the provided embedding model is a string,
        allowing configuration through JSON string inputs or direct model names.
        It also handles fallback logic for using existing `boto3` session attributes,
        AWS profile, and region settings. Furthermore, it supports setting a custom
        callback manager if applicable.

        :param embed_model: The embedding model to be set for the instance. This can
            either be an instance of `EmbeddingType` or a string representing the
            model name. JSON string input is supported for specifying model-specific
            configurations.
        :type embed_model: EmbeddingType
        :return: None
        """
        if isinstance(embed_model, str):

            boto3_session = self.session
            botocore_session = None
            if hasattr(boto3_session, 'get_session'):
                botocore_session = boto3_session.get_session()

            profile = self.aws_profile
            region = self.aws_region

            botocore_config = Config(
                retries={"total_max_attempts": 10, "mode": "adaptive"},
                connect_timeout=60.0,
                read_timeout=60.0,
            )

            if _is_json_string(embed_model):
                config = json.loads(embed_model)
                self._embed_model = BedrockEmbedding(
                    model_name=config['model_name'],
                    botocore_session=botocore_session,
                    region_name=config.get('region_name', region),
                    profile_name=config.get('profile_name', profile),
                    botocore_config=botocore_config,
                )
            else:
                self._embed_model = BedrockEmbedding(
                    model_name=embed_model,
                    botocore_session=botocore_session,
                    region_name=region,
                    profile_name=profile,
                    botocore_config=botocore_config,
                )
        else:
            self._embed_model = embed_model

        if hasattr(self._embed_model, 'callback_manager'):
            self._embed_model.callback_manager = Settings.callback_manager

    @property
    def embed_dimensions(self) -> int:
        """
        Gets or sets the embedding dimensions used within the application. Embedding
        dimensions are typically used in machine learning or natural language
        processing models to represent the size of the vector space for embeddings.

        This property retrieves the current value of embedding dimensions or sets it
        based on an environment variable or a predefined default if the dimensions
        are not already set.

        :raise ValueError: If the `EMBEDDINGS_DIMENSIONS` environmental variable is
            set but not a valid integer value.

        :return: An integer indicating the size of embedding dimensions.
        :rtype: int
        """
        if self._embed_dimensions is None:
            self.embed_dimensions = int(
                os.environ.get('EMBEDDINGS_DIMENSIONS', DEFAULT_EMBEDDINGS_DIMENSIONS)
            )

        return self._embed_dimensions

    @embed_dimensions.setter
    def embed_dimensions(self, embed_dimensions: int) -> None:
        """
        Sets the value of the embedding dimensions.

        :param embed_dimensions: The number of dimensions to set for the embedding. This
            value determines the dimensionality of the embedding space and must be an
            integer.
        :return: None
        """
        self._embed_dimensions = embed_dimensions

    @property
    def reranking_model(self) -> str:
        """
        Retrieve the reranking model configuration. If no explicit reranking
        model is set, it attempts to fetch the model name from the environment
        variable 'RERANKING_MODEL'. If the environment variable is not found,
        a default value will be used.

        :return: The name of the reranking model as a string.
        :rtype: str
        """
        if self._reranking_model is None:
            self._reranking_model = os.environ.get(
                'RERANKING_MODEL', DEFAULT_RERANKING_MODEL
            )

        return self._reranking_model

    @reranking_model.setter
    def reranking_model(self, reranking_model: str) -> None:
        """
        Sets the reranking model used for the instance. This method allows for configuring
        the model to be applied when reranking tasks are performed.

        :param reranking_model: The name of the reranking model to be assigned.
        :type reranking_model: str
        :return: None
        """
        self._reranking_model = reranking_model


GraphRAGConfig = _GraphRAGConfig()
