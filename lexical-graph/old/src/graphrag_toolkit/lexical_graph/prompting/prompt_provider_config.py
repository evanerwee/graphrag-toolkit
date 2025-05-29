# graphrag_toolkit/lexical_graph/prompts/prompt_provider_config.py

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Dict, Set, Any
import os
import boto3
from boto3.session import Session as Boto3Session

from graphrag_toolkit.lexical_graph.prompts.prompt_provider import PromptProvider
from graphrag_toolkit.lexical_graph.prompts.s3_prompt_provider import S3PromptProvider
from graphrag_toolkit.lexical_graph.prompts.bedrock_prompt_provider import BedrockPromptProvider
from graphrag_toolkit.lexical_graph.prompts.file_prompt_provider import FilePromptProvider


# ------------------------------------------------------------------------------
# AWSConfig – base for config classes needing AWS session/client support
# ------------------------------------------------------------------------------
@dataclass
class AWSConfig(ABC):
    _aws_profile: Optional[str] = None
    _aws_region: Optional[str] = None
    _aws_clients: Dict[str, Any] = field(default_factory=dict)

    _boto3_session: Optional[boto3.Session] = field(default=None, init=False)

    @property
    def session(self) -> Boto3Session:
        if self._boto3_session is None:
            if self.aws_profile:
                self._boto3_session = Boto3Session(
                    profile_name=self.aws_profile,
                    region_name=self.aws_region,
                )
            else:
                self._boto3_session = Boto3Session(region_name=self.aws_region)
        return self._boto3_session

    def _get_or_create_client(self, service_name: str) -> Any:
        if service_name in self._aws_clients:
            return self._aws_clients[service_name]
        client = self.session.client(service_name)
        self._aws_clients[service_name] = client
        return client

    @property
    def aws_profile(self) -> Optional[str]:
        if self._aws_profile is None:
            self._aws_profile = os.environ.get("AWS_PROFILE")
        return self._aws_profile

    @property
    def aws_region(self) -> str:
        if self._aws_region is None:
            self._aws_region = os.environ.get("AWS_REGION", self.session.region_name)
        return self._aws_region

    @property
    def s3(self) -> Any:
        return self._get_or_create_client("s3")

    @property
    def bedrock(self) -> Any:
        return self._get_or_create_client("bedrock-agent")

    @property
    def sts(self) -> Any:
        return self._get_or_create_client("sts")


# ------------------------------------------------------------------------------
# ProviderConfig – abstract interface for building PromptProviders
# ------------------------------------------------------------------------------
@dataclass
class ProviderConfig(AWSConfig):
    @abstractmethod
    def build(self) -> PromptProvider:
        pass


# ------------------------------------------------------------------------------
# BedrockPromptProviderConfig
# ------------------------------------------------------------------------------
@dataclass
class BedrockPromptProviderConfig(ProviderConfig):
    system_prompt_arn: str = field(default_factory=lambda: os.environ["SYSTEM_PROMPT_ARN"])
    user_prompt_arn: str = field(default_factory=lambda: os.environ["USER_PROMPT_ARN"])
    system_prompt_version: Optional[str] = field(default_factory=lambda: os.getenv("SYSTEM_PROMPT_VERSION"))
    user_prompt_version: Optional[str] = field(default_factory=lambda: os.getenv("USER_PROMPT_VERSION"))

    def build(self) -> PromptProvider:
        return BedrockPromptProvider(
            config=self,
            system_prompt_arn=self.system_prompt_arn,
            user_prompt_arn=self.user_prompt_arn,
            system_prompt_version=self.system_prompt_version,
            user_prompt_version=self.user_prompt_version,
        )


# ------------------------------------------------------------------------------
# S3PromptProviderConfig
# ------------------------------------------------------------------------------
@dataclass
class S3PromptProviderConfig(ProviderConfig):
    bucket: str = field(default_factory=lambda: os.environ["PROMPT_S3_BUCKET"])
    prefix: str = field(default_factory=lambda: os.getenv("PROMPT_S3_PREFIX", "prompts/"))

    def build(self) -> PromptProvider:
        return S3PromptProvider(
            config=self,
            bucket=self.bucket,
            prefix=self.prefix.rstrip("/") + "/"
        )


# ------------------------------------------------------------------------------
# FilePromptProviderConfig
# ------------------------------------------------------------------------------
@dataclass
class FilePromptProviderConfig(ProviderConfig):
    base_path: str = field(default_factory=lambda: os.getenv("PROMPT_PATH", "./prompts"))

    def build(self) -> PromptProvider:
        return FilePromptProvider(directory=self.base_path)
