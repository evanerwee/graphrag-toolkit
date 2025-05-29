# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import os
import boto3
from boto3.session import Session as Boto3Session

from graphrag_toolkit.lexical_graph.prompts.prompt_provider_base import PromptProvider
from graphrag_toolkit.lexical_graph.logging import logging

logger = logging.getLogger(__name__)

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

    @aws_profile.setter
    def aws_profile(self, value: Optional[str]):
        self._aws_profile = value

    @property
    def aws_region(self) -> str:
        if self._aws_region is None:
            self._aws_region = os.environ.get("AWS_REGION", self.session.region_name)
        return self._aws_region

    @aws_region.setter
    def aws_region(self, value: Optional[str]):
        self._aws_region = value

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

    @property
    def resolved_system_prompt_arn(self) -> str:
        return self._resolve_prompt_arn(self.system_prompt_arn)

    @property
    def resolved_user_prompt_arn(self) -> str:
        return self._resolve_prompt_arn(self.user_prompt_arn)

    def _resolve_prompt_arn(self, identifier: str) -> str:
        if identifier.startswith("arn:aws:bedrock:"):
            return identifier

        account_id = self.sts.get_caller_identity()["Account"]
        return f"arn:aws:bedrock:{self.aws_region}:{account_id}:prompt/{identifier}"

    def build(self) -> PromptProvider:
        from graphrag_toolkit.lexical_graph.prompts.bedrock_prompt_provider import BedrockPromptProvider
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
        from graphrag_toolkit.lexical_graph.prompts.s3_prompt_provider import S3PromptProvider
        return S3PromptProvider(config=self)


# ------------------------------------------------------------------------------
# FilePromptProviderConfig
# ------------------------------------------------------------------------------
@dataclass
class FilePromptProviderConfig(ProviderConfig):
    base_path: str = field(default_factory=lambda: os.getenv("PROMPT_PATH", "./prompts"))
    system_prompt_file: str = "system_prompt.txt"
    user_prompt_file: str = "user_prompt.txt"

    def build(self) -> PromptProvider:
        from graphrag_toolkit.lexical_graph.prompts.file_prompt_provider import FilePromptProvider
        return FilePromptProvider(
            config=self,
            system_prompt_file=self.system_prompt_file,
            user_prompt_file=self.user_prompt_file
        )


# ------------------------------------------------------------------------------
# StaticPromptProviderConfig
# ------------------------------------------------------------------------------
@dataclass
class StaticPromptProviderConfig:
    @staticmethod
    def build() -> PromptProvider:
        from graphrag_toolkit.lexical_graph.prompts.static_prompt_provider import StaticPromptProvider
        return StaticPromptProvider()


# ------------------------------------------------------------------------------
# Suggested Next Enhancements (Optional)
# ------------------------------------------------------------------------------

# 1. Unified PromptProviderRegistry or Factory
#    - Introduce a registry that maps provider types to config classes, e.g.,
#        registry = {
#            "static": StaticPromptProviderConfig,
#            "file": FilePromptProviderConfig,
#            "s3": S3PromptProviderConfig,
#            "bedrock": BedrockPromptProviderConfig
#        }
#    - Enable initialization from a config dict: registry[type](**params).build()

# 2. Config Serialization
#    - Add `.to_dict()` and `.from_dict()` methods to each config class for CLI/JSON compatibility.
#    - Useful for web UIs or YAML-driven orchestration.

# 3. Validation & Type Enforcement
#    - Use Pydantic or `__post_init__()` methods to validate inputs (e.g., ARN format, S3 bucket name).
#    - Example: validate AWS region format or prompt ARN prefix.

# 4. Logging Enhancements
#    - Add verbose logging on each provider (e.g., which prompt path or ARN was loaded).
#    - Include diagnostics for STS calls and client creation failures.

# 5. Caching Layer
#    - Cache resolved prompt text in memory or on disk (especially for S3 and Bedrock).
#    - Avoid unnecessary repeated fetches in batch queries.

# 6. Runtime Provider Switching
#    - Allow query-time override of prompt provider (e.g., via `query_engine.query(..., prompt_provider=...)`).
#    - Enables experimentation with different prompt strategies.

# 7. Prompt Fallback Strategy
#    - Support fallback to defaults or static provider if S3/Bedrock fails.
#    - Enables robust operation in partially degraded environments.

# 8. Custom Prompt Variables
#    - Support variable interpolation in prompt templates (e.g., using `{tenant_id}` or `{user_role}`).
#    - Useful for multi-tenant or role-specific prompting.

# 9. Multi-Language Prompt Support
#    - Load prompt variants based on locale/language code.
#    - Supports internationalization of RAG applications.

# 10. Bedrock Caching with Prompt Versioning
#     - Cache based on (ARN, version) tuple.
#     - Useful when managing multiple versions in experiments or A/B testing.



