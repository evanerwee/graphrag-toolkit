# graphrag_toolkit/lexical_graph/prompts/prompt_provider_factory.py

import os
from graphrag_toolkit.lexical_graph.prompts.prompt_provider_base import PromptProvider
from graphrag_toolkit.lexical_graph.prompts.prompt_provider_config import (
    BedrockPromptProviderConfig,
    S3PromptProviderConfig,
    FilePromptProviderConfig,
    StaticPromptProviderConfig,
    DynamoDBPromptProviderConfig
)
from graphrag_toolkit.lexical_graph.logging import logging

logger = logging.getLogger(__name__)


class PromptProviderFactory:
    """
    Factory class for creating PromptProvider instances based on environment configuration.

    This class selects and builds the appropriate PromptProvider implementation according to the PROMPT_PROVIDER environment variable.
    """
    @staticmethod
    def get_provider() -> PromptProvider:
        """
        Returns a PromptProvider instance based on the PROMPT_PROVIDER environment variable.

        This method selects and builds the appropriate PromptProvider implementation for Bedrock, S3, file, or static sources.

        Returns:
            PromptProvider: An instance of the selected PromptProvider implementation.
        """
        provider_type = os.getenv("PROMPT_PROVIDER", "static").lower()

        if provider_type == "bedrock":
            return BedrockPromptProviderConfig().build()
        elif provider_type == "s3":
            return S3PromptProviderConfig().build()
        elif provider_type == "file":
            return FilePromptProviderConfig().build()
        elif provider_type == "dynamodb":
            return DynamoDBPromptProviderConfig().build()
        else:
            # Final fallback to static default prompts
            return StaticPromptProviderConfig().build()

    @staticmethod
    def from_dict(config: dict) -> PromptProvider:
        provider_type = config.pop("provider_type", "static").lower()

        if provider_type == "bedrock":
            return BedrockPromptProviderConfig(**config).build()
        elif provider_type == "s3":
            return S3PromptProviderConfig(**config).build()
        elif provider_type == "file":
            return FilePromptProviderConfig(**config).build()
        elif provider_type == "dynamodb":
            return DynamoDBPromptProviderConfig(**config).build()
        elif provider_type == "static":
            return StaticPromptProviderConfig.build()
        else:
            raise ValueError(f"Unsupported provider_type: {provider_type}")
