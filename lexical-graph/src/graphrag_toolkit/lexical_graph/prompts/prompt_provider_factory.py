# graphrag_toolkit/lexical_graph/prompts/prompt_provider_factory.py

import os
from graphrag_toolkit.lexical_graph.prompts.prompt_provider_base import PromptProvider
from graphrag_toolkit.lexical_graph.prompts.prompt_provider_config import (
    BedrockPromptProviderConfig,
    S3PromptProviderConfig,
    FilePromptProviderConfig,
    StaticPromptProviderConfig,  # ✅ Correct import
)
from graphrag_toolkit.lexical_graph.logging import logging

logger = logging.getLogger(__name__)


class PromptProviderFactory:
    @staticmethod
    def get_provider() -> PromptProvider:
        provider_type = os.getenv("PROMPT_PROVIDER", "static").lower()

        if provider_type == "bedrock":
            return BedrockPromptProviderConfig().build()
        elif provider_type == "s3":
            return S3PromptProviderConfig().build()
        elif provider_type == "file":
            return FilePromptProviderConfig().build()
        else:
            # Final fallback to static default prompts
            return StaticPromptProvider()
