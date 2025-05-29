# graphrag_toolkit/lexical_graph/prompts/prompt_provider_factory.py
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from graphrag_toolkit.lexical_graph.prompts.prompt_provider import PromptProvider
from graphrag_toolkit.lexical_graph.prompts.prompt_provider_config import (
    BedrockPromptProviderConfig,
    S3PromptProviderConfig,
    FilePromptProviderConfig,
)


class PromptProviderFactory:
    """
    Factory to select and build the appropriate PromptProvider
    based on the PROMPT_PROVIDER environment variable.
    """

    @staticmethod
    def get_provider() -> PromptProvider:
        provider_type = os.getenv("PROMPT_PROVIDER", "file").lower()

        if provider_type == "bedrock":
            return BedrockPromptProviderConfig().build()

        elif provider_type == "s3":
            return S3PromptProviderConfig().build()

        else:  # default to filesystem
            return FilePromptProviderConfig().build()
