# graphrag_toolkit/lexical_graph/prompts/s3_prompt_provider.py
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from graphrag_toolkit.lexical_graph.prompts.prompt_provider_base import PromptProvider
from graphrag_toolkit.lexical_graph.prompts.prompt_provider_config import S3PromptProviderConfig
from graphrag_toolkit.lexical_graph.logging import logging

logger = logging.getLogger(__name__)

class S3PromptProvider(PromptProvider):
    def __init__(self, config: S3PromptProviderConfig):
        self.config = config

    def _load_prompt(self, filename: str) -> str:
        """
        Loads a prompt file from the configured S3 bucket and returns its contents as a string.

        Args:
            filename: The name of the prompt file to load from S3.

        Returns:
            The contents of the prompt file as a string.
        """
        key = f"{self.config.prefix.rstrip('/')}/{filename}"
        logger.info(f"[Prompt Debug] Loading prompt from S3: s3://{self.config.bucket}/{key}")
        response = self.config.s3.get_object(Bucket=self.config.bucket, Key=key)
        return response["Body"].read().decode("utf-8").rstrip()

    def get_system_prompt(self) -> str:
        return self._load_prompt("system_prompt.txt")

    def get_user_prompt(self) -> str:
        return self._load_prompt("user_prompt.txt")
