# graphrag_toolkit/lexical_graph/prompts/s3_prompt_provider.py
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from graphrag_toolkit.lexical_graph.prompts.prompt_provider_base import PromptProvider
from graphrag_toolkit.lexical_graph.prompts.prompt_provider_config import S3PromptProviderConfig
from graphrag_toolkit.lexical_graph.logging import logging

logger = logging.getLogger(__name__)

class S3PromptProvider(PromptProvider):
    """
    Loads system and user prompts from an S3 bucket using provided configuration.

    Supports both text and JSON prompt formats.
    """

    def __init__(self, config: S3PromptProviderConfig):
        self.config = config
        logger.info(f"[Prompt Debug] Initialized S3PromptProvider")
        logger.info(f"[Prompt Debug] Bucket: {config.bucket}")
        logger.info(f"[Prompt Debug] Prefix: {config.prefix}")
        logger.info(f"[Prompt Debug] Format: {config.format}")
        logger.info(f"[Prompt Debug] System file: {config.system_prompt_file}")
        logger.info(f"[Prompt Debug] User file: {config.user_prompt_file}")

    def _load_prompt(self, filename: str) -> str | dict:
        """
        Loads a prompt file from the configured S3 bucket.

        Args:
            filename: The name of the prompt file to load from S3.

        Returns:
            str | dict: The contents of the prompt file.
        """
        key = f"{self.config.prefix.rstrip('/')}/{filename}"
        logger.info(f"[Prompt Debug] Loading prompt from S3: s3://{self.config.bucket}/{key}")
        s3_client = self.config.s3

        try:
            response = s3_client.get_object(Bucket=self.config.bucket, Key=key)
            body = response["Body"].read().decode("utf-8")

            # Determine format: config takes priority, fallback to extension
            fmt = self.config.format or ("json" if filename.endswith(".json") else "text")

            return json.loads(body) if fmt == "json" else body.strip()
        except Exception as e:
            logger.error(f"Failed to load prompt from S3: s3://{self.config.bucket}/{key} - {e}")
            raise RuntimeError(f"Could not load prompt file from S3: {key}") from e

    def get_system_prompt(self) -> str | dict:
        return self._load_prompt(self.config.system_prompt_file)

    def get_user_prompt(self) -> str | dict:
        return self._load_prompt(self.config.user_prompt_file)
