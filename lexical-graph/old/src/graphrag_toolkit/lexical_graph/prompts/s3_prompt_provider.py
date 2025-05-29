# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from graphrag_toolkit.lexical_graph.prompts.file_prompt_provider import PromptProvider
from graphrag_toolkit.lexical_graph.prompts.prompt_provider_config import S3PromptProviderConfig

class S3PromptProvider(PromptProvider):
    def __init__(self, config: S3PromptProviderConfig):
        self.config = config

    def _load_prompt(self, filename: str) -> str:
        key = f"{self.config.prefix.rstrip('/')}/{filename}"
        response = self.config.s3.get_object(Bucket=self.config.bucket, Key=key)
        return response["Body"].read().decode("utf-8").rstrip()

    def get_system_prompt(self) -> str:
        return self._load_prompt("system_prompt.txt")

    def get_user_prompt(self) -> str:
        return self._load_prompt("user_prompt.txt")
