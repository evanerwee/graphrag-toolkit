# graphrag_toolkit/lexical_graph/prompts/file_prompt_provider.py
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import boto3
from graphrag_toolkit.lexical_graph.prompts.prompt_provider import (
    BedrockPromptProvider,
    S3PromptProvider,
    FileSystemPromptProvider,
    PromptProvider,
)

class PromptProviderFactory:
    def __init__(self, session: boto3.Session = None, aws_region: str = None):
        """
        Initializes the factory with an optional session and AWS region.
        If not provided, it will use environment variables or default values.
        """
        self.session = session or boto3.Session()
        self.aws_region = aws_region or os.getenv("AWS_REGION", "us-east-1")

    def get_provider(self) -> PromptProvider:
        """
        Selects and returns the appropriate PromptProvider based on environment variables.
        """
        provider_type = os.getenv("PROMPT_PROVIDER", "filesystem").lower()

        if provider_type == "bedrock":
            return BedrockPromptProvider(
                session=self.session,
                aws_region=self.aws_region,
            )

        elif provider_type == "s3":
            bucket = os.environ["PROMPT_S3_BUCKET"]
            prefix = os.getenv("PROMPT_S3_PREFIX", "prompts/")
            s3_client = self.session.client("s3", region_name=self.aws_region)
            return S3PromptProvider(
                s3_client=s3_client,
                bucket=bucket,
                prefix=prefix,
            )

        else:
            base_path = os.getenv("PROMPT_PATH", "./prompts")
            return FileSystemPromptProvider(base_path=base_path)
