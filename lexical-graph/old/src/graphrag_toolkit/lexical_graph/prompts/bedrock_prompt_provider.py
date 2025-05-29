# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Optional
from graphrag_toolkit.lexical_graph.prompts.prompt_provider import PromptProvider
from graphrag_toolkit.lexical_graph.prompts.prompt_provider_config import BedrockPromptProviderConfig
from graphrag_toolkit.lexical_graph.logging import logging

logger = logging.getLogger(__name__)


class BedrockPromptProvider(PromptProvider):
    """
    Loads system and user prompts from AWS Bedrock Agent using prompt ARNs.

    Parameters
    ----------
    config : BedrockPromptProviderConfig
        The configuration object providing AWS clients and region.
    system_prompt_arn : str
        The ARN or short name of the system prompt in Bedrock Agent.
    user_prompt_arn : str
        The ARN or short name of the user prompt in Bedrock Agent.
    system_prompt_version : Optional[str]
        The version of the system prompt (if applicable).
    user_prompt_version : Optional[str]
        The version of the user prompt (if applicable).
    """

    def __init__(
        self,
        config: BedrockPromptProviderConfig,
        system_prompt_arn: str,
        user_prompt_arn: str,
        system_prompt_version: Optional[str] = None,
        user_prompt_version: Optional[str] = None,
    ):
        if not system_prompt_arn or not user_prompt_arn:
            raise ValueError("System and user prompt ARNs must be provided.")

        self.config = config
        self.system_prompt_arn = system_prompt_arn
        self.user_prompt_arn = user_prompt_arn
        self.system_prompt_version = system_prompt_version
        self.user_prompt_version = user_prompt_version

        try:
            self._account_id = self.config.sts.get_caller_identity()["Account"]
        except Exception as e:
            logger.error(f"Unable to retrieve AWS account ID via STS: {str(e)}")
            raise RuntimeError("Failed to initialize BedrockPromptProvider due to STS failure.") from e

    def _resolve_prompt_arn(self, identifier: str) -> str:
        """
        Resolves short prompt name into full ARN if needed.
        """
        if identifier.startswith("arn:aws:bedrock:"):
            return identifier
        return f"arn:aws:bedrock:{self.config.aws_region}:{self._account_id}:prompt/{identifier}"

    def _load_prompt(self, prompt_arn: str, version: Optional[str] = None) -> str:
        """
        Loads a prompt from Bedrock Agent by ARN and optional version.
        """
        try:
            resolved_arn = self._resolve_prompt_arn(prompt_arn)
            kwargs = {"promptIdentifier": resolved_arn}
            if version:
                kwargs["version"] = version

            response = self.config.bedrock.get_prompt(**kwargs)

            variants = response.get("variants", [])
            if not variants:
                raise RuntimeError(f"No variants found for prompt: {resolved_arn}")

            text = variants[0].get("templateConfiguration", {}).get("text", {}).get("text")
            if not text:
                raise RuntimeError(f"Prompt text not found for: {resolved_arn}")

            return text.strip()

        except Exception as e:
            logger.error(f"Failed to load prompt for {prompt_arn}: {str(e)}")
            raise RuntimeError(f"Could not load prompt from Bedrock: {prompt_arn}") from e

    def get_system_prompt(self) -> str:
        return self._load_prompt(self.system_prompt_arn, self.system_prompt_version)

    def get_user_prompt(self) -> str:
        return self._load_prompt(self.user_prompt_arn, self.user_prompt_version)
