# graphrag_toolkit/lexical_graph/prompts/bedrock_prompt_provider.py
# SPDX-License-Identifier: Apache-2.0

from graphrag_toolkit.lexical_graph.prompts.prompt_provider_base import PromptProvider
from graphrag_toolkit.lexical_graph.prompts.prompt_provider_config import BedrockPromptProviderConfig
from graphrag_toolkit.lexical_graph.logging import logging

logger = logging.getLogger(__name__)

class BedrockPromptProvider(PromptProvider):
    def __init__(self, config: BedrockPromptProviderConfig):
        self.config = config

    def _load_prompt(self, prompt_arn: str, version: str = None) -> str:
        try:
            kwargs = {"promptIdentifier": prompt_arn}
            if version:
                kwargs["version"] = version

            response = self.config.bedrock.get_prompt(**kwargs)

            variants = response.get("variants", [])
            if not variants:
                raise RuntimeError(f"No variants found for prompt: {prompt_arn}")

            text = variants[0].get("templateConfiguration", {}).get("text", {}).get("text")
            if not text:
                raise RuntimeError(f"Prompt text not found for: {prompt_arn}")

            return text.strip()

        except Exception as e:
            logger.error(f"Failed to load prompt for {prompt_arn}: {str(e)}")
            raise RuntimeError(f"Could not load prompt from Bedrock: {prompt_arn}") from e

    def get_system_prompt(self) -> str:
        return self._load_prompt(
            self.config.resolved_system_prompt_arn,
            self.config.system_prompt_version,
        )

    def get_user_prompt(self) -> str:
        return self._load_prompt(
            self.config.resolved_user_prompt_arn,
            self.config.user_prompt_version,
        )
