# graphrag_toolkit/lexical_graph/prompts/bedrock_prompt_provider.py
# SPDX-License-Identifier: Apache-2.0

from graphrag_toolkit.lexical_graph.prompts.prompt_provider_base import PromptProvider
from graphrag_toolkit.lexical_graph.prompts.prompt_provider_config import BedrockPromptProviderConfig
from graphrag_toolkit.lexical_graph.logging import logging

logger = logging.getLogger(__name__)

class BedrockPromptProvider(PromptProvider):
    """Provides prompt templates from AWS Bedrock using specified ARNs and versions.
    
    This class loads and returns system and user prompt templates from AWS Bedrock, 
    based on configuration provided at initialization.
    """

    def __init__(self, config: BedrockPromptProviderConfig):
        self.config = config

        logger.info(
            f"[Prompt Debug] Using BedrockPromptProvider with:\n"
            f"  system_prompt_arn={config.system_prompt_arn} "
            f"(resolved={config.resolved_system_prompt_arn}, version={config.system_prompt_version})\n"
            f"  user_prompt_arn={config.user_prompt_arn} "
            f"(resolved={config.resolved_user_prompt_arn}, version={config.user_prompt_version})\n"
            f"  region={config.aws_region}, profile={config.aws_profile}"
        )

    def _load_prompt(self, prompt_arn: str, version: str = None) -> str | dict:
        try:
            kwargs = {"promptIdentifier": prompt_arn}
            if version:
                kwargs["promptVersion"] = version

            response = self.config.bedrock.get_prompt(**kwargs)
            variants = response.get("variants", [])
            if not variants:
                raise RuntimeError(f"No variants found for prompt: {prompt_arn}")

            template_config = variants[0].get("templateConfiguration", {}).get("text", {})

            # Support both text and json formats
            if self.config.format == "json":
                json_payload = template_config.get("json")
                if not json_payload:
                    raise RuntimeError(f"JSON prompt not found for: {prompt_arn}")
                return json_payload
            else:
                text = template_config.get("text")
                if not text:
                    raise RuntimeError(f"Text prompt not found for: {prompt_arn}")
                return text.strip()

        except Exception as e:
            logger.error(f"Failed to load prompt for {prompt_arn}: {str(e)}")
            raise RuntimeError(f"Could not load prompt from Bedrock: {prompt_arn}") from e

    def get_system_prompt(self) -> str | dict:
        """Retrieves the system prompt template from AWS Bedrock.

        Returns:
            The text of the system prompt template.
        """
        return self._load_prompt(
            self.config.resolved_system_prompt_arn,
            self.config.system_prompt_version,
        )

    def get_user_prompt(self) -> str | dict:
        """Retrieves the user prompt template from AWS Bedrock.

        Returns:
            The text of the user prompt template.
        """
        return self._load_prompt(
            self.config.resolved_user_prompt_arn,
            self.config.user_prompt_version,
        )
