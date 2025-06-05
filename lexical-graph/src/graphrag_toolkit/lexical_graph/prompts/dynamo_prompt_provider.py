# SPDX-License-Identifier: Apache-2.0

import json
from graphrag_toolkit.lexical_graph.prompts.prompt_provider_base import PromptProvider
from graphrag_toolkit.lexical_graph.prompts.prompt_provider_config import DynamoDBPromptProviderConfig
from graphrag_toolkit.lexical_graph.logging import logging

logger = logging.getLogger(__name__)

class DynamoDBPromptProvider(PromptProvider):
    """
    Loads system and user prompts from a DynamoDB table using provided configuration.

    Supports both text and JSON prompt formats.
    """

    def __init__(self, config: DynamoDBPromptProviderConfig):
        self.config = config
        self.dynamodb = self.config.dynamodb  # boto3 DynamoDB client from config
        logger.info("[Prompt Debug] Initialized DynamoDBPromptProvider")
        logger.info(f"[Prompt Debug] Table: {config.table_name}, Tenant: {config.tenant_id}")
        logger.info(f"[Prompt Debug] System Key: {config.system_prompt_key}")
        logger.info(f"[Prompt Debug] User Key: {config.user_prompt_key}")
        logger.info(f"[Prompt Debug] Format: {config.format}")

    def _load_prompt(self, key_name: str) -> str | dict:
        """
        Loads a prompt from DynamoDB using tenant ID and prompt key.

        Args:
            key_name: Attribute name for the prompt (system or user).

        Returns:
            str | dict: The loaded prompt content.
        """
        try:
            response = self.dynamodb.get_item(
                TableName=self.config.table_name,
                Key={
                    "tenant_id": {"S": self.config.tenant_id},
                    "prompt_key": {"S": key_name},
                }
            )

            item = response.get("Item")
            if not item or "prompt_content" not in item:
                raise RuntimeError(f"Prompt '{key_name}' not found for tenant '{self.config.tenant_id}'")

            prompt_raw = item["prompt_content"]["S"]

            is_json_format = (
                self.config.format == "json"
                or (self.config.format is None and key_name.lower().endswith(".json"))
            )

            return json.loads(prompt_raw) if is_json_format else prompt_raw.strip()
        except Exception as e:
            logger.error(f"Failed to load prompt '{key_name}' from DynamoDB: {e}")
            raise RuntimeError(f"Could not load prompt from DynamoDB: {key_name}") from e

    def get_system_prompt(self) -> str | dict:
        return self._load_prompt(self.config.system_prompt_key)

    def get_user_prompt(self) -> str | dict:
        return self._load_prompt(self.config.user_prompt_key)
