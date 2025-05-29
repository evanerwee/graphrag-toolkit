import os
from graphrag_toolkit.lexical_graph.prompts.prompt_provider_base import PromptProvider
from graphrag_toolkit.lexical_graph.prompts.prompt_provider_config import FilePromptProviderConfig
from graphrag_toolkit.lexical_graph.logging import logging

logger = logging.getLogger(__name__)

class FilePromptProvider(PromptProvider):
    """
    Loads system and user prompts from the local filesystem using a config object.
    """

    def __init__(self, config: FilePromptProviderConfig, system_prompt_file: str = "system_prompt.txt", user_prompt_file: str = "user_prompt.txt"):
        if not os.path.isdir(config.base_path):
            raise NotADirectoryError(f"Invalid or non-existent directory: {config.base_path}")
        self.config = config
        self.system_prompt_file = system_prompt_file
        self.user_prompt_file = user_prompt_file

        logger.info(f"[Prompt Debug] Initialized FilePromptProvider")
        logger.info(f"[Prompt Debug] Base path: {self.config.base_path}")
        logger.info(f"[Prompt Debug] System prompt file: {self.system_prompt_file}")
        logger.info(f"[Prompt Debug] User prompt file: {self.user_prompt_file}")

    def _load_prompt(self, filename: str) -> str:
        path = os.path.join(self.config.base_path, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Prompt file not found: {path}")
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read().rstrip()
        except OSError as e:
            raise OSError(f"Failed to read prompt file {path}: {str(e)}") from e

    def get_system_prompt(self) -> str:
        return self._load_prompt(self.system_prompt_file)

    def get_user_prompt(self) -> str:
        return self._load_prompt(self.user_prompt_file)
