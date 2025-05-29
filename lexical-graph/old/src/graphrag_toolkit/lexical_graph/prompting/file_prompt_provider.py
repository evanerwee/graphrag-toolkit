# graphrag_toolkit/lexical_graph/prompts/file_prompt_provider.py
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from typing import Optional
from graphrag_toolkit.lexical_graph.prompts.prompt_provider import PromptProvider
from graphrag_toolkit.lexical_graph.prompts.file_prompt_provider_config import FilePromptProviderConfig

class FilePromptProvider(PromptProvider):
    """
    Loads system and user prompts from the local filesystem using a config object.

    Parameters
    ----------
    config : FilePromptProviderConfig
        Configuration containing the prompt directory path.
    """

    def __init__(self, config: FilePromptProviderConfig):
        if not os.path.isdir(config.directory):
            raise NotADirectoryError(f"Invalid or non-existent directory: {config.directory}")
        self.config = config

    def _load_prompt(self, filename: str) -> str:
        path = os.path.join(self.config.directory, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Prompt file not found: {path}")
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read().rstrip()
        except OSError as e:
            raise OSError(f"Failed to read prompt file {path}: {str(e)}") from e

    def get_system_prompt(self) -> str:
        return self._load_prompt("system_prompt.txt")

    def get_user_prompt(self) -> str:
        return self._load_prompt("user_prompt.txt")
