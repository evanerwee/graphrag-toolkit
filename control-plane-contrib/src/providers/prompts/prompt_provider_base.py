# graphrag_toolkit/lexical_graph/prompts/prompt_provider_base.py

from abc import ABC, abstractmethod
from typing import Union, Dict, Optional
from graphrag_toolkit.lexical_graph.logging import logging

logger = logging.getLogger(__name__)

class PromptProvider(ABC):
    """
    Abstract base class for loading prompts from various sources.
    Supports both text and JSON formats in Prompts V2.
    """

    @abstractmethod
    def get_system_prompt(self) -> str:
        """
        Returns the system prompt as a plain string.
        Default fallback if format is not JSON.
        """
        pass

    @abstractmethod
    def get_user_prompt(self) -> str:
        """
        Returns the user prompt as a plain string.
        Default fallback if format is not JSON.
        """
        pass

    def get_system_prompt_object(self) -> Optional[Union[Dict, str]]:
        """
        Optional: Returns the system prompt as a parsed object (JSON or dict).
        Should be overridden by providers that support JSON format.
        """
        logger.debug("Default get_system_prompt_object() used; falling back to get_system_prompt()")
        return self.get_system_prompt()

    def get_user_prompt_object(self) -> Optional[Union[Dict, str]]:
        """
        Optional: Returns the user prompt as a parsed object (JSON or dict).
        Should be overridden by providers that support JSON format.
        """
        logger.debug("Default get_user_prompt_object() used; falling back to get_user_prompt()")
        return self.get_user_prompt()
