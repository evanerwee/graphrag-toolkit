# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Dict
from graphrag_toolkit.lexical_graph.prompts.prompt_provider_base import PromptProvider
from graphrag_toolkit.lexical_graph.logging import logging

logger = logging.getLogger(__name__)


class PromptProviderRegistry:
    """
    Global registry for managing and retrieving named PromptProvider instances.

    Supports:
    - Named registration of multiple providers (e.g., "aws-prod", "local-dev")
    - A default provider fallback
    """

    _registry: Dict[str, PromptProvider] = {}
    _default_provider_name: Optional[str] = None

    # ─────────────────────────────────────────────
    # Registration
    # ─────────────────────────────────────────────
    @classmethod
    def register(cls, name: str, provider: PromptProvider, default: bool = False) -> None:
        """
        Register a provider under a unique name. Optionally set it as default.

        Args:
            name: Unique name for the provider (e.g., "aws-prod", "local-dev").
            provider: The PromptProvider instance.
            default: If True, set this as the default provider.
        """
        cls._registry[name] = provider
        logger.info(f"[PromptProviderRegistry] Registered provider '{name}'")

        if default or cls._default_provider_name is None:
            cls._default_provider_name = name
            logger.info(f"[PromptProviderRegistry] Set default provider to '{name}'")

    @classmethod
    def force_default(cls, name: str) -> None:
        """
        Forcefully sets the default provider (if it was already registered).

        Args:
            name: Name of the provider to mark as default.
        """
        if name not in cls._registry:
            raise ValueError(f"Cannot set default: provider '{name}' is not registered.")
        cls._default_provider_name = name
        logger.info(f"[PromptProviderRegistry] Forced default provider to '{name}'")

    # ─────────────────────────────────────────────
    # Retrieval
    # ─────────────────────────────────────────────
    @classmethod
    def get(cls, name: Optional[str] = None) -> PromptProvider:
        """
        Get a provider by name, or return the default.

        Args:
            name: Optional name of the registered provider.

        Returns:
            The PromptProvider instance.

        Raises:
            ValueError: If no provider is found and no default is set.
        """
        provider = cls._registry.get(name) if name else cls._registry.get(cls._default_provider_name)

        if provider is None:
            raise ValueError(f"PromptProvider not found: '{name or cls._default_provider_name}'")

        return provider

    # ─────────────────────────────────────────────
    # Listing & Debug
    # ─────────────────────────────────────────────
    @classmethod
    def list_registered(cls) -> Dict[str, PromptProvider]:
        """
        Returns a copy of the provider registry.

        Returns:
            Dict of provider names and their instances.
        """
        return cls._registry.copy()

    @classmethod
    def get_default_name(cls) -> Optional[str]:
        """
        Returns the name of the currently set default provider.

        Returns:
            The default provider name, if any.
        """
        return cls._default_provider_name
