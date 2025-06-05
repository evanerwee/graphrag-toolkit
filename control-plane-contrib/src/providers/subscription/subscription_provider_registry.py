# control-plane-contrib/enrollment/enrollment_provider_registry.py

from typing import Optional, Dict
from control_plane_contrib.enrollment.enrollment_provider_base import EnrollmentProvider
from graphrag_toolkit.lexical_graph.logging import logging

logger = logging.getLogger(__name__)

class EnrollmentProviderRegistry:
    """
    Global registry for managing and retrieving named EnrollmentProvider instances.

    Supports use cases such as multi-tenant lookups, default fallbacks, and dynamic swapping.
    """

    _registry: Dict[str, EnrollmentProvider] = {}
    _default_provider_name: Optional[str] = None

    @classmethod
    def register(cls, name: str, provider: EnrollmentProvider, default: bool = False) -> None:
        """
        Register an enrollment provider under a unique name.

        Args:
            name (str): The unique identifier for the provider (e.g., "us-east", "tenant-acme").
            provider (EnrollmentProvider): The provider instance.
            default (bool): Whether this should be the default provider.
        """
        cls._registry[name] = provider
        if default or cls._default_provider_name is None:
            cls._default_provider_name = name
            logger.info(f"[Enrollment Registry] Set default provider: {name}")
        logger.info(f"[Enrollment Registry] Registered provider: {name}")

    @classmethod
    def get(cls, name: Optional[str] = None) -> Optional[EnrollmentProvider]:
        """
        Retrieve a provider by name or return the default.

        Args:
            name (Optional[str]): The name of the provider.

        Returns:
            Optional[EnrollmentProvider]: The matching provider instance.
        """
        if name:
            return cls._registry.get(name)
        if cls._default_provider_name:
            return cls._registry.get(cls._default_provider_name)
        return None

    @classmethod
    def list_registered(cls) -> Dict[str, EnrollmentProvider]:
        """
        Lists all registered enrollment providers.

        Returns:
            Dict[str, EnrollmentProvider]: Mapping of name to provider.
        """
        return cls._registry.copy()
